"""
Face Vector Nodes for ComfyUI

This module provides three nodes:
1. ExtractFaceVector - Extract face embeddings from images using InsightFace
2. SaveFaceVector - Save face vectors to disk for later use
3. LoadFaceVector - Load previously saved face vectors
"""

import torch
import os
import folder_paths
import comfy.model_management as model_management


def insightface_loader(provider):
    """Load InsightFace model."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise Exception(f"InsightFace is not installed. Install it with: pip install insightface\nOriginal error: {e}")

    path = os.path.join(folder_paths.models_dir, "insightface")
    model = FaceAnalysis(name="buffalo_l", root=path, providers=[provider + 'ExecutionProvider',])
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model


def tensor_to_image(tensor):
    """Convert ComfyUI image tensor to numpy array for InsightFace."""
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image


class ExtractFaceVector:
    """
    Extract face vector (embedding) from an image using InsightFace.
    Returns a tensor that can be used later for face generation.
    Also returns vectors as JSON in API response.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "insightface": ("INSIGHTFACE",),
                "provider": (["CPU", "CUDA", "ROCM"], {"default": "CPU"}),
                "use_normed": ("BOOLEAN", {"default": True, "label_on": "Normed", "label_off": "Raw"}),
            }
        }

    RETURN_TYPES = ()  # Empty to allow UI data return for OUTPUT_NODE
    RETURN_NAMES = ()
    FUNCTION = "extract_face_vector"
    OUTPUT_NODE = True
    CATEGORY = "vector_face"
    DESCRIPTION = "Extract face embedding vector from image using InsightFace. Returns vectors in API response."
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution to ensure fresh vectors
        return float("nan")

    def extract_face_vector(self, image, insightface=None, provider="CPU", use_normed=True):
        """
        Extract face embedding vector from image.
        
        Args:
            image: Input image tensor (B, H, W, C) in range [0, 1]
            insightface: Optional InsightFace model (if None, will load one)
            provider: Execution provider for InsightFace ("CPU", "CUDA", "ROCM")
            use_normed: If True, use normed_embedding (standard). If False, use raw embedding.
        
        Returns:
            Tuple: (face_vector_tensor, ui_dict) - Face vector tensor and UI data for API
        """
        # Load InsightFace model if not provided
        if insightface is None:
            insightface = insightface_loader(provider)
        
        # Convert image tensor to numpy array for InsightFace
        image_iface = tensor_to_image(image)
        
        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if model_management.should_use_fp16() else torch.float32
        
        face_embeds = []
        
        # Reset detection size
        insightface.det_model.input_size = (640, 640)
        
        # Process each image in the batch
        for i in range(image_iface.shape[0]):
            face = None
            detection_size = None
            
            # Try different detection sizes if face is not detected at 640x640
            for size in range(640, 256, -64):
                insightface.det_model.input_size = (size, size)
                face = insightface.get(image_iface[i])
                if face:
                    detection_size = size
                    break
            
            if not face:
                raise Exception(f'InsightFace: No face detected in image {i+1} of batch.')
            
            if detection_size != 640:
                print(f"\033[33mINFO: InsightFace detection resolution lowered to {detection_size}.\033[0m")
            
            # Extract embedding
            if use_normed:
                # Standard normalized embedding (used for most FaceID models)
                embedding = face[0].normed_embedding
            else:
                # Raw embedding (used for portrait unnorm models)
                embedding = face[0].embedding
            
            face_embeds.append(torch.from_numpy(embedding).unsqueeze(0))
        
        # Stack all face embeddings into a batch tensor
        face_embeds_tensor = torch.stack(face_embeds).to(device, dtype=dtype)
        
        # Convert to list for JSON serialization
        face_embeds_cpu = face_embeds_tensor.cpu()
        vectors_array = []
        for i in range(face_embeds_cpu.shape[0]):
            vector_list = face_embeds_cpu[i].tolist()
            vectors_array.append(vector_list)
        
        # For OUTPUT_NODE with RETURN_TYPES = (), ComfyUI extracts UI data from dict return
        # This format allows vectors to appear in API response
        # Note: RETURN_TYPES = () means this node can't be chained to other nodes
        # Use SaveFaceVector if you need to chain ExtractFaceVector to other nodes
        return {
            "ui": {
                "vectors": vectors_array  # Array of vectors, each is a list of 512 floats
            }
        }


class SaveFaceVector:
    """
    Save face vector to a file for later retrieval via API or loading.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_vector": ("FACE_VECTOR",),
                "filename_prefix": ("STRING", {"default": "face_vector"})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "vector_face"
    DESCRIPTION = "Save face vector to disk for later use"

    def save(self, face_vector, filename_prefix):
        # Get batch size
        batch_size = face_vector.shape[0] if len(face_vector.shape) > 1 else 1
        
        # Convert to CPU and numpy for JSON serialization
        face_vector_cpu = face_vector.cpu()
        
        # Get base path for saving files
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        
        # Save one file per face vector in the batch
        saved_files = []
        vectors_array = []
        
        for i in range(batch_size):
            # Extract single face vector from batch
            if batch_size == 1:
                single_vector = face_vector_cpu
            else:
                single_vector = face_vector_cpu[i:i+1]  # Keep batch dimension for consistency
            
            # Generate filename with index for batch items
            if batch_size > 1:
                file = f"{filename}_{counter:05}_{i:04d}.facevec"
            else:
                file = f"{filename}_{counter:05}.facevec"
            
            file_path = os.path.join(full_output_folder, file)
            
            # Save individual face vector
            torch.save(single_vector, file_path)
            saved_files.append({
                "filename": file,
                "subfolder": subfolder if subfolder else "",
                "type": "output"
            })
            
            # Convert to list for JSON serialization (remove batch dimension)
            vector_list = single_vector.squeeze(0).tolist() if single_vector.shape[0] == 1 else single_vector.tolist()
            vectors_array.append(vector_list)
        
        if batch_size > 1:
            print(f"\033[33mINFO: Saved {batch_size} face vector(s) to {full_output_folder}\033[0m")
        else:
            print(f"\033[33mINFO: Face vector saved to {file_path}\033[0m")
        
        # Return file information AND vector data for API response
        # OUTPUT_NODE nodes return a dict with UI info
        # Note: ComfyUI expects this format - the "ui" dict is extracted automatically
        return {
            "ui": {
                "face_vectors": saved_files,
                "vectors": vectors_array  # Array of vectors, each is a list of 512 floats
            }
        }


class LoadFaceVector:
    """
    Load a previously saved face vector from file.
    """
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        file_list = []
        for root, dirs, filenames in os.walk(input_dir):
            for filename in filenames:
                if filename.endswith('.facevec'):
                    file_list.append(os.path.relpath(os.path.join(root, filename), input_dir))
        return {"required": {"face_vector": [sorted(file_list), ]}, }

    RETURN_TYPES = ("FACE_VECTOR",)
    RETURN_NAMES = ("face_vector",)
    FUNCTION = "load"
    CATEGORY = "vector_face"
    DESCRIPTION = "Load a previously saved face vector from disk"

    def load(self, face_vector):
        path = folder_paths.get_annotated_filepath(face_vector)
        return (torch.load(path, map_location="cpu"),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ExtractFaceVector": ExtractFaceVector,
    "SaveFaceVector": SaveFaceVector,
    "LoadFaceVector": LoadFaceVector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractFaceVector": "Extract Face Vector",
    "SaveFaceVector": "Save Face Vector",
    "LoadFaceVector": "Load Face Vector",
}

