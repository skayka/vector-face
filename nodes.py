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

    RETURN_TYPES = ("FACE_VECTOR",)
    RETURN_NAMES = ("face_vector",)
    FUNCTION = "extract_face_vector"
    CATEGORY = "vector_face"
    DESCRIPTION = "Extract face embedding vector from image using InsightFace"

    def extract_face_vector(self, image, insightface=None, provider="CPU", use_normed=True):
        """
        Extract face embedding vector from image.
        
        Args:
            image: Input image tensor (B, H, W, C) in range [0, 1]
            insightface: Optional InsightFace model (if None, will load one)
            provider: Execution provider for InsightFace ("CPU", "CUDA", "ROCM")
            use_normed: If True, use normed_embedding (standard). If False, use raw embedding.
        
        Returns:
            Face vector tensor (B, 512) - batch of face embeddings
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
        face_embeds = torch.stack(face_embeds).to(device, dtype=dtype)
        
        return (face_embeds,)


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
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        file = f"{filename}_{counter:05}.facevec"
        file = os.path.join(full_output_folder, file)

        # Save face vector to CPU for storage
        torch.save(face_vector.cpu(), file)
        print(f"\033[33mINFO: Face vector saved to {file}\033[0m")
        return (None,)


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

