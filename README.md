# ComfyUI Vector Face

A ComfyUI custom node pack for extracting, saving, and loading face vectors (embeddings) from images using InsightFace.

## Features

- **Extract Face Vector**: Extract face embeddings from images using InsightFace
- **Save Face Vector**: Save face vectors to disk for later use via API or workflows
- **Load Face Vector**: Load previously saved face vectors back into workflows

## Installation

### Using ComfyUI Manager

1. Open ComfyUI Manager
2. Search for "Vector Face" or "comfyui-vector-face"
3. Click Install

### Manual Installation

1. Navigate to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/comfyui-vector-face.git
   ```

3. Install dependencies:
   ```bash
   pip install insightface onnxruntime
   ```

4. Restart ComfyUI

## Nodes

### Extract Face Vector

Extracts face embedding vectors from images using InsightFace.

**Inputs:**
- `image` (IMAGE, required): Input image containing a face
- `insightface` (INSIGHTFACE, optional): Pre-loaded InsightFace model (if not provided, will load automatically)
- `provider` (CPU/CUDA/ROCM, optional): Execution provider for InsightFace (default: CPU)
- `use_normed` (BOOLEAN, optional): Use normalized embedding (default: True). Set to False for raw embeddings.

**Outputs:**
- `face_vector` (FACE_VECTOR): Face embedding tensor (shape: [batch_size, 512])

**Usage:**
Connect an image to the node. The node will automatically detect faces and extract embeddings. Supports batch processing for multiple images.

### Save Face Vector

Saves face vectors to disk for later retrieval via API or loading.

**Inputs:**
- `face_vector` (FACE_VECTOR, required): Face vector to save
- `filename_prefix` (STRING, required): Prefix for the saved file (default: "face_vector")

**Outputs:**
- None (output node)

**Usage:**
Connect a face vector from Extract Face Vector node. The file will be saved to ComfyUI's output directory with extension `.facevec`.

### Load Face Vector

Loads previously saved face vectors from disk.

**Inputs:**
- `face_vector` (FILE, required): Previously saved `.facevec` file

**Outputs:**
- `face_vector` (FACE_VECTOR): Loaded face embedding tensor

**Usage:**
Select a `.facevec` file from the input directory. The face vector will be loaded and can be used in your workflow.

## Requirements

- ComfyUI (latest version)
- Python 3.8+
- InsightFace (`pip install insightface`)
- ONNX Runtime (`pip install onnxruntime`)

## InsightFace Models

InsightFace models will be automatically downloaded to `ComfyUI/models/insightface/` on first use. The default model used is `buffalo_l`.

## API Usage

When using the Save Face Vector node, the saved file path will be available in ComfyUI's API response. You can retrieve face vectors via:

1. **File Path**: The saved `.facevec` file path is returned in the API response
2. **Load Node**: Use the Load Face Vector node in subsequent workflows
3. **Direct Access**: Load the `.facevec` file using `torch.load()` in your Python code

## Example Workflow

```
[Load Image] → [Extract Face Vector] → [Save Face Vector]
                                              ↓
                                    [File saved to output/]
                                              ↓
                                    [Load Face Vector] → [Use in generation]
```

## Notes

- Face vectors are 512-dimensional embeddings extracted using InsightFace's `buffalo_l` model
- Normalized embeddings (default) are recommended for most use cases
- Raw embeddings are available for portrait unnorm models
- The node automatically adjusts detection resolution if faces are not detected at default size

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.

