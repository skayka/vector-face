# Reading Face Vectors from API

The `SaveFaceVector` node returns face vectors directly in the API response as JSON, plus saves them to `.facevec` files. Here's how to access them:

## Direct API Access (Recommended)

The API response includes the vectors directly as JSON arrays:

```python
import urllib.request
import json

server_address = "127.0.0.1:8188"

def get_history(prompt_id):
    """Get execution history for a specific prompt"""
    url = f"http://{server_address}/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

# Get history for your prompt
prompt_id = "your-prompt-id-here"
history = get_history(prompt_id)[prompt_id]

# Extract face vectors from API response
for node_id in history['outputs']:
    node_output = history['outputs'][node_id]
    if 'ui' in node_output and 'vectors' in node_output['ui']:
        vectors = node_output['ui']['vectors']
        # vectors is a list of lists, each inner list is a 512-dimensional face vector
        print(f"Found {len(vectors)} face vector(s)")
        for i, vector in enumerate(vectors):
            print(f"  Vector {i}: {len(vector)} dimensions")
            print(f"  First 5 values: {vector[:5]}")
```

## Reading .facevec Files

Alternatively, you can read the `.facevec` files saved to disk. These are PyTorch tensors containing face embeddings:

## Basic Usage

### Python (PyTorch)

```python
import torch

# Read a single .facevec file
face_vector = torch.load("output/face_vector_00001.facevec", map_location='cpu')

# face_vector is a torch.Tensor with shape [1, 512] or [batch_size, 512]
print(f"Shape: {face_vector.shape}")  # [1, 512] for single vector
print(f"Dtype: {face_vector.dtype}")  # torch.float32 or torch.float16
```

### Convert to NumPy

```python
import torch
import numpy as np

face_vector = torch.load("output/face_vector_00001.facevec", map_location='cpu')

# Convert to numpy array
face_vector_np = face_vector.squeeze(0).numpy()  # Remove batch dimension
# or
face_vector_np = face_vector.numpy()[0]  # Get first item from batch

print(f"Shape: {face_vector_np.shape}")  # [512]
```

## Reading from ComfyUI API Response

After executing a workflow with `SaveFaceVector`, you can retrieve the file paths from the API:

```python
import urllib.request
import json
import torch

server_address = "127.0.0.1:8188"

def get_history(prompt_id):
    """Get execution history for a specific prompt"""
    url = f"http://{server_address}/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

# Get history for your prompt
prompt_id = "your-prompt-id-here"
history = get_history(prompt_id)[prompt_id]

# Extract face vector file paths
for node_id in history['outputs']:
    node_output = history['outputs'][node_id]
    if 'ui' in node_output and 'face_vectors' in node_output['ui']:
        for face_vec_info in node_output['ui']['face_vectors']:
            filename = face_vec_info['filename']
            subfolder = face_vec_info.get('subfolder', '')
            
            # Construct full path
            if subfolder:
                file_path = f"ComfyUI/output/{subfolder}/{filename}"
            else:
                file_path = f"ComfyUI/output/{filename}"
            
            # Load the face vector
            face_vector = torch.load(file_path, map_location='cpu')
            print(f"Loaded: {filename}, Shape: {face_vector.shape}")
```

## Batch Processing

If you saved multiple face vectors in a batch, each will be in a separate file:

```python
import torch
from pathlib import Path

# Read all .facevec files from a directory
output_dir = Path("ComfyUI/output")
face_vectors = []

for facevec_file in output_dir.glob("*.facevec"):
    face_vector = torch.load(facevec_file, map_location='cpu')
    face_vectors.append(face_vector)
    print(f"Loaded {facevec_file.name}: {face_vector.shape}")

# Stack into a batch tensor
if face_vectors:
    batch_tensor = torch.cat(face_vectors, dim=0)
    print(f"Batch tensor shape: {batch_tensor.shape}")  # [N, 512]
```

## Comparing Face Vectors

Calculate similarity between face vectors:

```python
import torch

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two face vectors."""
    # Normalize vectors
    vec1_norm = vec1 / torch.norm(vec1)
    vec2_norm = vec2 / torch.norm(vec2)
    # Calculate cosine similarity
    similarity = torch.dot(vec1_norm.flatten(), vec2_norm.flatten())
    return similarity.item()

# Load two face vectors
vec1 = torch.load("output/face_vector_00001.facevec", map_location='cpu')
vec2 = torch.load("output/face_vector_00002.facevec", map_location='cpu')

# Remove batch dimension if present
if vec1.shape[0] > 1:
    vec1 = vec1[0]
if vec2.shape[0] > 1:
    vec2 = vec2[0]

similarity = cosine_similarity(vec1, vec2)
print(f"Cosine similarity: {similarity:.4f}")
# 1.0 = identical faces
# 0.7-0.9 = same person
# < 0.5 = different person
```

## File Format

- **Format**: PyTorch tensor (saved with `torch.save()`)
- **Shape**: `[1, 512]` for single vector, `[batch_size, 512]` for batch
- **Dtype**: `torch.float32` or `torch.float16`
- **Content**: Face embedding vector extracted using InsightFace (buffalo_l model)
- **Normalization**: Uses normalized embeddings by default (normed_embedding)

## Example: Complete Workflow

```python
import torch
import urllib.request
import json
from pathlib import Path

def read_facevec_from_api(server_address, prompt_id, output_dir="ComfyUI/output"):
    """Read face vectors from ComfyUI API response."""
    # Get history
    url = f"http://{server_address}/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        history = json.loads(response.read())[prompt_id]
    
    face_vectors = []
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'ui' in node_output and 'face_vectors' in node_output['ui']:
            for face_vec_info in node_output['ui']['face_vectors']:
                filename = face_vec_info['filename']
                subfolder = face_vec_info.get('subfolder', '')
                
                file_path = Path(output_dir) / subfolder / filename if subfolder else Path(output_dir) / filename
                face_vector = torch.load(str(file_path), map_location='cpu')
                face_vectors.append({
                    'filename': filename,
                    'vector': face_vector,
                    'path': str(file_path)
                })
    
    return face_vectors

# Usage
face_vectors = read_facevec_from_api("127.0.0.1:8188", "your-prompt-id")
for fv in face_vectors:
    print(f"{fv['filename']}: {fv['vector'].shape}")
```

