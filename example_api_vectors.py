#!/usr/bin/env python3
"""
Example: Reading face vectors directly from ComfyUI API response.

The SaveFaceVector node returns vectors as JSON arrays in the API response,
so you don't need to read files - just parse the JSON!
"""

import urllib.request
import json
import numpy as np


def get_history(server_address, prompt_id):
    """Get execution history for a specific prompt"""
    url = f"http://{server_address}/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def extract_face_vectors_from_api(server_address, prompt_id):
    """
    Extract face vectors directly from ComfyUI API response.
    
    Args:
        server_address: ComfyUI server address (e.g., "127.0.0.1:8188")
        prompt_id: Prompt ID from API response
        
    Returns:
        list: List of face vectors, each is a list of 512 floats
    """
    history = get_history(server_address, prompt_id)[prompt_id]
    
    all_vectors = []
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'ui' in node_output and 'vectors' in node_output['ui']:
            vectors = node_output['ui']['vectors']
            all_vectors.extend(vectors)
    
    return all_vectors


def cosine_similarity_numpy(vec1, vec2):
    """Calculate cosine similarity between two vectors (numpy arrays or lists)."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Normalize vectors
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    
    # Calculate cosine similarity
    similarity = np.dot(vec1_norm, vec2_norm)
    return float(similarity)


# Example usage
if __name__ == "__main__":
    server_address = "127.0.0.1:8188"
    prompt_id = "your-prompt-id-here"  # Replace with actual prompt ID
    
    print("Example: Reading face vectors from API")
    print("=" * 50)
    
    try:
        # Extract vectors from API
        vectors = extract_face_vectors_from_api(server_address, prompt_id)
        
        print(f"Found {len(vectors)} face vector(s)")
        print()
        
        # Display vector information
        for i, vector in enumerate(vectors):
            print(f"Vector {i}:")
            print(f"  Dimensions: {len(vector)}")
            print(f"  First 5 values: {vector[:5]}")
            print(f"  Last 5 values: {vector[-5:]}")
            print(f"  Min value: {min(vector):.4f}")
            print(f"  Max value: {max(vector):.4f}")
            print(f"  Mean value: {sum(vector)/len(vector):.4f}")
            print()
        
        # Compare vectors if we have multiple
        if len(vectors) >= 2:
            print("Comparing vectors:")
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    similarity = cosine_similarity_numpy(vectors[i], vectors[j])
                    print(f"  Vector {i} vs Vector {j}: {similarity:.4f}")
                    if similarity > 0.7:
                        print(f"    → Likely the same person!")
                    elif similarity > 0.5:
                        print(f"    → Possibly the same person")
                    else:
                        print(f"    → Different person")
        
        # Convert to numpy arrays for further processing
        vectors_np = [np.array(v) for v in vectors]
        print()
        print("Converted to numpy arrays:")
        for i, vec_np in enumerate(vectors_np):
            print(f"  Vector {i}: shape {vec_np.shape}, dtype {vec_np.dtype}")
        
    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Make sure:")
        print("1. ComfyUI is running")
        print("2. Replace 'prompt_id' with an actual prompt ID from your workflow")
        print("3. The workflow includes a SaveFaceVector node")
    
    print()
    print("=" * 50)
    print("Complete API Response Structure:")
    print()
    print("""
    {
        "ui": {
            "face_vectors": [
                {
                    "filename": "face_vector_00001.facevec",
                    "subfolder": "",
                    "type": "output"
                }
            ],
            "vectors": [
                [0.123, -0.456, 0.789, ...],  // 512 float values
                [0.234, -0.567, 0.890, ...]   // Another vector if batch > 1
            ]
        }
    }
    """)

