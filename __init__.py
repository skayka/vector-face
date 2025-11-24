"""
ComfyUI Vector Face - Extract, Save, and Load Face Vectors

A ComfyUI custom node pack for extracting face embeddings (vectors) from images
using InsightFace, with support for saving and loading face vectors.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

