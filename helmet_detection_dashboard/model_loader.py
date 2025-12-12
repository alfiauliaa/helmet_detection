"""
Model Loader Module
Load Vision Transformer model from checkpoint
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor

def load_model(model_path, device):
    """
    Load Vision Transformer model from checkpoint
    
    Args:
        model_path: Path to saved model (.pth file)
        device: torch device (cuda/cpu)
        
    Returns:
        model: Loaded ViT model
        processor: ViT image processor
    """
    
    model_name = 'google/vit-base-patch16-224'
    num_classes = 2
    
    # Load processor
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # Load model architecture
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        attn_implementation="eager"  # For attention rollout
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    print(f"   Device: {device}")
    
    return model, processor