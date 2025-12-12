"""
Prediction Module
Functions for single and batch image prediction
"""

import torch
import numpy as np
from PIL import Image

def predict_single(image, model, processor, device):
    """
    Predict single image
    
    Args:
        image: PIL Image
        model: Loaded ViT model
        processor: ViT image processor
        device: torch device
        
    Returns:
        pred_class: Predicted class (0 or 1)
        confidence: Confidence score
        all_probs: Array of probabilities for all classes
    """
    
    # Ensure image is PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        all_probs = probs[0].cpu().numpy()
    
    return pred_class, confidence, all_probs


def predict_batch(images, model, processor, device):
    """
    Predict batch of images
    
    Args:
        images: List of PIL Images
        model: Loaded ViT model
        processor: ViT image processor
        device: torch device
        
    Returns:
        predictions: List of dicts with pred_class, confidence, all_probs
    """
    
    results = []
    
    for image in images:
        pred_class, confidence, all_probs = predict_single(
            image, model, processor, device
        )
        
        results.append({
            'pred_class': pred_class,
            'confidence': confidence,
            'all_probs': all_probs
        })
    
    return results