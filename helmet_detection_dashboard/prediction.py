"""
Prediction Module - Support All Model Types
Single and batch prediction for Traditional ML, CNN, and ViT
"""

import torch
import numpy as np
import cv2
from PIL import Image


def predict_single(image, model, processor, device, model_type='vit'):
    """
    Make prediction on a single image
    
    Args:
        image: PIL Image (RGB)
        model: Loaded model (ViT, CNN, or Traditional ML dict)
        processor: ViT processor or None
        device: torch device or 'cpu'
        model_type: 'vit', 'cnn', or 'traditional'
    
    Returns:
        pred_class (int): 0 or 1
        confidence (float): confidence score
        all_probs (numpy array): [prob_class0, prob_class1]
    """
    
    if model_type == 'traditional':
        # ========== TRADITIONAL ML (PART 1) ==========
        return predict_traditional(image, model)
    
    elif model_type == 'cnn':
        # ========== CNN FROM SCRATCH (PART 2) ==========
        return predict_cnn(image, model)
    
    elif model_type == 'vit':
        # ========== VISION TRANSFORMER (PART 3) ==========
        return predict_vit(image, model, processor, device)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def predict_traditional(image, model_data):
    """
    Predict using Traditional ML (SVM + Manual Features)
    
    Args:
        image: PIL Image
        model_data: Dictionary with model, scaler, pca, feature_extractor
    
    Returns:
        pred_class, confidence, all_probs
    """
    # Convert PIL to numpy if needed
    if hasattr(image, 'convert'):
        image_np = np.array(image.convert('RGB'))
    else:
        image_np = image
    
    # Extract features using the loaded feature extractor
    feature_extractor = model_data['feature_extractor']
    features = feature_extractor(image_np)  # Returns (n_features,) or (1, n_features)
    
    # DEBUG: Print feature count
    print(f"\n🔍 DEBUG - Feature Extraction:")
    print(f"   Extracted features: {len(features) if len(features.shape) == 1 else features.shape[1]}")
    print(f"   Feature shape: {features.shape}")
    
    # Ensure 2D shape (1, n_features)
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    print(f"   After reshape: {features.shape}")
    
    # Apply preprocessing pipeline if available
    if model_data.get('scaler') is not None:
        print(f"   Applying StandardScaler...")
        features = model_data['scaler'].transform(features)
        print(f"   After scaling: {features.shape}")
    
    if model_data.get('pca') is not None:
        print(f"   Applying PCA (from {model_data['pca'].n_features_in_} to {model_data['pca'].n_components_} features)...")
        features = model_data['pca'].transform(features)
        print(f"   After PCA: {features.shape}")
    
    # Get model
    svm_model = model_data['model']
    print(f"   SVM expects: {svm_model.n_features_in_} features")
    print(f"   We have: {features.shape[1]} features")
    
    if features.shape[1] != svm_model.n_features_in_:
        raise ValueError(f"Feature mismatch! SVM expects {svm_model.n_features_in_}, but got {features.shape[1]}")
    
    # Predict
    pred_class = int(svm_model.predict(features)[0])
    
    # Get probabilities
    if hasattr(svm_model, 'predict_proba'):
        # SVM trained with probability=True
        probs = svm_model.predict_proba(features)[0]
    elif hasattr(svm_model, 'decision_function'):
        # Fallback: use decision function
        decision = svm_model.decision_function(features)
        
        if len(decision.shape) > 1 and decision.shape[1] > 1:
            # Multi-class: use softmax
            exp_scores = np.exp(decision[0] - np.max(decision[0]))
            probs = exp_scores / np.sum(exp_scores)
        else:
            # Binary: use sigmoid
            decision_val = decision[0] if hasattr(decision, '__len__') else decision
            prob_class_1 = 1 / (1 + np.exp(-decision_val))
            probs = np.array([1 - prob_class_1, prob_class_1])
    else:
        # Last resort: one-hot encoding
        probs = np.zeros(2)
        probs[pred_class] = 1.0
    
    confidence = float(probs[pred_class])
    
    print(f"   ✅ Prediction: Class {pred_class}, Confidence: {confidence:.2%}\n")
    
    return pred_class, confidence, probs


def predict_cnn(image, model):
    """
    Predict using CNN From Scratch
    
    Args:
        image: PIL Image
        model: Keras/TensorFlow model
    
    Returns:
        pred_class, confidence, all_probs
    """
    # Preprocess image
    img_array = np.array(image.resize((128, 128)))
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict
    probs = model.predict(img_array, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    
    return pred_class, confidence, probs


def predict_vit(image, model, processor, device):
    """
    Predict using Vision Transformer
    
    Args:
        image: PIL Image
        model: ViT model
        processor: ViT processor
        device: torch device
    
    Returns:
        pred_class, confidence, all_probs
    """
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]
    
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    
    return pred_class, confidence, probs


def predict_batch(images, model, processor, device, model_type='vit', progress_callback=None):
    """
    Make predictions on a batch of images
    
    Args:
        images: List of PIL Images
        model: Loaded model
        processor: Processor or None
        device: Device
        model_type: Model type
        progress_callback: Optional callback function(current, total)
    
    Returns:
        List of dicts with results
    """
    results = []
    
    for idx, image in enumerate(images):
        pred_class, confidence, all_probs = predict_single(
            image, model, processor, device, model_type
        )
        
        results.append({
            'pred_class': pred_class,
            'confidence': confidence,
            'all_probs': all_probs
        })
        
        if progress_callback:
            progress_callback(idx + 1, len(images))
    
    return results

# """
# Prediction Module
# Functions for single and batch image prediction
# """

# import torch
# import numpy as np
# from PIL import Image

# def predict_single(image, model, processor, device):
#     """
#     Predict single image
    
#     Args:
#         image: PIL Image
#         model: Loaded ViT model
#         processor: ViT image processor
#         device: torch device
        
#     Returns:
#         pred_class: Predicted class (0 or 1)
#         confidence: Confidence score
#         all_probs: Array of probabilities for all classes
#     """
    
#     # Ensure image is PIL Image
#     if not isinstance(image, Image.Image):
#         image = Image.fromarray(image)
    
#     # Preprocess
#     inputs = processor(images=image, return_tensors="pt")
#     pixel_values = inputs['pixel_values'].to(device)
    
#     # Predict
#     with torch.no_grad():
#         outputs = model(pixel_values=pixel_values)
#         logits = outputs.logits
#         probs = torch.softmax(logits, dim=1)
#         pred_class = torch.argmax(probs, dim=1).item()
#         confidence = probs[0][pred_class].item()
#         all_probs = probs[0].cpu().numpy()
    
#     return pred_class, confidence, all_probs


# def predict_batch(images, model, processor, device):
#     """
#     Predict batch of images
    
#     Args:
#         images: List of PIL Images
#         model: Loaded ViT model
#         processor: ViT image processor
#         device: torch device
        
#     Returns:
#         predictions: List of dicts with pred_class, confidence, all_probs
#     """
    
#     results = []
    
#     for image in images:
#         pred_class, confidence, all_probs = predict_single(
#             image, model, processor, device
#         )
        
#         results.append({
#             'pred_class': pred_class,
#             'confidence': confidence,
#             'all_probs': all_probs
#         })
    
#     return results