"""
Model Loader Module - Support All Parts (1, 2, 3)
Load Traditional ML, CNN, and Vision Transformer models
"""
import torch
import pickle
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor
from tensorflow import keras
import cv2
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

# =====================================================
# FEATURE EXTRACTION FOR TRADITIONAL ML (PART 1)
# =====================================================

def extract_set1_features(image):
    """
    Extract Histogram + HOG + LBP features (SET 1 from Part 1)
    
    Args:
        image: PIL Image or numpy array (RGB/BGR)
    
    Returns:
        numpy array of features (flattened)
    """
    # Convert PIL to numpy BGR
    if hasattr(image, 'convert'):
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize to 128x128
    image = cv2.resize(image, (128, 128))
    
    # ==================== HISTOGRAM (COLOR DISTRIBUTION) ====================
    hist_features = []
    for i in range(3):  # BGR channels
        hist = cv2.calcHist([image], [i], None, [32], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-7)
        hist_features.extend(hist)
    
    # ==================== HOG (SHAPE/EDGE) ====================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray, 
        orientations=9, 
        pixels_per_cell=(8, 8),  # Note: 8x8 for SET 1
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        visualize=False, 
        feature_vector=True
    )
    
    # ==================== LBP (TEXTURE) ====================
    radius = 3
    n_points = 8 * radius  # 24
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    
    # Combine all features
    return np.concatenate([hist_features, hog_feat, lbp_hist])


def extract_set2_features(image):
    """
    Extract Color Moments + HOG + GLCM features (SET 2 from Part 1)
    This MUST match the exact parameters used during training!
    
    Expected output: 1793 features total
      - Color Moments: 9 (3 channels x [mean, std, skew])
      - HOG: 1764 (128x128 image, 16x16 cells, 2x2 blocks, 9 orientations)
      - GLCM: 20 (5 properties x 4 angles)
    
    Args:
        image: PIL Image or numpy array (RGB/BGR)
    
    Returns:
        numpy array of features (1793,)
    """
    # Convert PIL to numpy BGR (OpenCV format)
    if hasattr(image, 'convert'):
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize to 128x128 (CRITICAL: same as training)
    image = cv2.resize(image, (128, 128))
    
    features = []
    
    # ==================== COLOR MOMENTS ====================
    # Mean, Std, Skewness for each BGR channel
    for i in range(3):
        channel = image[:,:,i].astype(np.float64)
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        centered = channel - mean_val
        skew_val = np.mean(centered**3) / (std_val**3 + 1e-7)
        features.extend([mean_val, std_val, skew_val])
    
    # ==================== HOG ====================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    hog_feat = hog(
        gray, 
        orientations=9, 
        pixels_per_cell=(16, 16),  # CRITICAL: 16x16, NOT 8x8!
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        visualize=False, 
        feature_vector=True
    )
    
    # ==================== GLCM TEXTURE ====================
    # Normalize gray to 16 levels (reduces from 256 to 16)
    gray_norm = (gray / 16).astype(np.uint8)
    
    glcm = graycomatrix(
        gray_norm, 
        distances=[1], 
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=16,  # CRITICAL: 16 levels, NOT 256!
        symmetric=True, 
        normed=True
    )
    
    glcm_features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        glcm_features.extend(graycoprops(glcm, prop).flatten())
    
    # Combine all features: 9 + 1764 + 20 = 1793
    all_features = np.concatenate([features, hog_feat, glcm_features])
    
    # Verify feature count
    expected_count = 1793
    actual_count = len(all_features)
    
    if actual_count != expected_count:
        print(f"⚠️  WARNING: Feature count mismatch!")
        print(f"   Expected: {expected_count}")
        print(f"   Got: {actual_count}")
        print(f"   Breakdown:")
        print(f"     - Color Moments: {len(features)}")
        print(f"     - HOG: {len(hog_feat)}")
        print(f"     - GLCM: {len(glcm_features)}")
    
    return all_features


def extract_set3_features(image):
    """
    Extract Edge + HOG + Color Histogram features (SET 3 from Part 1)
    
    Args:
        image: PIL Image or numpy array (RGB/BGR)
    
    Returns:
        numpy array of features (flattened)
    """
    # Convert PIL to numpy BGR
    if hasattr(image, 'convert'):
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize to 128x128
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    features = []
    
    # ==================== EDGE FEATURES ====================
    # Canny edge detection
    edges_canny = cv2.Canny(gray, 50, 150)
    features.extend([
        np.sum(edges_canny > 0) / edges_canny.size,  # Edge density
        np.mean(edges_canny),                         # Mean edge intensity
        np.std(edges_canny)                           # Std edge intensity
    ])
    
    # Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    features.extend([
        np.mean(np.abs(sobelx)),
        np.mean(np.abs(sobely))
    ])
    
    # ==================== HOG (COMPACT) ====================
    hog_feat = hog(
        gray, 
        orientations=8,  # 8 for SET 3
        pixels_per_cell=(16, 16), 
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        visualize=False, 
        feature_vector=True
    )
    
    # ==================== COLOR HISTOGRAM (REDUCED) ====================
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [16], [0, 256])  # 16 bins
        hist = hist.flatten() / (hist.sum() + 1e-7)
        hist_features.extend(hist)
    
    return np.concatenate([features, hog_feat, hist_features])


# =====================================================
# AUTO-DETECT FEATURE SET
# =====================================================

def detect_and_extract_features(image, expected_features):
    """
    Auto-detect which feature set to use based on expected feature count
    
    Args:
        image: Input image
        expected_features: Expected feature count from model (before PCA)
    
    Returns:
        features: Extracted features matching expected count
        feature_set_name: Name of detected feature set
    """
    print(f"\n🔍 Auto-detecting feature set (expecting {expected_features} features before PCA)...")
    
    feature_sets = {
        'SET 1: Histogram+HOG+LBP': extract_set1_features,
        'SET 2: ColorMoments+HOG+GLCM': extract_set2_features,
        'SET 3: Edge+HOG+ColorHist': extract_set3_features
    }
    
    results = []
    
    for set_name, extractor in feature_sets.items():
        try:
            features = extractor(image)
            feature_count = len(features)
            results.append((set_name, extractor, features, feature_count))
            
            if feature_count == expected_features:
                print(f"✅ EXACT MATCH! Using {set_name} ({feature_count} features)")
                return features, set_name
            else:
                print(f"   {set_name}: {feature_count} features")
        except Exception as e:
            print(f"   {set_name}: Failed ({str(e)})")
    
    # If no exact match, find the closest
    if results:
        results.sort(key=lambda x: abs(x[3] - expected_features))
        best_match = results[0]
        
        print(f"\n⚠️  No exact match found!")
        print(f"   Expected: {expected_features}")
        print(f"   Closest match: {best_match[0]} with {best_match[3]} features")
        print(f"   Difference: {abs(best_match[3] - expected_features)} features")
        
        # Use the closest match
        return best_match[2], best_match[0]
    
    # Last resort: use SET 2
    print(f"⚠️  All feature extractors failed, using SET 2 as default")
    features = extract_set2_features(image)
    return features, 'SET 2: ColorMoments+HOG+GLCM'


# =====================================================
# MODEL LOADERS
# =====================================================

def load_model(model_path, device, model_type='vit'):
    """
    Load model based on type
    
    Args:
        model_path: Path to model file
        device: torch device (for ViT) or 'cpu'
        model_type: 'vit' (Part 3), 'cnn' (Part 2), or 'traditional' (Part 1)
    
    Returns:
        model, processor/None
    """
    
    print(f"\n📥 Loading model: {model_type}")
    print(f"   Path: {model_path}")
    
    if model_type == 'vit':
        # Part 3: Vision Transformer
        return load_vit_model(model_path, device)
    
    elif model_type == 'cnn':
        # Part 2: CNN From Scratch
        return load_cnn_model(model_path), None
    
    elif model_type == 'traditional':
        # Part 1: Traditional ML (SVM-RBF)
        return load_traditional_model(model_path), None
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_vit_model(model_path, device):
    """Load Vision Transformer model (Part 3)"""
    
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
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✅ Vision Transformer loaded successfully!")
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, processor


def load_cnn_model(model_path):
    """Load CNN model (Part 2)"""
    
    try:
        model = keras.models.load_model(model_path)
        model.trainable = False  # Set to inference mode
        
        print(f"✅ CNN model loaded successfully!")
        print(f"   Parameters: {model.count_params():,}")
        print(f"   Input shape: {model.input_shape}")
        
        return model
        
    except Exception as e:
        raise Exception(f"Failed to load CNN model: {str(e)}")


def load_traditional_model(model_path):
    """
    Load Traditional ML model (Part 1)
    Handles multiple pickle formats robustly with auto feature detection
    """
    
    try:
        # CRITICAL: Import feature extractors into __main__ namespace
        import __main__
        __main__.extract_set1_features = extract_set1_features
        __main__.extract_set2_features = extract_set2_features
        __main__.extract_set3_features = extract_set3_features
        
        print(f"📥 Loading Traditional ML model from: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check if it's a dictionary (expected format from Part 1)
        if isinstance(model_data, dict):
            
            # Validate required keys
            required_keys = ['model', 'scaler', 'pca']
            missing_keys = [key for key in required_keys if key not in model_data]
            
            if missing_keys:
                raise ValueError(f"Model file is missing required keys: {missing_keys}")
            
            # Get expected feature count from model (after PCA)
            svm_model = model_data['model']
            expected_features_pca = svm_model.n_features_in_
            
            # Get PCA to find original feature count
            pca = model_data['pca']
            expected_features_original = pca.n_features_in_
            
            print(f"📊 Model expects:")
            print(f"   Original features (before PCA): {expected_features_original}")
            print(f"   PCA components: {pca.n_components_}")
            print(f"   Features after PCA: {expected_features_pca}")
            
            # Auto-detect correct feature extractor
            # Create dummy image to test
            dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
            
            detected_features, detected_set = detect_and_extract_features(
                dummy_img, 
                expected_features_original
            )
            
            # Override feature_extractor based on detection
            if 'SET 1' in detected_set:
                model_data['feature_extractor'] = extract_set1_features
            elif 'SET 3' in detected_set:
                model_data['feature_extractor'] = extract_set3_features
            else:
                model_data['feature_extractor'] = extract_set2_features
            
            print(f"✅ Traditional ML model loaded successfully!")
            print(f"   Algorithm: {model_data.get('algorithm', 'SVM-RBF')}")
            print(f"   Feature Set: {detected_set}")
            
            if 'test_acc' in model_data:
                print(f"   Test Accuracy: {model_data['test_acc']:.2%}")
            
            return model_data
        
        else:
            # Fallback: wrap bare model in dictionary
            print(f"⚠️  Model is not in expected dictionary format, wrapping...")
            
            wrapped_data = {
                'model': model_data,
                'scaler': None,
                'pca': None,
                'feature_extractor': extract_set2_features,
                'algorithm': 'SVM-RBF',
                'feature_set': 'SET 2: ColorMoments+HOG+GLCM'
            }
            
            print(f"✅ Model wrapped successfully!")
            return wrapped_data
        
    except Exception as e:
        print(f"\n❌ ERROR DETAILS:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        
        # Check if it's the feature extraction error
        if "features" in str(e).lower():
            print(f"\n💡 SOLUTION:")
            print(f"   Feature count mismatch detected.")
            print(f"   The model expects a specific feature set (SET 1, 2, or 3).")
            print(f"   Auto-detection will now attempt to find the correct set.")
        
        raise Exception(f"Failed to load Traditional ML model: {str(e)}")
    
# """
# Model Loader Module
# Load Vision Transformer model from checkpoint
# """

# import torch
# from transformers import ViTForImageClassification, ViTImageProcessor

# def load_model(model_path, device):
#     """
#     Load Vision Transformer model from checkpoint
    
#     Args:
#         model_path: Path to saved model (.pth file)
#         device: torch device (cuda/cpu)
        
#     Returns:
#         model: Loaded ViT model
#         processor: ViT image processor
#     """
    
#     model_name = 'google/vit-base-patch16-224'
#     num_classes = 2
    
#     # Load processor
#     processor = ViTImageProcessor.from_pretrained(model_name)
    
#     # Load model architecture
#     model = ViTForImageClassification.from_pretrained(
#         model_name,
#         num_labels=num_classes,
#         ignore_mismatched_sizes=True,
#         attn_implementation="eager"  # For attention rollout
#     )
    
#     # Load trained weights
#     checkpoint = torch.load(model_path, map_location=device)
    
#     # Handle different checkpoint formats
#     if 'model_state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dict'])
#     else:
#         model.load_state_dict(checkpoint)
    
#     # Move to device and set to eval mode
#     model = model.to(device)
#     model.eval()
    
#     print(f"✅ Model loaded from {model_path}")
#     print(f"   Device: {device}")
    
#     return model, processor
