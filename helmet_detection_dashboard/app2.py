"""
Helmet Detection Dashboard - Improved Version
Vision Transformer with Explainable AI (Single & Batch)
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import time
import pandas as pd
from pathlib import Path
import os
import zipfile
import io

# Force reload modules to avoid cache issues
import sys
if 'model_loader' in sys.modules:
    del sys.modules['model_loader']
if 'prediction' in sys.modules:
    del sys.modules['prediction']

# Now import
from model_loader import load_model
from prediction import predict_single, predict_batch

# Import local modules
from model_loader import load_model
from prediction import predict_single, predict_batch
from xai_attention import generate_attention_rollout
from visualization import (
    plot_single_prediction, 
    plot_batch_statistics,
    create_attention_grid,
    plot_single_attention
)

# Page config
st.set_page_config(
    page_title="Helmet Detection - ViT Dashboard",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    /* Main Header - Bigger and Centered */
    .main-header {
        font-size: 4.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        letter-spacing: 2px;
    }
    
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .success-box {
        background-color: #C1E59F;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    
    .xai-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border: 2px solid #e9ecef;
    }
    
    /* About Section Styling */
    .about-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #667eea;
    }
    
    .best-model {
        background: linear-gradient(135deg, #10b98115 0%, #059669 100%);
        border-left: 4px solid #10b981;
        font-weight: 600;
    }
    
    .model-comparison {
        font-size: 0.85rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'device' not in st.session_state:
    st.session_state.device = None

# Sidebar
with st.sidebar:
    st.markdown("## 🔧 Configuration")
    
    # ========== TAMBAHAN BARU: Model Selection ==========
    st.markdown("### 🎯 Select Model")
    
    model_choice = st.selectbox(
        "Choose Model Part",
        options=[
            "Part 3: Vision Transformer + LoRA ⭐ (Best - 94.78%)",
            "Part 2: CNN (Good - 90.30%)",
            "Part 1: Traditional ML SVM-RBF (Baseline - 82.84%)"
        ],
        index=0,
        help="Select which model to use for prediction"
    )
    
    # Determine model type and default path
    if "Part 3" in model_choice:
        model_type = 'vit'
        default_path = "models/part3_vit_model.pth"
        model_name_display = "Vision Transformer (ViT-Base/16)"
        xai_supported = True
        model_description = "Uses self-attention mechanism, pretrained on ImageNet-21k"
    elif "Part 2" in model_choice:
        model_type = 'cnn'
        default_path = "models/part2_cnn_model.h5"
        model_name_display = "CNN (3 Conv Blocks)"
        xai_supported = False
        model_description = "Custom CNN architecture trained from random initialization"
    elif "Part 1" in model_choice:
        model_type = 'traditional'
        default_path = "models/part1_svm_model.pkl"
        model_name_display = "SVM-RBF + Manual Features"
        xai_supported = False
        model_description = "HOG + LBP + GLCM features with Support Vector Machine"
    
    st.info(f"📌 **Selected:** {model_name_display}")
    st.caption(model_description)
    
    if not xai_supported:
        st.warning("⚠️ XAI (Attention Rollout) not supported for this model")
    
    st.markdown("---")
    # ========== AKHIR TAMBAHAN BARU ==========
    
    # Model path (UPDATE: ganti value)
    model_path = st.text_input(
        "Model Path",
        value=default_path,  # GANTI dari "models/best_vit_model.pth"
        help="Path to the saved model file"
    )
    
    # Device selection (tetap sama, tapi update logic)
    device_option = st.radio(
        "Device",
        options=["Auto", "CPU", "CUDA"],
        index=0,
        help="Select computation device"
    )
    
    # Load model button (UPDATE logic)
    if st.button("🚀 Load Model", type="primary", use_container_width=True):
        with st.spinner(f"Loading {model_name_display}..."):
            try:
                # Determine device
                if model_type == 'vit':
                    # Only ViT uses GPU
                    if device_option == "Auto":
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    elif device_option == "CUDA":
                        if torch.cuda.is_available():
                            device = torch.device('cuda')
                        else:
                            st.error("CUDA not available! Using CPU instead.")
                            device = torch.device('cpu')
                    else:
                        device = torch.device('cpu')
                else:
                    # CNN & Traditional ML always use CPU
                    device = 'cpu'
                    if device_option == "CUDA":
                        st.info("ℹ️ CNN and Traditional ML models run on CPU only")
                
                # Load model
                model, processor = load_model(model_path, device, model_type)
                
                # Store in session state
                st.session_state.model = model
                st.session_state.processor = processor
                st.session_state.device = device
                st.session_state.model_type = model_type
                st.session_state.xai_supported = xai_supported
                st.session_state.model_name = model_name_display
                
                st.success(f"✅ {model_name_display} loaded successfully!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.exception(e)  # Show full traceback
    
    # Model status
    st.markdown("---")
    st.markdown("### 📊 Model Status")
    
    if st.session_state.model is not None:
        st.markdown('<div class="success-box">✅ Model Loaded</div>', 
                    unsafe_allow_html=True)
        st.info(f"Device: **{st.session_state.device}**")
    else:
        st.warning("⚠️ Model Not Loaded")
    
    # Enhanced About Section
    st.markdown("---")
    st.markdown("### ℹ️ About This System")
    
    st.markdown("""
    <div class="about-card">
        <strong>🎯 Current Model</strong><br>
        Architecture: <strong>Vision Transformer (ViT-Base/16)</strong><br>
        Pretrained: <strong>ImageNet-21k</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Model Comparison")
    
    # Part 1: Traditional ML
    st.markdown("""
    <div class="about-card">
        <div class="model-comparison">
            <strong>Part 1: Traditional ML</strong><br>
            Algorithm: <strong>SVM-RBF</strong><br>
            Features: HOG + LBP + GLCM<br>
            Accuracy: <span style="color: #f59e0b;">82.84%</span><br>
            Training: ~0.24s (Fast ⚡)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Part 2: CNN From Scratch
    st.markdown("""
    <div class="about-card">
        <div class="model-comparison">
            <strong>Part 2: CNN</strong><br>
            Architecture: <strong>Custom CNN (3 Conv Blocks)</strong><br>
            Parameters: 1.27M (all trainable)<br>
            Accuracy: <span style="color: #3b82f6;">90.30%</span><br>
            Training: ~272s (Medium ⏱️)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Part 3: Vision Transformer (BEST)
    st.markdown("""
    <div class="about-card best-model">
        <div class="model-comparison">
            <strong>🏆 Part 3: Vision Transformer + LoRA</strong><br>
            <em>(Current Model - BEST)</em><br><br>
            Architecture: <strong>ViT-Base/16</strong><br>
            Strategy: Transfer Learning + LoRA<br>
            Parameters: 85.8M (only 1,538 trainable)<br>
            <strong>Accuracy: <span style="color: #10b981;">94.78%</span> ✨</strong><br>
            Precision: <span style="color: #10b981;">94.93%</span><br>
            Recall: <span style="color: #10b981;">94.78%</span><br>
            F1-Score: <span style="color: #10b981;">94.80%</span><br>
            Training: ~249s (Efficient 🚀)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.1); padding: 0.8rem; border-radius: 8px; margin-top: 0.5rem;">
        <strong style="color: #10b981;">✅ Why Part 3 is the Best:</strong><br>
        <ul style="font-size: 0.85rem; margin-top: 0.5rem;">
            <li>Highest accuracy (+11.94% vs Part 1)</li>
            <li>Best precision & recall balance</li>
            <li>Pretrained on 14M images</li>
            <li>Uses attention mechanism (XAI)</li>
            <li>Efficient training with LoRA</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.7); font-size: 0.8rem;">
        <strong>Features</strong><br>
        ✓ Single image prediction<br>
        ✓ Batch processing (ZIP support)<br>
        ✓ Attention visualization (XAI)
    </div>
    """, unsafe_allow_html=True)


# # Main content - HEADER Opsi 1: Dark Elegant
# st.markdown("""
# <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 2.5rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.3);">
#     <h1 style="font-size: 4.5rem; font-weight: 900; text-align: center; color: #eee; margin: 0; letter-spacing: 3px; text-shadow: 2px 2px 8px rgba(0,0,0,0.5);">
#         🏍️🛵🛡️🪖 HELMET DETECTION
#     </h1>
#     <div style="width: 200px; height: 4px; background: linear-gradient(90deg, #ffd700, #ffa500, #ffd700); margin: 1rem auto; border-radius: 2px;"></div>
#     <h2 style="font-size: 1.7rem; text-align: center; color: #ffd700; margin-top: 0.5rem; font-weight: 600; letter-spacing: 2px;">
#         Vision Transformer + LoRA with Explainable AI
#     </h2>
#     <p style="font-size: 1.3rem; text-align: center; color: rgba(255,255,255,0.7); margin-top: 0.5rem; font-weight: 400;">
#         🎯 Accuracy: <strong style="color: #4ade80;">94.78%</strong> • 
#         🧠 Architecture: <strong>ViT-Base/16</strong> • 
#         ⚡ Parameters: <strong>85.8M</strong> (1,538 trainable)
#     </p>
# </div>
# """, unsafe_allow_html=True)

# # Main content - HEADER Opsi 2: Cyberpunk Neon
# st.markdown("""
# <div style="background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 2.5rem; border-radius: 20px; margin-bottom: 2rem; border: 2px solid #00ffff; box-shadow: 0 0 30px rgba(0,255,255,0.3), 0 0 60px rgba(255,0,255,0.2);">
#     <h1 style="font-size: 4.5rem; font-weight: 900; text-align: center; background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; letter-spacing: 4px; text-shadow: 0 0 20px rgba(0,255,255,0.5);">
#         🪖 HELMET DETECTION
#     </h1>
#     <div style="width: 300px; height: 2px; background: linear-gradient(90deg, transparent, #00ffff, #ff00ff, #00ffff, transparent); margin: 1rem auto;"></div>
#     <h2 style="font-size: 1.5rem; text-align: center; color: #00ffff; margin-top: 0.5rem; font-weight: 600; letter-spacing: 2px; text-shadow: 0 0 10px rgba(0,255,255,0.8);">
#         Vision Transformer + LoRA with Explainable AI
#     </h2>
#     <p style="font-size: 1rem; text-align: center; color: rgba(255,255,255,0.8); margin-top: 0.5rem;">
#         <span style="color: #00ffff;">●</span> Accuracy: <strong style="color: #4ade80;">94.78%</strong> 
#         <span style="color: #ff00ff;">●</span> Architecture: <strong style="color: #00ffff;">ViT-Base/16</strong> 
#         <span style="color: #00ffff;">●</span> Trainable: <strong style="color: #ff00ff;">1,538</strong> params
#     </p>
# </div>
# """, unsafe_allow_html=True)

# Main content - HEADER Opsi 5: Professional Corporate
st.markdown("""
<div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 2.5rem; border-radius: 15px; margin-bottom: 2rem; border-top: 5px solid #3498db; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
    <h1 style="font-size: 4rem; font-weight: 700; text-align: center; color: white; margin: 0; letter-spacing: 2px;">
        🏍️🛵🛡️🪖 HELMET DETECTION 
    </h1>
    <div style="width: 150px; height: 3px; background: #3498db; margin: 1rem auto; border-radius: 2px;"></div>
    <h2 style="font-size: 1.5rem; text-align: center; color: #3498db; margin-top: 0.5rem; font-weight: 600;">
        Vision Transformer + LoRA with Explainable AI
    </h2>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; flex-wrap: wrap;">
        <div style="text-align: center;">
            <div style="color: #3498db; font-size: 2rem; font-weight: 700;">94.78%</div>
            <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Accuracy</div>
        </div>
        <div style="text-align: center;">
            <div style="color: #2ecc71; font-size: 2rem; font-weight: 700;">ViT-Base/16</div>
            <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Architecture</div>
        </div>
        <div style="text-align: center;">
            <div style="color: #e74c3c; font-size: 2rem; font-weight: 700;">1.5K</div>
            <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Trainable Params</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Check if model is loaded
if st.session_state.model is None:
    st.info("👈 Please load the model from the sidebar to get started!")
    st.markdown("""
    <div style="background: #f0f7ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3498db; margin-top: 1rem; color: #034a86;">
        <h4>📋 Quick Start Guide:</h4>
        <ol>
            <li>Select a model (Part 1, 2, or 3) from the sidebar</li>
            <li>Click <strong>"🚀 Load Model"</strong></li>
            <li>Upload image(s) and start detecting!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    st.stop()
    
# Tabs
tab1, tab2 = st.tabs(["📸 Single Image", "📁 Batch Processing"])

# ==================== TAB 1: SINGLE IMAGE ====================
with tab1:
    st.markdown("## Single Image Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            key="single_upload"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        # XAI Option (only for models that support it)
       # ✅ KODE BENAR
        if st.session_state.get('xai_supported', False):
            show_xai = st.checkbox("🔍 Show Attention Map (XAI)", value=True, 
                                help="Visualize where the model focuses")
        else:
            show_xai = False
            st.info("ℹ️ XAI (Attention Rollout) is only available for Part 3 (Vision Transformer)")

        # ✅ Button di LUAR blok if-else, jadi selalu muncul
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                start_time = time.time()
                
                # Predict
                pred_class, confidence, all_probs = predict_single(
                    image,
                    st.session_state.model,
                    st.session_state.processor,
                    st.session_state.device,
                    model_type=st.session_state.model_type
                )
                
                # Generate attention if requested
                attention_map = None
                if show_xai:
                    attention_map, _, _ = generate_attention_rollout(
                        image,
                        st.session_state.model,
                        st.session_state.processor,
                        st.session_state.device
                    )
                
                inference_time = time.time() - start_time
                
                st.session_state.single_result = {
                    'pred_class': pred_class,
                    'confidence': confidence,
                    'all_probs': all_probs,
                    'inference_time': inference_time,
                    'attention_map': attention_map,
                    'image': np.array(image)
                }
    
    with col2:
        st.markdown("### 📊 Prediction Result")
        
        if 'single_result' in st.session_state:
            result = st.session_state.single_result
            class_names = ['No Helmet', 'With Helmet']
            pred_name = class_names[result['pred_class']]
            
            # Display prediction
            if result['pred_class'] == 1:
                st.markdown(f"""
                <div class="success-box">
                    <h2 style="color: #28a745; margin: 0;">✅ {pred_name}</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; color: #155724;">
                        Confidence: <strong>{result['confidence']:.2%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                    <h2 style="color: #dc3545; margin: 0;">⚠️ {pred_name}</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">
                        Confidence: <strong>{result['confidence']:.2%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Prediction", pred_name)
            with col_m2:
                st.metric("Inference Time", f"{result['inference_time']*1000:.0f} ms")
            
            # Probability distribution
            st.markdown("### 📈 Probability Distribution")
            fig = plot_single_prediction(result['all_probs'], class_names)
            st.pyplot(fig)
        else:
            st.info("Upload an image and click 'Predict' to see results")
    
    # XAI Visualization (full width below)
    if 'single_result' in st.session_state and st.session_state.single_result['attention_map'] is not None:
        st.markdown("---")
        st.markdown('<div class="xai-section">', unsafe_allow_html=True)
        st.markdown("## 🔍 Explainable AI - Attention Rollout")
        st.markdown("**Red/Yellow areas** = High attention (important for prediction)")
        st.markdown("**Blue/Green areas** = Low attention")
        
        fig_attention = plot_single_attention(
            st.session_state.single_result['image'],
            st.session_state.single_result['attention_map'],
            st.session_state.single_result['pred_class'],
            st.session_state.single_result['confidence']
        )
        st.pyplot(fig_attention)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 2: BATCH PROCESSING ====================
with tab2:
    st.markdown("## Batch Image Processing")
    
    # Upload method selection
    upload_method = st.radio(
        "Upload Method",
        options=["📁 Multiple Files", "🗂️ ZIP Folder"],
        horizontal=True,
        help="Choose how to upload images"
    )
    
    uploaded_files = []
    
    if upload_method == "📁 Multiple Files":
        uploaded_files = st.file_uploader(
            "Choose images...",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="batch_upload"
        )
    else:
        zip_file = st.file_uploader(
            "Upload ZIP folder containing images...",
            type=['zip'],
            key="zip_upload"
        )
        
        if zip_file is not None:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                uploaded_files = []

                for name in zip_ref.namelist():
                    if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_bytes = zip_ref.read(name)

                        class FileObj:
                            def __init__(self, name, data):
                                self.name = name
                                self.data = data

                        uploaded_files.append(FileObj(name, img_bytes))

            st.success(f"✅ Found {len(uploaded_files)} images in ZIP file")
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} images ready to process")
        
        # XAI Options
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            show_batch_xai = st.checkbox("🔍 Generate Attention Maps (XAI)", 
                                         value=True,
                                         help="Generate attention visualization for selected samples")
        with col_opt2:
            if show_batch_xai:
                num_xai_samples = st.slider("Number of XAI samples", 4, 20, 8, 4)
        
        if st.button("🚀 Process Batch", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            images_for_xai = []
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {file.name}")
                
                # Load image
                if hasattr(file, 'data'):
                    image = Image.open(io.BytesIO(file.data)).convert('RGB')
                else:
                    image = Image.open(file).convert('RGB')

                # Predict
                pred_class, confidence, all_probs = predict_single(
                    image,
                    st.session_state.model,
                    st.session_state.processor,
                    st.session_state.device
                )
                
                results.append({
                    'filename': file.name,
                    'prediction': pred_class,
                    'confidence': confidence,
                    'no_helmet_prob': all_probs[0],
                    'with_helmet_prob': all_probs[1]
                })
                
                # Store images for XAI
                if show_batch_xai:
                    images_for_xai.append({
                        'image': image,
                        'filename': file.name,
                        'pred_class': pred_class,
                        'confidence': confidence
                    })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Processing complete!")
            st.session_state.batch_results = results
            st.session_state.images_for_xai = images_for_xai
            
            # ==================== RESULTS SECTION ====================
            st.markdown("---")
            st.markdown("## 📊 Batch Results")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total = len(results)
            no_helmet = sum(1 for r in results if r['prediction'] == 0)
            with_helmet = sum(1 for r in results if r['prediction'] == 1)
            avg_conf = np.mean([r['confidence'] for r in results])
            
            with col1:
                st.metric("Total Images", total)
            with col2:
                st.metric("No Helmet", no_helmet, 
                         delta=f"{no_helmet/total*100:.1f}%",
                         delta_color="inverse")
            with col3:
                st.metric("With Helmet", with_helmet,
                         delta=f"{with_helmet/total*100:.1f}%",
                         delta_color="normal")
            with col4:
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            
            # # Visualization
            # st.markdown("### 📈 Distribution")
            # col_v1, col_v2 = st.columns(2)
            
            # with col_v1:
            #     fig_pie = plot_batch_statistics(results, 'pie')
            #     st.pyplot(fig_pie)
            
            # with col_v2:
            #     fig_bar = plot_batch_statistics(results, 'bar')
            #     st.pyplot(fig_bar)
            
            # Detailed Results - Table with Images
            st.markdown("### 📋 Detailed Results")
            
            # Table with thumbnail images
            for idx, result in enumerate(results):
                # Load image
                file = uploaded_files[idx]
                if hasattr(file, 'data'):
                    img = Image.open(io.BytesIO(file.data))
                else:
                    img = Image.open(file)
                
                # Create row with image + info
                col_img, col_info = st.columns([1, 4])
                
                with col_img:
                    # Thumbnail image (small)
                    st.image(img, use_container_width=True)
                
                with col_info:
                    # Prediction info
                    pred_label = 'With Helmet' if result['prediction'] == 1 else 'No Helmet'
                    pred_color = '#28a745' if result['prediction'] == 1 else '#dc3545'
                    pred_icon = '✅' if result['prediction'] == 1 else '⚠️'
                    
                    st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 8px; border-left: 4px solid {pred_color}; height: 100%;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 1.1rem; font-weight: bold; color: {pred_color};">
                                    {pred_icon} {pred_label}
                                </div>
                                <div style="font-size: 0.85rem; color: #666; margin-top: 0.2rem;">
                                    📁 {result['filename']}
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.3rem; font-weight: bold; color: {pred_color};">
                                    {result['confidence']:.1%}
                                </div>
                                <div style="font-size: 0.75rem; color: #666;">Confidence</div>
                            </div>
                        </div>
                        <div style="display: flex; gap: 1.5rem; margin-top: 0.8rem; font-size: 0.9rem;">
                            <div>
                                <span style="color: #666;">No Helmet:</span>
                                <strong style="color: #dc3545;">{result['no_helmet_prob']:.1%}</strong>
                            </div>
                            <div>
                                <span style="color: #666;">With Helmet:</span>
                                <strong style="color: #28a745;">{result['with_helmet_prob']:.1%}</strong>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                
                
                # Add divider between rows
                if idx < len(results) - 1:
                    st.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
            
            # Create DataFrame for download
            df = pd.DataFrame(results)
            df['prediction_label'] = df['prediction'].map({0: 'No Helmet', 1: 'With Helmet'})
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results (CSV)",
                csv,
                "helmet_detection_results.csv",
                "text/csv",
                use_container_width=True
            )
            # Visualization
            st.markdown("### 📈 Distribution")
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                fig_pie = plot_batch_statistics(results, 'pie')
                st.pyplot(fig_pie)
            
            with col_v2:
                fig_bar = plot_batch_statistics(results, 'bar')
                st.pyplot(fig_bar)
            
            # ==================== XAI SECTION ====================
            
            # ==================== XAI SECTION ====================
            if show_batch_xai and images_for_xai:
                st.markdown("---")
                st.markdown('<div class="xai-section">', unsafe_allow_html=True)
                st.markdown("## 🔍 Explainable AI - Attention Rollout")
                st.markdown("Visualizing model attention on selected samples")
                
                with st.spinner(f"Generating attention maps for {num_xai_samples} samples..."):
                    # Select samples (balanced)
                    no_helmet_idx = [i for i, img in enumerate(images_for_xai) 
                                    if img['pred_class'] == 0]
                    with_helmet_idx = [i for i, img in enumerate(images_for_xai) 
                                      if img['pred_class'] == 1]
                    
                    selected = []
                    n_per_class = num_xai_samples // 2
                    
                    if len(no_helmet_idx) >= n_per_class:
                        selected.extend(np.random.choice(no_helmet_idx, n_per_class, replace=False))
                    else:
                        selected.extend(no_helmet_idx)
                    
                    remaining = num_xai_samples - len(selected)
                    if len(with_helmet_idx) >= remaining:
                        selected.extend(np.random.choice(with_helmet_idx, remaining, replace=False))
                    else:
                        selected.extend(with_helmet_idx)
                    
                    # Generate attention maps
                    attention_results = []
                    xai_progress = st.progress(0)
                    xai_status = st.empty()
                    
                    for idx, img_idx in enumerate(selected):
                        img_data = images_for_xai[img_idx]
                        xai_status.text(f"Generating attention map {idx+1}/{len(selected)}")
                        
                        attention_map, pred_class, confidence = generate_attention_rollout(
                            img_data['image'],
                            st.session_state.model,
                            st.session_state.processor,
                            st.session_state.device
                        )
                        
                        attention_results.append({
                            'image': np.array(img_data['image']),
                            'attention_map': attention_map,
                            'pred_class': pred_class,
                            'confidence': confidence,
                            'filename': img_data['filename']
                        })
                        
                        xai_progress.progress((idx + 1) / len(selected))
                    
                    xai_status.text("✅ Attention maps generated!")
                    
                    # Display grid
                    st.markdown("### 🎨 Attention Visualization Grid")
                    fig_xai = create_attention_grid(attention_results)
                    st.pyplot(fig_xai)
                
                st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>Helmet Detection Dashboard</strong> | Powered by Vision Transformer & Streamlit</p>
    <p style="font-size: 0.9rem;">Model: ViT-Base/16 | Accuracy: 94.78% | Pretrained: ImageNet-21k</p>
</div>
""", unsafe_allow_html=True)