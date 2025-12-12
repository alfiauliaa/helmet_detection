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
import tempfile

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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
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
    
    # Model path
    model_path = st.text_input(
        "Model Path",
        value="models/best_vit_model.pth",
        help="Path to the saved ViT model"
    )
    
    # Device selection
    device_option = st.radio(
        "Device",
        options=["Auto", "CPU", "CUDA"],
        index=0,
        help="Select computation device"
    )
    
    # Load model button
    if st.button("🚀 Load Model", type="primary", use_container_width=True):
        with st.spinner("Loading Vision Transformer model..."):
            try:
                # Determine device
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
                
                # Load model
                model, processor = load_model(model_path, device)
                
                st.session_state.model = model
                st.session_state.processor = processor
                st.session_state.device = device
                
                st.success(f"✅ Model loaded on {device}!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Model status
    st.markdown("---")
    st.markdown("### 📊 Model Status")
    
    if st.session_state.model is not None:
        st.markdown('<div class="success-box">✅ Model Loaded</div>', 
                    unsafe_allow_html=True)
        st.info(f"Device: **{st.session_state.device}**")
    else:
        st.warning("⚠️ Model Not Loaded")
    
    # About
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Model**: ViT-Base/16  
    **Accuracy**: 94.78%  
    **Pretrained**: ImageNet-21k  
    **Classes**: No Helmet, With Helmet
    
    ---
    
    **Features**:
    - Single image prediction with XAI
    - Batch processing with folder upload
    - Attention rollout visualization
    """)

# Main content
st.markdown('<p class="main-header">🪖 Helmet Detection Dashboard</p>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Vision Transformer with Explainable AI</p>', 
            unsafe_allow_html=True)

# Check if model is loaded
if st.session_state.model is None:
    st.info("👈 Please load the model from the sidebar to get started!")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["📸 Single Image", "📁 Batch Processing"])

# ==================== TAB 1: SINGLE IMAGE ====================
with tab1:
    st.markdown("## Single Image Prediction with XAI")
    
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
            
            # XAI Option
            show_xai = st.checkbox("🔍 Show Attention Map (XAI)", value=True, 
                                   help="Visualize where the model focuses")
            
            if st.button("🔮 Predict", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    start_time = time.time()
                    
                    # Predict
                    pred_class, confidence, all_probs = predict_single(
                        image,
                        st.session_state.model,
                        st.session_state.processor,
                        st.session_state.device
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
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">
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
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract ZIP
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find all images
                image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                uploaded_files = []
                
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if any(file.endswith(ext) for ext in image_extensions):
                            file_path = os.path.join(root, file)
                            # Create file-like object
                            class FileObj:
                                def __init__(self, path):
                                    self.name = os.path.basename(path)
                                    self.path = path
                            uploaded_files.append(FileObj(file_path))
                
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
                if hasattr(file, 'path'):  # ZIP file
                    image = Image.open(file.path).convert('RGB')
                else:  # Uploaded file
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
            
            # Visualization
            st.markdown("### 📈 Distribution")
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                fig_pie = plot_batch_statistics(results, 'pie')
                st.pyplot(fig_pie)
            
            with col_v2:
                fig_bar = plot_batch_statistics(results, 'bar')
                st.pyplot(fig_bar)
            
            # Detailed table
            st.markdown("### 📋 Detailed Results")
            df = pd.DataFrame(results)
            df['prediction_label'] = df['prediction'].map({0: 'No Helmet', 1: 'With Helmet'})
            
            # Format columns
            display_df = df[['filename', 'prediction_label', 'confidence', 
                           'no_helmet_prob', 'with_helmet_prob']].copy()
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
            display_df['no_helmet_prob'] = display_df['no_helmet_prob'].apply(lambda x: f"{x:.2%}")
            display_df['with_helmet_prob'] = display_df['with_helmet_prob'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(display_df, use_container_width=True, height=300)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results (CSV)",
                csv,
                "helmet_detection_results.csv",
                "text/csv",
                use_container_width=True
            )
            
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