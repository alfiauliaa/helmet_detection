"""
Visualization Module
Plotting functions for results and attention maps
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from xai_attention import create_attention_overlay

def plot_single_prediction(probs, class_names):
    """Plot probability distribution"""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#ff6b6b', '#6bcf7f']
    bars = ax.bar(class_names, probs, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_ylabel('Probability', fontweight='bold', fontsize=12)
    ax.set_title('Prediction Confidence', fontweight='bold', fontsize=14)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.2%}', ha='center', va='bottom',
                fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_single_attention(image, attention_map, pred_class, confidence):
    """Plot single image attention visualization"""
    
    class_names = ['No Helmet', 'With Helmet']
    pred_name = class_names[pred_class]
    color = 'darkgreen' if pred_class == 1 else 'darkred'
    
    # Create overlay
    overlay = create_attention_overlay(image, attention_map, alpha=0.4)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontweight='bold', fontsize=12)
    axes[0].axis('off')
    
    # Attention heatmap only
    axes[1].imshow(attention_map, cmap='jet')
    axes[1].set_title('Attention Heatmap', fontweight='bold', fontsize=12)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay: {pred_name} ({confidence:.1%})', 
                     fontweight='bold', fontsize=12, color=color)
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def plot_batch_statistics(results, chart_type='pie'):
    """Plot batch statistics"""
    
    no_helmet = sum(1 for r in results if r['prediction'] == 0)
    with_helmet = sum(1 for r in results if r['prediction'] == 1)
    
    counts = [no_helmet, with_helmet]
    labels = ['No Helmet', 'With Helmet']
    colors = ['#ff6b6b', '#6bcf7f']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if chart_type == 'pie':
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, autopct='%1.1f%%',
            startangle=90, colors=colors, explode=(0.05, 0.05),
            shadow=True, textprops={'fontweight': 'bold', 'fontsize': 12}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
        ax.set_title('Distribution', fontweight='bold', fontsize=14)
    else:
        bars = ax.bar(labels, counts, color=colors, edgecolor='black', alpha=0.8)
        ax.set_ylabel('Count', fontweight='bold', fontsize=12)
        ax.set_title('Results', fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = count / sum(counts) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.02,
                   f'{count}\n({percentage:.1f}%)', ha='center', va='bottom',
                   fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    return fig


def create_attention_grid(attention_results, n_cols=4):
    """Create grid of attention visualizations"""
    
    n_samples = len(attention_results)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    class_names = ['No Helmet', 'With Helmet']
    
    for idx, result in enumerate(attention_results):
        overlay = create_attention_overlay(result['image'], result['attention_map'])
        
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        ax.imshow(overlay)
        
        pred_name = class_names[result['pred_class']]
        color = 'darkgreen' if result['pred_class'] == 1 else 'darkred'
        
        title = f"{result['filename']}\n{pred_name}\n{result['confidence']:.2%}"
        ax.set_title(title, fontweight='bold', fontsize=10, color=color)
        ax.axis('off')
    
    plt.suptitle('Attention Rollout Visualization', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig