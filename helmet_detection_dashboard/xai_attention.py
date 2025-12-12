"""
XAI Attention Rollout Module
Generate attention maps for explainability
"""

import torch
import numpy as np
import cv2
from PIL import Image

class VitAttentionRollout:
    """
    Attention Rollout for Vision Transformer
    """
    
    def __init__(self, model, head_fusion="mean", discard_ratio=0.1):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
    
    def __call__(self, input_tensor):
        """Generate attention rollout map"""
        
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.vit(input_tensor, output_attentions=True)
            attentions = outputs.attentions
        
        result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
        
        for attention in attentions:
            if self.head_fusion == "mean":
                attention_heads_fused = attention.mean(dim=1)
            elif self.head_fusion == "max":
                attention_heads_fused = attention.max(dim=1)[0]
            else:
                attention_heads_fused = attention.min(dim=1)[0]
            
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(
                int(flat.size(-1) * self.discard_ratio),
                dim=-1,
                largest=False
            )
            flat[0, indices[0]] = 0
            
            I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            
            result = torch.matmul(a, result)
        
        mask = result[0, 0, 1:]
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width)
        mask = mask / mask.max()
        
        return mask.cpu().numpy()


def generate_attention_rollout(image, model, processor, device):
    """
    Generate attention rollout visualization
    
    Args:
        image: PIL Image
        model: ViT model
        processor: ViT processor
        device: torch device
        
    Returns:
        attention_map: Attention map
        pred_class: Predicted class
        confidence: Confidence
    """
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    attention_rollout = VitAttentionRollout(model)
    attention_map = attention_rollout(pixel_values)
    
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    attention_map_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)
    attention_map_resized = cv2.GaussianBlur(attention_map_resized, (5, 5), 0)
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / \
                           (attention_map_resized.max() - attention_map_resized.min() + 1e-8)
    
    return attention_map_resized, pred_class, confidence


def create_attention_overlay(image, attention_map, alpha=0.4):
    """Create overlay of attention on image"""
    
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    
    return overlay