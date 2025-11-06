"""
Grad-CAM Implementation for Neural Network Explainability
Provides visual explanations for deep learning models
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import cv2

class GradCAMExplainer:
    """Grad-CAM implementation for model explainability"""
    
    def __init__(self, model, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )
    
    def generate_heatmap(self, image: np.ndarray, class_idx: int, 
                        eps: float = 1e-8) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a given image and class
        
        Args:
            image: Input image/data
            class_idx: Class index to generate heatmap for
            eps: Small value to avoid division by zero
            
        Returns:
            Heatmap array
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            loss = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the convolution outputs with pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + eps)
        
        return heatmap.numpy()
    
    def visualize_heatmap(self, heatmap: np.ndarray, original_image: np.ndarray,
                         alpha: float = 0.4) -> plt.Figure:
        """
        Visualize Grad-CAM heatmap overlayed on original image
        
        Args:
            heatmap: Generated heatmap
            original_image: Original input image
            alpha: Transparency for heatmap overlay
            
        Returns:
            matplotlib figure
        """
        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], 
                                             original_image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        superimposed = heatmap_colored * alpha + original_image
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(original_image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(heatmap_resized, cmap='jet')
        ax2.set_title('Grad-CAM Heatmap')
        ax2.axis('off')
        
        ax3.imshow(superimposed)
        ax3.set_title('Superimposed')
        ax3.axis('off')
        
        plt.tight_layout()
        return fig

# Example usage for ECG data
class ECG_GradCAM(GradCAMExplainer):
    """Specialized Grad-CAM for ECG signal analysis"""
    
    def generate_ecg_heatmap(self, ecg_signal: np.ndarray, class_idx: int) -> np.ndarray:
        """
        Generate Grad-CAM for ECG signals
        
        Args:
            ecg_signal: ECG time-series data
            class_idx: Prediction class index
            
        Returns:
            Temporal importance heatmap
        """
        # Reshape ECG signal for model input
        ecg_reshaped = ecg_signal.reshape(1, -1, 1)
        
        # Generate heatmap using parent method
        heatmap = self.generate_heatmap(ecg_reshaped, class_idx)
        
        return heatmap
    
    def plot_ecg_with_importance(self, ecg_signal: np.ndarray, 
                               importance_weights: np.ndarray) -> plt.Figure:
        """
        Plot ECG signal with importance weights
        
        Args:
            ecg_signal: Original ECG signal
            importance_weights: Grad-CAM importance scores
            
        Returns:
            matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot original ECG
        ax1.plot(ecg_signal, color='blue', linewidth=1)
        ax1.set_title('ECG Signal')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        
        # Plot importance weights
        ax2.plot(importance_weights, color='red', linewidth=2)
        ax2.set_title('Feature Importance (Grad-CAM)')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Importance')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig