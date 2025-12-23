"""
Multi-Class Building Damage Assessment Model
Outputs color-coded heatmap with damage severity levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import cv2


# Damage categories (matching XBD dataset)
DAMAGE_CLASSES = {
    0: {'name': 'no-damage', 'color': (0, 255, 0), 'label': 'No Damage'},      # Green
    1: {'name': 'minor-damage', 'color': (255, 255, 0), 'label': 'Minor'},     # Yellow
    2: {'name': 'major-damage', 'color': (255, 165, 0), 'label': 'Major'},     # Orange
    3: {'name': 'destroyed', 'color': (255, 0, 0), 'label': 'Destroyed'},      # Red
    4: {'name': 'background', 'color': (128, 128, 128), 'label': 'Background'} # Gray
}

NUM_DAMAGE_CLASSES = 5  # 4 damage levels + background


class MultiClassChangeDetectionDecoder(nn.Module):
    """
    Multi-class decoder for building damage assessment.
    Outputs 5-channel prediction (4 damage levels + background).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_size: Tuple[int, int] = (1024, 1024),
        num_classes: int = NUM_DAMAGE_CLASSES
    ):
        super().__init__()
        self.output_size = output_size
        self.num_classes = num_classes
        
        # Progressive upsampling decoder
        self.decoder = nn.ModuleList([
            # Stage 1: input_dim -> 512
            nn.Sequential(
                nn.Conv2d(input_dim, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            # Stage 2: 512 -> 256
            nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            # Stage 3: 256 -> 128
            nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            # Stage 4: 128 -> 64
            nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        ])
        
        # Multi-class prediction head
        self.prediction_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)  # 5 channels output
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
        
        Returns:
            Multi-class predictions (B, num_classes, H, W)
        """
        for stage in self.decoder:
            x = stage(x)
        
        logits = self.prediction_head(x)
        
        # Resize to target size
        if logits.shape[2:] != self.output_size:
            logits = F.interpolate(
                logits,
                size=self.output_size,
                mode='bilinear',
                align_corners=False
            )
        
        return logits


class GeoSenseMultiClassModel(nn.Module):
    """
    Multi-class damage assessment model.
    Outputs color-coded damage heatmap.
    """
    
    def __init__(
        self,
        dinov2_model: str = "facebook/dinov2-large",
        image_size: Tuple[int, int] = (1024, 1024),
        patch_size: int = 518,
        overlap: int = 64,
        fusion_type: str = "concat_diff",
        freeze_encoder: bool = True
    ):
        super().__init__()
        
        from geosense_model import PatchBasedDINOv2Encoder, FeatureFusion
        
        self.image_size = image_size
        
        # Encoder
        self.encoder = PatchBasedDINOv2Encoder(
            dinov2_model,
            patch_size=patch_size,
            overlap=overlap,
            freeze=freeze_encoder
        )
        
        # Fusion
        self.fusion = FeatureFusion(
            input_dim=self.encoder.hidden_size,
            fusion_type=fusion_type
        )
        
        # Multi-class decoder
        self.decoder = MultiClassChangeDetectionDecoder(
            input_dim=self.encoder.hidden_size,
            output_size=image_size,
            num_classes=NUM_DAMAGE_CLASSES
        )
        
        print(f"\nMulti-Class GeoSense Model:")
        print(f"  Output classes: {NUM_DAMAGE_CLASSES}")
        print(f"  Classes: {[DAMAGE_CLASSES[i]['label'] for i in range(NUM_DAMAGE_CLASSES)]}")
    
    def forward(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pre_image: Pre-disaster image (B, 3, H, W)
            post_image: Post-disaster image (B, 3, H, W)
        
        Returns:
            logits: Class logits (B, num_classes, H, W)
            metadata: Processing metadata
        """
        # Extract features
        pre_features, pre_metadata = self.encoder(pre_image, self.image_size)
        post_features, post_metadata = self.encoder(post_image, self.image_size)
        
        # Fuse
        fused_features = self.fusion(pre_features, post_features)
        
        # Decode to multi-class mask
        logits = self.decoder(fused_features)
        
        metadata = {
            'pre_metadata': pre_metadata,
            'post_metadata': post_metadata,
            'feature_shape': fused_features.shape
        }
        
        return logits, metadata
    
    def predict(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference with class predictions and probabilities.
        
        Args:
            pre_image: Pre-disaster image (B, 3, H, W)
            post_image: Post-disaster image (B, 3, H, W)
        
        Returns:
            class_predictions: Predicted class indices (B, H, W)
            class_probabilities: Class probabilities (B, num_classes, H, W)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(pre_image, post_image)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds, probs


class MultiClassVisualizer:
    """Visualization tools for multi-class damage assessment."""
    
    @staticmethod
    def create_color_coded_heatmap(
        class_predictions: np.ndarray,
        alpha: float = 1.0
    ) -> np.ndarray:
        """
        Create color-coded heatmap from class predictions.
        
        Args:
            class_predictions: Class indices (H, W)
            alpha: Transparency (0-1)
        
        Returns:
            RGB heatmap (H, W, 3)
        """
        h, w = class_predictions.shape
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, class_info in DAMAGE_CLASSES.items():
            mask = class_predictions == class_id
            heatmap[mask] = class_info['color']
        
        if alpha < 1.0:
            heatmap = (heatmap * alpha).astype(np.uint8)
        
        return heatmap
    
    @staticmethod
    def overlay_heatmap_on_image(
        image: np.ndarray,
        class_predictions: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Overlay color-coded heatmap on image.
        
        Args:
            image: Original image (H, W, 3)
            class_predictions: Class indices (H, W)
            alpha: Heatmap transparency
        
        Returns:
            Overlayed image
        """
        heatmap = MultiClassVisualizer.create_color_coded_heatmap(class_predictions)
        
        # Blend
        result = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
        
        return result
    
    @staticmethod
    def create_legend() -> np.ndarray:
        """Create legend for damage classes."""
        legend_height = 50 * NUM_DAMAGE_CLASSES
        legend_width = 300
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
        
        y_offset = 10
        for class_id in range(NUM_DAMAGE_CLASSES):
            class_info = DAMAGE_CLASSES[class_id]
            
            # Color box
            cv2.rectangle(
                legend,
                (10, y_offset),
                (50, y_offset + 30),
                class_info['color'],
                -1
            )
            cv2.rectangle(
                legend,
                (10, y_offset),
                (50, y_offset + 30),
                (0, 0, 0),
                2
            )
            
            # Text
            cv2.putText(
                legend,
                class_info['label'],
                (60, y_offset + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )
            
            y_offset += 50
        
        return legend
    
    @staticmethod
    def visualize_multiclass_results(
        pre_image: np.ndarray,
        post_image: np.ndarray,
        pred_classes: np.ndarray,
        gt_classes: np.ndarray = None,
        class_probs: np.ndarray = None,
        save_path: str = None
    ):
        """
        Comprehensive visualization for multi-class predictions.
        
        Args:
            pre_image: Pre-disaster image
            post_image: Post-disaster image  
            pred_classes: Predicted class indices (H, W)
            gt_classes: Ground truth classes (optional)
            class_probs: Class probabilities (optional)
            save_path: Path to save visualization
        """
        # Determine layout
        num_plots = 3  # pre, post, prediction
        if gt_classes is not None:
            num_plots += 1
        if class_probs is not None:
            num_plots += 1  # Max probability map
        
        fig = plt.figure(figsize=(6*num_plots, 6))
        
        plot_idx = 1
        
        # Pre-disaster
        plt.subplot(1, num_plots, plot_idx)
        plt.imshow(pre_image)
        plt.title('Pre-Disaster', fontsize=14, fontweight='bold')
        plt.axis('off')
        plot_idx += 1
        
        # Post-disaster
        plt.subplot(1, num_plots, plot_idx)
        plt.imshow(post_image)
        plt.title('Post-Disaster', fontsize=14, fontweight='bold')
        plt.axis('off')
        plot_idx += 1
        
        # Prediction overlay
        plt.subplot(1, num_plots, plot_idx)
        pred_overlay = MultiClassVisualizer.overlay_heatmap_on_image(
            post_image, pred_classes, alpha=0.6
        )
        plt.imshow(pred_overlay)
        plt.title('Predicted Damage\n(Color-Coded)', fontsize=14, fontweight='bold')
        plt.axis('off')
        plot_idx += 1
        
        # Ground truth if available
        if gt_classes is not None:
            plt.subplot(1, num_plots, plot_idx)
            gt_overlay = MultiClassVisualizer.overlay_heatmap_on_image(
                post_image, gt_classes, alpha=0.6
            )
            plt.imshow(gt_overlay)
            plt.title('Ground Truth', fontsize=14, fontweight='bold')
            plt.axis('off')
            plot_idx += 1
        
        # Confidence map
        if class_probs is not None:
            plt.subplot(1, num_plots, plot_idx)
            max_prob = np.max(class_probs, axis=0)
            plt.imshow(max_prob, cmap='jet', vmin=0, vmax=1)
            plt.colorbar(label='Confidence', fraction=0.046)
            plt.title('Prediction Confidence', fontsize=14, fontweight='bold')
            plt.axis('off')
            plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        # Create separate legend
        legend = MultiClassVisualizer.create_legend()
        plt.figure(figsize=(4, 5))
        plt.imshow(legend)
        plt.title('Damage Classification Legend', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            legend_path = save_path.replace('.png', '_legend.png')
            plt.savefig(legend_path, dpi=150, bbox_inches='tight')
            print(f"Legend saved to {legend_path}")
        
        plt.show()
    
    @staticmethod
    def create_damage_statistics(
        class_predictions: np.ndarray,
        building_mask: np.ndarray = None
    ) -> Dict:
        """
        Calculate damage statistics from predictions.
        
        Args:
            class_predictions: Predicted classes (H, W)
            building_mask: Mask of building pixels (optional)
        
        Returns:
            Dictionary with statistics
        """
        if building_mask is not None:
            # Only count building pixels
            class_predictions = class_predictions[building_mask > 0]
        
        stats = {}
        total_pixels = len(class_predictions.flatten())
        
        for class_id in range(NUM_DAMAGE_CLASSES):
            class_pixels = np.sum(class_predictions == class_id)
            percentage = (class_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            stats[DAMAGE_CLASSES[class_id]['name']] = {
                'pixels': int(class_pixels),
                'percentage': float(percentage)
            }
        
        return stats


# Factory function
def create_multiclass_geosense_model(
    dinov2_size: str = "large",
    image_size: Tuple[int, int] = (1024, 1024),
    patch_overlap: int = 64,
    **kwargs
) -> GeoSenseMultiClassModel:
    """Create multi-class damage assessment model."""
    
    dinov2_models = {
        "small": ("facebook/dinov2-small", 322),
        "base": ("facebook/dinov2-base", 322),
        "large": ("facebook/dinov2-large", 518),
        "giant": ("facebook/dinov2-giant", 518)
    }
    
    model_name, patch_size = dinov2_models[dinov2_size]
    
    model = GeoSenseMultiClassModel(
        dinov2_model=model_name,
        image_size=image_size,
        patch_size=patch_size,
        overlap=patch_overlap,
        **kwargs
    )
    
    return model


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Class Building Damage Assessment Model")
    print("=" * 70)
    
    # Create model
    model = create_multiclass_geosense_model(
        dinov2_size='large',
        image_size=(1024, 1024),
        patch_overlap=64,
        fusion_type='concat_diff'
    )
    
    # Test
    pre_img = torch.randn(2, 3, 1024, 1024)
    post_img = torch.randn(2, 3, 1024, 1024)
    
    logits, metadata = model(pre_img, post_img)
    print(f"\nOutput shape: {logits.shape}")  # (2, 5, 1024, 1024)
    
    # Get predictions
    preds, probs = model.predict(pre_img, post_img)
    print(f"Predictions shape: {preds.shape}")  # (2, 1024, 1024)
    print(f"Probabilities shape: {probs.shape}")  # (2, 5, 1024, 1024)
    
    print("\n" + "=" * 70)
    print("✓ Multi-class model ready for color-coded heatmaps!")
    print("=" * 70)
