"""
GeoSense model with SegFormer head instead of custom decoder.
SegFormer is lighter and pre-trained on segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SegformerForSemanticSegmentation
from typing import Tuple, Optional, List


class SegFormerHead(nn.Module):
    """
    Lightweight segmentation head using SegFormer architecture.
    This is specifically designed for pixel-level tasks.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_size: Tuple[int, int] = (1024, 1024),
        num_classes: int = 5
    ):
        """
        Args:
            input_dim: Input feature dimension from encoder
            output_size: Target output size (H, W)
            num_classes: Number of output classes
        """
        super().__init__()
        self.output_size = output_size
        
        # SegFormer-style head: lightweight projection + upsampling
        # Much simpler than the custom decoder
        
        # Progressive upsampling (SegFormer approach)
        self.seg_head = nn.Sequential(
            # Project to smaller dimension
            nn.Conv2d(input_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Upsample 2x
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsample 2x
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample 2x
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample 2x
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Final prediction
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
        
        Returns:
            Logits (B, num_classes, H, W)
        """
        # Progressive upsampling
        x = self.seg_head(x)
        
        # Final classification
        logits = self.classifier(x)
        
        # Resize to target size if needed
        if logits.shape[2:] != self.output_size:
            logits = F.interpolate(
                logits,
                size=self.output_size,
                mode='bilinear',
                align_corners=False
            )
        
        return logits


class SegFormerFeatureFusion(nn.Module):
    """
    Fuses pre and post features using SegFormer-style projection.
    More efficient than the previous fusion method.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256
    ):
        super().__init__()
        
        self.fusion = nn.Sequential(
            # Project both features to same dimension
            nn.Conv2d(input_dim * 3, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        pre_features: torch.Tensor,
        post_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pre_features: Pre-disaster features (B, C, H, W)
            post_features: Post-disaster features (B, C, H, W)
        
        Returns:
            Fused features (B, output_dim, H, W)
        """
        # Compute difference
        diff = post_features - pre_features
        
        # Concatenate: [pre, post, diff]
        fused = torch.cat([pre_features, post_features, diff], dim=1)
        
        # Fuse with projection
        return self.fusion(fused)


class GeoSenseSegFormer(nn.Module):
    """
    GeoSense model with SegFormer head.
    Uses DINOv2 encoder + SegFormer segmentation head.
    """
    
    def __init__(
        self,
        dinov2_model: str = "facebook/dinov2-large",
        image_size: Tuple[int, int] = (1024, 1024),
        patch_size: int = 518,
        overlap: int = 64,
        freeze_encoder: bool = True,
        num_classes: int = 5
    ):
        """
        Args:
            dinov2_model: DINOv2 model identifier
            image_size: Input image size (H, W)
            patch_size: Patch size for processing
            overlap: Overlap between patches
            freeze_encoder: Whether to freeze DINOv2
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.image_size = image_size
        
        # Import encoder and processor here to avoid circular imports
        from geosense_model import PatchBasedDINOv2Encoder
        from patch_processor import PatchProcessor
        
        # Initialize encoder
        self.encoder = PatchBasedDINOv2Encoder(
            dinov2_model,
            patch_size=patch_size,
            overlap=overlap,
            freeze=freeze_encoder
        )
        
        # Feature fusion
        self.fusion = SegFormerFeatureFusion(
            input_dim=self.encoder.hidden_size,
            output_dim=256
        )
        
        # SegFormer head
        self.segformer_head = SegFormerHead(
            input_dim=256,
            output_size=image_size,
            num_classes=num_classes
        )
        
        print(f"\nGeoSense with SegFormer Head:")
        print(f"  Input size: {image_size}")
        print(f"  Encoder: {dinov2_model}")
        print(f"  Output classes: {num_classes}")
        print(f"  SegFormer head: lightweight decoder")
    
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
            logits: Output logits (B, num_classes, H, W)
            metadata: Processing metadata
        """
        # Extract features
        pre_features, pre_metadata = self.encoder(pre_image, self.image_size)
        post_features, post_metadata = self.encoder(post_image, self.image_size)
        
        # Fuse features
        fused_features = self.fusion(pre_features, post_features)
        
        # Decode with SegFormer head
        logits = self.segformer_head(fused_features)
        
        metadata = {
            'pre_metadata': pre_metadata,
            'post_metadata': post_metadata,
            'fused_shape': fused_features.shape
        }
        
        return logits, metadata
    
    def predict(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference with predictions and probabilities.
        
        Args:
            pre_image: Pre-disaster image (B, 3, H, W)
            post_image: Post-disaster image (B, 3, H, W)
            threshold: Confidence threshold
        
        Returns:
            preds: Predicted classes (B, H, W)
            probs: Class probabilities (B, num_classes, H, W)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(pre_image, post_image)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds, probs


# Factory function
def create_geosense_segformer(
    dinov2_size: str = "large",
    image_size: Tuple[int, int] = (1024, 1024),
    patch_overlap: int = 64,
    freeze_encoder: bool = True,
    **kwargs
) -> GeoSenseSegFormer:
    """
    Factory function to create GeoSense model with SegFormer head.
    
    Args:
        dinov2_size: "small", "base", "large", or "giant"
        image_size: Input image size
        patch_overlap: Overlap between patches
        freeze_encoder: Whether to freeze encoder
        **kwargs: Additional arguments
    
    Returns:
        Initialized model
    """
    dinov2_models = {
        "small": ("facebook/dinov2-small", 322),
        "base": ("facebook/dinov2-base", 322),
        "large": ("facebook/dinov2-large", 518),
        "giant": ("facebook/dinov2-giant", 518)
    }
    
    model_name, patch_size = dinov2_models[dinov2_size]
    
    model = GeoSenseSegFormer(
        dinov2_model=model_name,
        image_size=image_size,
        patch_size=patch_size,
        overlap=patch_overlap,
        freeze_encoder=freeze_encoder,
        **kwargs
    )
    
    return model


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Testing GeoSense with SegFormer Head")
    print("=" * 70)
    
    # Create model
    model = create_geosense_segformer(
        dinov2_size="large",
        image_size=(1024, 1024),
        patch_overlap=64,
        freeze_encoder=True
    )
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    pre_img = torch.randn(batch_size, 3, 1024, 1024)
    post_img = torch.randn(batch_size, 3, 1024, 1024)
    
    model.to('cuda')
    pre_img = pre_img.to('cuda')
    post_img = post_img.to('cuda')
    
    logits, metadata = model(pre_img, post_img)
    print(f"✓ Output logits shape: {logits.shape}")
    print(f"✓ Expected: torch.Size([{batch_size}, 5, 1024, 1024])")
    
    # Get predictions
    preds, probs = model.predict(pre_img, post_img)
    print(f"✓ Predictions shape: {preds.shape}")
    print(f"✓ Probabilities shape: {probs.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("\n" + "=" * 70)
    print("✓ SegFormer model test completed successfully!")
    print("=" * 70)