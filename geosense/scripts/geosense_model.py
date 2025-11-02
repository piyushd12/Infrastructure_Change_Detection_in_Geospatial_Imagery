import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
# from segment_anything import sam_model_registry
from typing import Tuple, Optional, List
import math


class PatchBasedDINOv2Encoder(nn.Module):
    """
    Patch-based DINOv2 encoder that processes large images in patches.
    No information loss from resizing!
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        patch_size: int = 518,
        overlap: int = 64,
        freeze: bool = True
    ):
        """
        Args:
            model_name: HuggingFace model identifier for DINOv2
            patch_size: Size of patches (518 for large/giant, 322 for base/small)
            overlap: Overlap between patches in pixels
            freeze: If True, freeze all encoder weights
        """
        super().__init__()
        
        # Load DINOv2 model
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.patch_size = patch_size
        self.overlap = overlap
        
        # DINOv2 internal patch size (typically 14)
        self.dinov2_patch_size = self.encoder.config.patch_size
        
        # Calculate output spatial dimensions
        # DINOv2 outputs (num_patches_per_side)^2 patches
        self.output_spatial_dim = patch_size // self.dinov2_patch_size
        
        # Freeze weights if specified
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
        print(f"DINOv2 Encoder Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Patch size for processing: {patch_size}x{patch_size}")
        print(f"  DINOv2 internal patch size: {self.dinov2_patch_size}")
        print(f"  Output spatial dim per patch: {self.output_spatial_dim}x{self.output_spatial_dim}")
        print(f"  Frozen: {freeze}")
    
    def process_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Process patches through DINOv2 encoder.
        
        Args:
            patches: Tensor of patches (num_patches, C, H, W)
        
        Returns:
            Features (num_patches, num_tokens, hidden_dim)
        """
        with torch.set_grad_enabled(self.training and self._is_frozen() == False):
            outputs = self.encoder(patches, return_dict=True)
            # Get patch embeddings (excluding CLS token)
            features = outputs.last_hidden_state[:, 1:, :]
        
        return features
    
    def forward(
        self,
        x: torch.Tensor,
        image_size: Tuple[int, int] = (1024, 1024)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with patch-based processing.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            image_size: Original image size
        
        Returns:
            features: Spatial features (B, hidden_dim, H_out, W_out)
            metadata: Patch processing metadata
        """
        from patch_processor import PatchProcessor
        
        B, C, H, W = x.shape
        
        # Initialize patch processor
        processor = PatchProcessor(
            patch_size=self.patch_size,
            overlap=self.overlap,
            original_size=(H, W)
        )
        
        # Extract patches
        patches, metadata = processor.extract_patches(x)
        # patches: (B * num_patches, C, patch_size, patch_size)
        
        # Process through DINOv2
        patch_features = self.process_patches(patches)
        # patch_features: (B * num_patches, num_tokens, hidden_dim)
        
        # Reshape to spatial format
        # num_tokens = output_spatial_dim^2
        num_patches_total = patches.shape[0]
        patch_features = patch_features.reshape(
            num_patches_total,
            self.output_spatial_dim,
            self.output_spatial_dim,
            self.hidden_size
        )
        patch_features = patch_features.permute(0, 3, 1, 2)
        # patch_features: (B * num_patches, hidden_dim, H_feat, W_feat)
        
        # Reconstruct full feature map
        features = processor.reconstruct_from_patches(
            patch_features,
            metadata,
            use_averaging=True
        )
        # features: (B, hidden_dim, H_out, W_out)
        
        return features, metadata
    
    def _is_frozen(self) -> bool:
        """Check if encoder is frozen."""
        return not any(p.requires_grad for p in self.encoder.parameters())


class FeatureFusion(nn.Module):
    """
    Fuses features from pre and post disaster images.
    Works with spatial feature maps.
    """
    
    def __init__(
        self,
        input_dim: int,
        fusion_type: str = "concat_diff",
    ):
        """
        Args:
            input_dim: Dimension of input features
            fusion_type: Type of fusion strategy
        """
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            output_dim = input_dim * 2
        elif fusion_type == "diff":
            output_dim = input_dim
        elif fusion_type == "concat_diff":
            output_dim = input_dim * 3
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Projection with spatial convolutions
        self.projection = nn.Sequential(
            nn.Conv2d(output_dim, input_dim, kernel_size=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU()
        )
    
    def forward(
        self,
        pre_features: torch.Tensor,
        post_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pre_features: Features from pre-disaster image (B, C, H, W)
            post_features: Features from post-disaster image (B, C, H, W)
        
        Returns:
            Fused features (B, C, H, W)
        """
        if self.fusion_type == "concat":
            fused = torch.cat([pre_features, post_features], dim=1)
        elif self.fusion_type == "diff":
            fused = post_features - pre_features
        elif self.fusion_type == "concat_diff":
            diff = post_features - pre_features
            fused = torch.cat([pre_features, post_features, diff], dim=1)
        
        return self.projection(fused)


class ChangeDetectionDecoder(nn.Module):
    """
    Lightweight decoder for change detection.
    Upsamples features and generates change mask.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_size: Tuple[int, int] = (1024, 1024),
        num_classes: int = 1
    ):
        """
        Args:
            input_dim: Input feature dimension
            output_size: Output mask size (H, W)
            num_classes: Number of output classes (1 for binary)
        """
        super().__init__()
        self.output_size = output_size
        
        # Progressive upsampling with skip connections
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
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
        
        Returns:
            Change mask (B, num_classes, output_H, output_W)
        """
        # Progressive upsampling
        for stage in self.decoder:
            x = stage(x)
        
        # Final prediction
        mask = self.prediction_head(x)
        
        # Resize to target output size
        if mask.shape[2:] != self.output_size:
            mask = F.interpolate(
                mask,
                size=self.output_size,
                mode='bilinear',
                align_corners=False
            )
        
        return mask


class GeoSenseModel(nn.Module):
    """
    Complete GeoSense change detection model with patch-based processing.
    Processes large images without information loss!
    """
    
    def __init__(
        self,
        dinov2_model: str = "facebook/dinov2-large",
        image_size: Tuple[int, int] = (1024, 1024),
        patch_size: int = 518,
        overlap: int = 64,
        fusion_type: str = "concat_diff",
        freeze_encoder: bool = True,
        num_classes: int = 1
    ):
        """
        Args:
            dinov2_model: DINOv2 model identifier
            image_size: Input image size (H, W)
            patch_size: Patch size for DINOv2 (518 for large/giant)
            overlap: Overlap between patches
            fusion_type: Feature fusion strategy
            freeze_encoder: Whether to freeze DINOv2 encoder
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.image_size = image_size
        
        # Initialize patch-based encoder
        self.encoder = PatchBasedDINOv2Encoder(
            dinov2_model,
            patch_size=patch_size,
            overlap=overlap,
            freeze=freeze_encoder
        )
        
        # Initialize feature fusion
        self.fusion = FeatureFusion(
            input_dim=self.encoder.hidden_size,
            fusion_type=fusion_type
        )
        
        # Initialize decoder
        self.decoder = ChangeDetectionDecoder(
            input_dim=self.encoder.hidden_size,
            output_size=image_size,
            num_classes=num_classes
        )
        
        print(f"\nGeoSense Model Configuration:")
        print(f"  Input size: {image_size}")
        print(f"  Patch size: {patch_size}")
        print(f"  Overlap: {overlap}")
        print(f"  Fusion type: {fusion_type}")
        print(f"  Output classes: {num_classes}")
    
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
            masks: Change detection masks (B, num_classes, H, W)
            metadata: Processing metadata
        """
        # Extract features from both images using patches
        pre_features, pre_metadata = self.encoder(pre_image, self.image_size)
        post_features, post_metadata = self.encoder(post_image, self.image_size)
        
        # Fuse features
        fused_features = self.fusion(pre_features, post_features)
        
        # Decode to change mask
        masks = self.decoder(fused_features)
        
        metadata = {
            'pre_metadata': pre_metadata,
            'post_metadata': post_metadata,
            'feature_shape': fused_features.shape
        }
        
        return masks, metadata
    
    def predict(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Inference method with thresholding.
        
        Args:
            pre_image: Pre-disaster image (B, 3, H, W)
            post_image: Post-disaster image (B, 3, H, W)
            threshold: Threshold for binary mask
        
        Returns:
            Binary change mask (B, 1, H, W)
        """
        self.eval()
        with torch.no_grad():
            masks, _ = self.forward(pre_image, post_image)
            binary_masks = (torch.sigmoid(masks) > threshold).float()
        
        return binary_masks


# Model configuration helper
def create_geosense_model(
    dinov2_size: str = "large",  # "small", "base", "large", "giant"
    image_size: Tuple[int, int] = (1024, 1024),
    patch_overlap: int = 64,
    **kwargs
) -> GeoSenseModel:
    """
    Factory function to create GeoSense model with different configurations.
    
    Args:
        dinov2_size: Size of DINOv2 model
        image_size: Input image size
        patch_overlap: Overlap between patches
        **kwargs: Additional arguments for GeoSenseModel
    
    Returns:
        Initialized GeoSense model
    """
    dinov2_models = {
        "small": ("facebook/dinov2-small", 322),
        "base": ("facebook/dinov2-base", 322),
        "large": ("facebook/dinov2-large", 518),
        "giant": ("facebook/dinov2-giant", 518)
    }
    
    model_name, patch_size = dinov2_models[dinov2_size]
    
    model = GeoSenseModel(
        dinov2_model=model_name,
        image_size=image_size,
        patch_size=patch_size,
        overlap=patch_overlap,
        **kwargs
    )
    
    return model


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Patch-Based GeoSense Model")
    print("=" * 60)
    
    # Create model
    model = create_geosense_model(
        dinov2_size="large",
        image_size=(1024, 1024),
        patch_overlap=64,
        fusion_type="concat_diff",
        freeze_encoder=True
    )
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    pre_img = torch.randn(batch_size, 3, 1024, 1024)
    post_img = torch.randn(batch_size, 3, 1024, 1024)
    
    masks, metadata = model(pre_img, post_img)
    print(f"\n✓ Output mask shape: {masks.shape}")
    print(f"✓ Feature shape: {metadata['feature_shape']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("✓ Patch-based model test completed successfully!")
    print("=" * 60)
