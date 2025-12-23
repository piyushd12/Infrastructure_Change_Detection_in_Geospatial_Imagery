import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import math


class PatchProcessor:
    """
    Handles splitting large images into patches and reconstructing them.
    This ensures no information loss when processing with DINOv2.
    """
    
    def __init__(
        self,
        patch_size: int = 518,  # DINOv2 large/giant input size
        overlap: int = 64,      # Overlap between patches for smooth merging
        original_size: Tuple[int, int] = (1024, 1024)
    ):
        """
        Args:
            patch_size: Size of each patch (518 for DINOv2 large/giant)
            overlap: Overlap between adjacent patches in pixels
            original_size: Original image size (H, W)
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.original_size = original_size
        
        # Calculate number of patches needed
        self.num_patches_h = math.ceil((original_size[0] - overlap) / self.stride)
        self.num_patches_w = math.ceil((original_size[1] - overlap) / self.stride)
        
        # Calculate actual padded size
        self.padded_h = self.stride * (self.num_patches_h - 1) + patch_size
        self.padded_w = self.stride * (self.num_patches_w - 1) + patch_size
        
        print(f"Patch Configuration:")
        print(f"  Original size: {original_size}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Overlap: {overlap}px")
        print(f"  Stride: {self.stride}px")
        print(f"  Number of patches: {self.num_patches_h}x{self.num_patches_w} = {self.num_patches_h * self.num_patches_w}")
        print(f"  Padded size: {self.padded_h}x{self.padded_w}")
    
    def extract_patches(self, image: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Extract overlapping patches from image.
        
        Args:
            image: Input image tensor (B, C, H, W)
        
        Returns:
            patches: Tensor of patches (B * num_patches, C, patch_size, patch_size)
            metadata: Dictionary containing reconstruction info
        """
        B, C, H, W = image.shape
        
        # Pad image if necessary
        pad_h = self.padded_h - H
        pad_w = self.padded_w - W
        
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
        
        patches = []
        patch_positions = []
        
        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                # Calculate patch position
                start_h = i * self.stride
                start_w = j * self.stride
                end_h = start_h + self.patch_size
                end_w = start_w + self.patch_size
                
                # Extract patch
                patch = image[:, :, start_h:end_h, start_w:end_w]
                patches.append(patch)
                patch_positions.append((start_h, start_w, end_h, end_w))
        
        # Stack patches: (B * num_patches, C, patch_size, patch_size)
        patches = torch.cat(patches, dim=0)
        
        metadata = {
            'patch_positions': patch_positions,
            'original_size': (H, W),
            'padded_size': (self.padded_h, self.padded_w),
            'num_patches_h': self.num_patches_h,
            'num_patches_w': self.num_patches_w,
            'batch_size': B
        }
        
        return patches, metadata
    
    def reconstruct_from_patches(
        self,
        patch_features: torch.Tensor,
        metadata: dict,
        use_averaging: bool = True
    ) -> torch.Tensor:
        """
        Reconstruct full feature map from patch features.
        
        Args:
            patch_features: Features from patches (B * num_patches, C, H_feat, W_feat)
            metadata: Metadata from extract_patches
            use_averaging: If True, average overlapping regions
        
        Returns:
            Reconstructed features (B, C, H_out, W_out)
        """
        batch_size = metadata['batch_size']
        num_patches_h = metadata['num_patches_h']
        num_patches_w = metadata['num_patches_w']
        patch_positions = metadata['patch_positions']
        
        # Get feature dimensions
        _, C, H_feat, W_feat = patch_features.shape
        
        # Calculate output size based on feature map dimensions
        # DINOv2 typically reduces spatial dims by patch_size (14 for large)
        scale_factor = H_feat / self.patch_size
        
        output_h = int(metadata['padded_size'][0] * scale_factor)
        output_w = int(metadata['padded_size'][1] * scale_factor)
        
        # Initialize output tensor and count tensor for averaging
        output = torch.zeros(batch_size, C, output_h, output_w, 
                           device=patch_features.device, dtype=patch_features.dtype)
        
        if use_averaging:
            count = torch.zeros(batch_size, 1, output_h, output_w, 
                              device=patch_features.device, dtype=torch.float32)
        
        # Calculate scaled positions
        scaled_stride = int(self.stride * scale_factor)
        scaled_patch_size = int(self.patch_size * scale_factor)
        
        # Reconstruct by placing patches
        patch_idx = 0
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * scaled_stride
                start_w = j * scaled_stride
                end_h = start_h + scaled_patch_size
                end_w = start_w + scaled_patch_size
                
                # Ensure we don't exceed bounds
                end_h = min(end_h, output_h)
                end_w = min(end_w, output_w)
                
                # Get features for each batch item
                for b in range(batch_size):
                    feat_idx = b * (num_patches_h * num_patches_w) + patch_idx
                    
                    # Handle size mismatch at boundaries
                    feat_h = end_h - start_h
                    feat_w = end_w - start_w
                    
                    if use_averaging:
                        output[b, :, start_h:end_h, start_w:end_w] += \
                            patch_features[feat_idx, :, :feat_h, :feat_w]
                        count[b, :, start_h:end_h, start_w:end_w] += 1
                    else:
                        # Simple overwrite (last patch wins)
                        output[b, :, start_h:end_h, start_w:end_w] = \
                            patch_features[feat_idx, :, :feat_h, :feat_w]
                
                patch_idx += 1
        
        # Average overlapping regions
        if use_averaging:
            output = output / (count + 1e-6)
        
        # Crop to original size (scaled)
        original_h = int(metadata['original_size'][0] * scale_factor)
        original_w = int(metadata['original_size'][1] * scale_factor)
        output = output[:, :, :original_h, :original_w]
        
        return output
    
    def create_weight_map(self) -> torch.Tensor:
        """
        Create a weight map for smooth blending in overlapping regions.
        Uses cosine weighting to reduce edge artifacts.
        
        Returns:
            Weight map (1, 1, patch_size, patch_size)
        """
        # Create 1D weight profiles
        weight_1d = torch.ones(self.patch_size)
        
        if self.overlap > 0:
            fade_length = self.overlap
            # Cosine fade
            fade = torch.cos(torch.linspace(0, math.pi/2, fade_length)) ** 2
            
            # Apply fade at edges
            weight_1d[:fade_length] *= fade
            weight_1d[-fade_length:] *= fade.flip(0)
        
        # Create 2D weight map
        weight_map = weight_1d.unsqueeze(1) * weight_1d.unsqueeze(0)
        
        return weight_map.unsqueeze(0).unsqueeze(0)


def get_dinov2_input_size(model_name: str) -> int:
    """
    Get the expected input size for different DINOv2 models.
    
    Args:
        model_name: DINOv2 model name
    
    Returns:
        Input size in pixels
    """
    if 'small' in model_name or 'base' in model_name:
        return 322  # Base/Small models use 322x322
    else:
        return 518  # Large/Giant models use 518x518


# Example usage and testing
if __name__ == "__main__":
    # Test patch processor
    print("Testing PatchProcessor...")
    
    # Initialize processor
    processor = PatchProcessor(
        patch_size=518,
        overlap=64,
        original_size=(1024, 1024)
    )
    
    # Create dummy image
    dummy_image = torch.randn(2, 3, 1024, 1024)
    
    # Extract patches
    patches, metadata = processor.extract_patches(dummy_image)
    print(f"\nExtracted patches shape: {patches.shape}")
    
    # Simulate feature extraction (DINOv2 reduces spatial dims)
    # For DINOv2, output is typically (batch, num_patches, hidden_dim)
    # We'll simulate this with spatial features
    patch_size_out = 518 // 14  # Patch size 14 for DINOv2
    dummy_features = torch.randn(patches.shape[0], 256, patch_size_out, patch_size_out)
    print(f"Simulated features shape: {dummy_features.shape}")
    
    # Reconstruct
    reconstructed = processor.reconstruct_from_patches(dummy_features, metadata)
    print(f"Reconstructed features shape: {reconstructed.shape}")
    
    # Verify dimensions
    expected_h = int(1024 * (patch_size_out / 518))
    expected_w = int(1024 * (patch_size_out / 518))
    print(f"\nExpected output size: {expected_h}x{expected_w}")
    print(f"Actual output size: {reconstructed.shape[2]}x{reconstructed.shape[3]}")
    
    print("\n✓ Patch processing test completed!")
