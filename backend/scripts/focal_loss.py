"""
Combined Cross-Entropy + Focal Loss for multi-class damage assessment.

Why this works:
- Cross-Entropy: Standard loss, handles overall classification
- Focal Loss: Focuses on hard examples and rare classes
- Together: You get robustness + focus on difficult cases

This is better than using them separately.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import WeightedRandomSampler



class ImprovedCombinedLoss(nn.Module):
    """
    Combines Cross-Entropy + Focal Loss + Dice Loss for robust training.
    
    The three components work together:
    1. Cross-Entropy: Overall classification accuracy
    2. Focal Loss: Focus on hard/misclassified examples
    3. Dice Loss: Improve segmentation quality
    """
    
    def __init__(
        self,
        ce_weight: float = 0.5,
        focal_weight: float = 1.0,
        dice_weight: float = 2.0,
        class_weights: list = None,
        gamma: float = 2.0,
        ignore_index: int = -100
    ):
        """
        Args:
            ce_weight: Weight for cross-entropy component
            focal_weight: Weight for focal loss component
            dice_weight: Weight for dice loss component
            class_weights: Class weights (inverse frequency)
            gamma: Focal loss focusing parameter
            ignore_index: Index to ignore in loss
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        
        # Standard cross-entropy with class weights
        if class_weights is not None:
            self.register_buffer('class_weights', 
                               torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
    
    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal Loss component."""
        # Get softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get cross entropy
        ce = F.cross_entropy(
            inputs, targets, 
            weight=self.class_weights,
            reduction='none', 
            ignore_index=self.ignore_index
        )
        
        # Get probability of true class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal loss: FL = -(1 - p_t)^gamma * log(p_t)
        focal_loss = ((1 - p_t) ** self.gamma) * ce
        
        return focal_loss.mean()
    
    def cross_entropy_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Standard Cross-Entropy Loss component."""
        return F.cross_entropy(
            inputs, targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index
        )
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice Loss component for segmentation."""
        num_classes = pred.shape[1]
        smooth = 1.0
        
        # Softmax predictions
        pred_soft = F.softmax(pred, dim=1)
        
        # One-hot encode targets
        target_one_hot = F.one_hot(target.long(), num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate Dice for each class
        dice_loss = 0.0
        for c in range(num_classes):
            if c == self.ignore_index:
                continue
            
            pred_c = pred_soft[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice_c = (2.0 * intersection + smooth) / (union + smooth)
            dice_loss += (1.0 - dice_c)
        
        return dice_loss / (num_classes - 1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Combined loss computation.
        
        Args:
            pred: Logits (B, C, H, W)
            target: Ground truth (B, H, W)
        
        Returns:
            Total loss
        """

        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(pred.device)

        ce = self.cross_entropy_loss(pred, target)
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        total_loss = (
            self.ce_weight * ce + 
            self.focal_weight * focal + 
            self.dice_weight * dice
        )
        
        return total_loss


class LossWithDebug(nn.Module):
    """
    Same as ImprovedCombinedLoss but with per-component loss tracking.
    Useful for debugging and understanding which component is working.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.base_loss = ImprovedCombinedLoss(*args, **kwargs)
        self.ce_weight = self.base_loss.ce_weight
        self.focal_weight = self.base_loss.focal_weight
        self.dice_weight = self.base_loss.dice_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns total loss and stores components for logging."""
        ce = self.base_loss.cross_entropy_loss(pred, target)
        focal = self.base_loss.focal_loss(pred, target)
        dice = self.base_loss.dice_loss(pred, target)
        
        # Store for logging
        self.last_ce = ce.item()
        self.last_focal = focal.item()
        self.last_dice = dice.item()
        
        total_loss = (
            self.ce_weight * ce + 
            self.focal_weight * focal + 
            self.dice_weight * dice
        )
        
        return total_loss
    
    def get_components(self):
        """Get individual loss components."""
        return {
            'ce': self.last_ce,
            'focal': self.last_focal,
            'dice': self.last_dice
        }

class WeightedSampler:
    """
    Utility class for creating weighted samplers.
    This oversamples rare classes during training.
    """
    
    @staticmethod
    def get_sampler_from_dataset(dataset, num_classes=5):
        """
        Create a weighted sampler that oversamples rare classes.
        
        Args:
            dataset: XBDDataset instance
            num_classes: Number of damage classes (5 by default)
        
        Returns:
            WeightedRandomSampler
        """
        print("\nCalculating sample weights...")
        
        # Count classes in dataset
        class_counts = [0] * num_classes
        
        # Sample the dataset to count classes
        sample_size = min(len(dataset), 200)  # Sample up to 200 images
        
        for i in range(sample_size):
            try:
                mask = dataset[i]['mask'].numpy()
                unique, counts = np.unique(mask, return_counts=True)
                
                for cls, cnt in zip(unique, counts):
                    class_counts[int(cls)] += int(cnt)
            except Exception as e:
                print(f"  Warning: Could not process sample {i}: {e}")
                continue
        
        # Calculate weights (inverse frequency)
        total = sum(class_counts)
        if total == 0:
            print("  Warning: No pixels counted, using uniform weights")
            class_weights = [1.0] * num_classes
        else:
            class_weights = [total / (cnt + 1e-6) for cnt in class_counts]
        
        print(f"  Class weights: {[f'{w:.4f}' for w in class_weights]}")
        
        # Assign weight to each sample based on its class distribution
        sample_weights = []
        
        for i in range(len(dataset)):
            try:
                mask = dataset[i]['mask'].numpy()
                unique, counts = np.unique(mask, return_counts=True)
                
                # Weight sample by rarest class it contains
                rare_class_weight = 0
                for cls, cnt in zip(unique, counts):
                    rare_class_weight = max(rare_class_weight, class_weights[int(cls)])
                
                sample_weights.append(rare_class_weight)
            except Exception as e:
                # Default weight if error
                sample_weights.append(1.0)
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        
        print(f"  Weighted sampler created for {len(dataset)} samples")
        
        return sampler

# ============================================================================
# USAGE IN multiclass_training.py
# ============================================================================

"""
Replace this in multiclass_training.py:

    criterion = CombinedLoss(
        focal_weight=1.0,
        dice_weight=2.0,
        class_weights=class_weights,
        gamma=2.0
    )

With this:

    criterion = ImprovedCombinedLoss(
        ce_weight=0.5,        # Cross-entropy weight
        focal_weight=1.0,     # Focal loss weight
        dice_weight=2.0,      # Dice loss weight
        class_weights=class_weights,
        gamma=2.5            # Focal gamma
    )

Or with debug version to track components:

    criterion = LossWithDebug(
        ce_weight=0.5,
        focal_weight=1.0,
        dice_weight=2.0,
        class_weights=class_weights,
        gamma=2.5
    )
    
    # Then in training loop, after computing loss:
    loss.backward()
    
    # Log individual components:
    components = criterion.get_components()
    print(f"CE: {components['ce']:.4f}, Focal: {components['focal']:.4f}, Dice: {components['dice']:.4f}")
"""


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Improved Combined Loss")
    print("=" * 70)
    
    batch_size, height, width, num_classes = 2, 256, 256, 5
    
    # Create dummy data
    logits = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test 1: Basic loss
    print("\n1. Testing ImprovedCombinedLoss:")
    criterion = ImprovedCombinedLoss(
        ce_weight=0.5,
        focal_weight=1.0,
        dice_weight=2.0,
        class_weights=[1.0, 22.0, 173.0, 137.0, 291.0],
        gamma=2.5
    )
    
    loss = criterion(logits, targets)
    print(f"   Total loss: {loss.item():.4f}")
    print("   ✓ ImprovedCombinedLoss working!")
    
    # Test 2: With debug tracking
    print("\n2. Testing LossWithDebug:")
    criterion_debug = LossWithDebug(
        ce_weight=0.5,
        focal_weight=1.0,
        dice_weight=2.0,
        class_weights=[1.0, 22.0, 173.0, 137.0, 291.0],
        gamma=2.5
    )
    
    loss = criterion_debug(logits, targets)
    components = criterion_debug.get_components()
    
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   - CE component: {components['ce']:.4f}")
    print(f"   - Focal component: {components['focal']:.4f}")
    print(f"   - Dice component: {components['dice']:.4f}")
    print("   ✓ LossWithDebug working!")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)