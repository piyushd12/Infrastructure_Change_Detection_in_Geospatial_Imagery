"""
AGGRESSIVE loss configuration specifically for your case.
This will FORCE the model to learn damage classes.

Problem: Background IoU 0.94, Damage IoU 0.003
Solution: Much more aggressive focal loss + stronger class weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtraAggressiveLoss(nn.Module):
    """
    Extra aggressive loss for severe class imbalance.
    Specifically designed for when model predicts 99% background.
    """
    
    def __init__(
        self,
        ce_weight: float = 0.3,
        focal_weight: float = 2.0,
        dice_weight: float = 3.0,
        class_weights: list = None,
        gamma: float = 4.0,  # VERY aggressive
        ignore_index: int = -100
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        
        if class_weights is not None:
            self.register_buffer('class_weights', 
                               torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
    
    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss component."""
        p = F.softmax(inputs, dim=1)
        ce = F.cross_entropy(
            inputs, targets, 
            weight=self.class_weights,
            reduction='none', 
            ignore_index=self.ignore_index
        )
        
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = ((1 - p_t) ** self.gamma) * ce
        
        return focal_loss.mean()
    
    def cross_entropy_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss."""
        return F.cross_entropy(
            inputs, targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index
        )
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice loss for segmentation quality."""
        num_classes = pred.shape[1]
        smooth = 1.0
        
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
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
        """Combined loss."""
        
        # 🔧 Ensure class weights are on the same device as the model output
        if self.class_weights is not None and self.class_weights.device != pred.device:
            self.class_weights = self.class_weights.to(pred.device)
        ce = self.cross_entropy_loss(pred, target)
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        total = self.ce_weight * ce + self.focal_weight * focal + self.dice_weight * dice
        
        return total


# ============================================================================
# EXACT CONFIGURATION TO USE
# ============================================================================

def get_aggressive_loss_config():
    """
    Returns the exact loss configuration you need.
    Based on your results: Background 0.94, Damage 0.003
    """
    
    # MUCH stronger class weights
    # Background gets HEAVILY downweighted
    class_weights = [
        0.001,    # Background - VERY LOW (we don't care about this class anymore)
        50.0,     # No Damage - high priority
        200.0,    # Minor Damage - very high priority
        150.0,    # Major Damage - very high priority  
        250.0     # Destroyed - HIGHEST priority (rarest)
    ]
    
    # Normalize
    max_weight = max(class_weights)
    class_weights = [w / max_weight for w in class_weights]
    
    print("Aggressive Class Weights:")
    class_names = ["Background", "No Damage", "Minor", "Major", "Destroyed"]
    for i, (name, w) in enumerate(zip(class_names, class_weights)):
        print(f"  Class {i} ({name:12s}): {w:.6f}")
    
    criterion = ExtraAggressiveLoss(
        ce_weight=0.3,        # Reduce CE (less focus on easy examples)
        focal_weight=2.0,     # INCREASE focal (focus on hard examples)
        dice_weight=3.0,      # INCREASE dice (better segmentation)
        class_weights=class_weights,
        gamma=4.0            # VERY aggressive gamma (was 2.5, now 4.0)
    )
    
    return criterion


# ============================================================================
# ADDITIONAL FIX: WEIGHTED SAMPLING
# ============================================================================

def get_extreme_weighted_sampler(dataset, num_classes=5):
    """
    Create an EXTREMELY weighted sampler.
    Oversample rare classes much more aggressively.
    """
    from torch.utils.data import WeightedRandomSampler
    import numpy as np
    
    print("\nCalculating extreme sample weights...")
    
    # Count classes
    class_counts = [0] * num_classes
    sample_size = min(len(dataset), 100)
    
    for i in range(sample_size):
        try:
            mask = dataset[i]['mask'].numpy()
            unique, counts = np.unique(mask, return_counts=True)
            for cls, cnt in zip(unique, counts):
                class_counts[int(cls)] += int(cnt)
        except:
            continue
    
    print("Class distribution in sampled data:")
    for i, cnt in enumerate(class_counts):
        print(f"  Class {i}: {cnt:,} pixels")
    
    # EXTREME inverse frequency weights
    # Make rare classes MUCH more likely to be sampled
    total = sum(class_counts)
    class_weights = []
    
    for i, cnt in enumerate(class_counts):
        if i == 0:  # Background
            # Very low weight - we don't want background-only images
            weight = 0.1
        else:
            # Rare damage classes get HUGE weights
            weight = (total / (cnt + 1e-6)) ** 2  # Squared for extra emphasis
        
        class_weights.append(weight)
    
    print("\nSample weights for oversampling:")
    for i, w in enumerate(class_weights):
        print(f"  Class {i}: {w:.2f}")
    
    # Assign weights to samples
    sample_weights = []
    for i in range(len(dataset)):
        try:
            mask = dataset[i]['mask'].numpy()
            unique = np.unique(mask)
            
            # Sample weight = max weight of any class present
            max_weight = max([class_weights[int(cls)] for cls in unique])
            
            # BOOST if sample contains rare damage classes (2, 3, 4)
            has_rare_damage = any(cls in [2, 3, 4] for cls in unique)
            if has_rare_damage:
                max_weight *= 5.0  # 5x boost for samples with rare damage
            
            sample_weights.append(max_weight)
        except:
            sample_weights.append(1.0)
    
    print(f"\nSample weights range: {min(sample_weights):.2f} to {max(sample_weights):.2f}")
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler

# Test
if __name__ == "__main__":
    print("=" * 70)
    print("AGGRESSIVE LOSS CONFIGURATION")
    print("=" * 70)
    
    criterion = get_aggressive_loss_config()
    
    # Test
    batch_size, height, width, num_classes = 2, 256, 256, 5
    logits = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    loss = criterion(logits, targets)
    print(f"\n✓ Loss computed: {loss.item():.4f}")
    
    print("\n" + "=" * 70)
    print("WHAT THIS WILL DO:")
    print("=" * 70)
    print("1. Background weight: 0.0004 (almost ignored)")
    print("2. Damage weights: 0.2 - 1.0 (heavily prioritized)")
    print("3. Gamma: 4.0 (very aggressive focal loss)")
    print("4. Weighted sampling: 5x oversample rare damage images")
    print("\nExpected results after 5-10 epochs:")
    print("- Background IoU: might drop to 0.85 (acceptable)")
    print("- Damage IoU: should improve to 0.15-0.30")
    print("- Mean IoU: target 0.30-0.40")