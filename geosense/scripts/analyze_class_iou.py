"""
Quick script to check class-wise IoU with automatic path detection.
Works with both Kaggle and local paths.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import os

# Automatic path detection
def get_dataset_paths():
    """Automatically detect if running on Kaggle or local."""
    
    # Try Kaggle paths
    kaggle_base = '/kaggle/input/xbd-organized-zipped/data'
    if os.path.exists(kaggle_base):
        return {
            'val_pre': f'{kaggle_base}/val/pre',
            'val_post': f'{kaggle_base}/val/post',
            'val_pre_labels': f'{kaggle_base}/val/pre_labels',
            'val_post_labels': f'{kaggle_base}/val/post_labels'
        }
    
    # Try local paths
    local_bases = [
        # Add local data paths 
        'data',
        '../data'
    ]
    
    for base in local_bases:
        if os.path.exists(os.path.join(base, 'val', 'pre')):
            return {
                'val_pre': f'{base}/val/pre',
                'val_post': f'{base}/val/post',
                'val_pre_labels': f'{base}/val/pre_labels',
                'val_post_labels': f'{base}/val/post_labels'
            }
    
    # Manual input
    print("Could not auto-detect dataset paths.")
    base = input("Enter your dataset base path: ")
    return {
        'val_pre': f'{base}/val/pre',
        'val_post': f'{base}/val/post',
        'val_pre_labels': f'{base}/val/pre_labels',
        'val_post_labels': f'{base}/val/post_labels'
    }


def quick_iou_check(checkpoint_path):
    """Quick IoU check with minimal dependencies."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get paths
    print("\nDetecting dataset paths...")
    paths = get_dataset_paths()
    print(f"✓ Using validation data from: {paths['val_pre']}")
    
    # Load checkpoint and detect model type
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    
    # Detect model type
    if 'segformer_head.seg_head.0.weight' in state_dict_keys:
        print("✓ Detected: SegFormer model")
        from geosense_segformer import create_geosense_segformer
        model = create_geosense_segformer(
            dinov2_size='base',
            image_size=(1024, 1024),
            patch_overlap=32,
            freeze_encoder=False
        )
    elif 'decoder.decoder.0.0.weight' in state_dict_keys:
        print("✓ Detected: Custom decoder model")
        from multiclass_model import create_multiclass_geosense_model
        model = create_multiclass_geosense_model(
            dinov2_size='base',
            image_size=(1024, 1024),
            patch_overlap=32,
            freeze_encoder=False
        )
    else:
        print("ERROR: Unknown model type!")
        print("First few keys:", state_dict_keys[:5])
        return
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'Unknown')
    print(f"✓ Loaded checkpoint from epoch {epoch}")
    
    # Load dataset
    print("\nLoading validation dataset...")
    from xbd_dataset import XBDDataset, get_validation_augmentation
    
    val_dataset = XBDDataset(
        pre_image_dir='',
        post_image_dir='',
        pre_json_dir='',
        post_json_dir='',
        image_size=(1024, 1024),
        transform=get_validation_augmentation(),
        mode='multiclass'
    )
    
    print(f"✓ Loaded {len(val_dataset)} validation samples")
    
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Calculate metrics
    print("\nCalculating class-wise IoU...")
    
    class_names = {
        0: "Background",
        1: "No Damage",
        2: "Minor Damage",
        3: "Major Damage",
        4: "Destroyed"
    }
    
    all_ious = {c: [] for c in range(5)}
    all_recalls = {c: [] for c in range(5)}
    all_precisions = {c: [] for c in range(5)}
    class_pixel_counts = {c: 0 for c in range(5)}
    
    num_batches = min(20, len(val_loader))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_batches:
                break
            
            if batch_idx % 5 == 0:
                print(f"  Processing batch {batch_idx}/{num_batches}...")
            
            pre_img = batch['pre_image'].to(device)
            post_img = batch['post_image'].to(device)
            targets = batch['mask'].to(device)
            
            # Get predictions
            logits, _ = model(pre_img, post_img)
            preds = torch.argmax(logits, dim=1)
            
            # Calculate per-class IoU
            for c in range(5):
                pred_c = (preds == c)
                target_c = (targets == c)
                
                intersection = (pred_c & target_c).float().sum()
                union = (pred_c | target_c).float().sum()
                
                iou = (intersection / (union + 1e-6)).item()
                all_ious[c].append(iou)
                
                # Recall
                tp = (pred_c & target_c).float().sum()
                fn = (~pred_c & target_c).float().sum()
                recall = (tp / (tp + fn + 1e-6)).item()
                all_recalls[c].append(recall)
                
                # Precision
                fp = (pred_c & ~target_c).float().sum()
                precision = (tp / (tp + fp + 1e-6)).item()
                all_precisions[c].append(precision)
                
                # Count pixels
                class_pixel_counts[c] += (targets == c).sum().item()
    
    # Print results
    print("\n" + "=" * 80)
    print("CLASS-WISE METRICS")
    print("=" * 80)
    print(f"\n{'Class':<18} {'IoU':>8} {'Recall':>8} {'Precision':>10} {'Pixels':>12}")
    print("-" * 80)
    
    for c in range(5):
        avg_iou = np.mean(all_ious[c])
        avg_recall = np.mean(all_recalls[c])
        avg_precision = np.mean(all_precisions[c])
        pixels = class_pixel_counts[c]
        pixel_pct = (pixels / sum(class_pixel_counts.values())) * 100
        
        print(f"{class_names[c]:<18} {avg_iou:>8.4f} {avg_recall:>8.4f} {avg_precision:>10.4f} {pixels:>8,} ({pixel_pct:>5.1f}%)")
    
    mean_iou = np.mean([np.mean(all_ious[c]) for c in range(5)])
    print("-" * 80)
    print(f"{'MEAN IoU':>18} {mean_iou:>8.4f}")
    print("=" * 80)
    
    # Quick diagnosis
    print("\n" + "=" * 80)
    print("QUICK DIAGNOSIS")
    print("=" * 80)
    
    bg_iou = np.mean(all_ious[0])
    damage_ious = [np.mean(all_ious[c]) for c in range(1, 5)]
    avg_damage_iou = np.mean(damage_ious)
    
    print(f"\nBackground IoU: {bg_iou:.4f}")
    print(f"Avg Damage IoU: {avg_damage_iou:.4f} (classes 1-4)")
    
    if bg_iou > 0.8 and avg_damage_iou < 0.2:
        print("\n⚠️  MODEL IS BIASED TOWARD BACKGROUND!")
        print("   - Predicting mostly background")
        print("   - Not learning damage classes well")
        print("\n   ACTIONS:")
        print("   1. Increase focal loss gamma to 3.0")
        print("   2. Verify class weights are correct")
        print("   3. Check if loss is actually decreasing")
    
    elif mean_iou < 0.15:
        print("\n⚠️  VERY LOW OVERALL IoU")
        print("   Model is not learning well")
        print("\n   ACTIONS:")
        print("   1. Check training loss - is it decreasing?")
        print("   2. Verify loss function is working")
        print("   3. Try lower learning rate (2e-5)")
    
    elif mean_iou < 0.30:
        print("\n~ MODERATE IoU - ROOM FOR IMPROVEMENT")
        print("   Model is learning but needs more training")
        print("\n   ACTIONS:")
        print("   1. Train for more epochs")
        print("   2. Consider adjusting loss weights")
    
    else:
        print("\n✓ DECENT IoU")
        print("   Model is learning well!")
    
    return mean_iou


if __name__ == "__main__":
    # Get checkpoint path
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Try common paths
        possible_paths = [
            # Add checkpoints paths to try
            'checkpoints_multiclass/best_model_multiclass.pth',
            'chkps/best_model_multiclass.pth'
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            checkpoint_path = input("Enter checkpoint path: ")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    quick_iou_check(checkpoint_path)