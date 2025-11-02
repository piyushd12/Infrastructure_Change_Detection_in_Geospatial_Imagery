"""
Training script for multi-class building damage assessment.
UPDATED with exact class weights from diagnostic output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from tqdm import tqdm
from typing import Dict
import os

# ADD THIS IMPORT
from focal_loss import ImprovedCombinedLoss, WeightedSampler
from aggressive_loss_config import get_aggressive_loss_config, get_extreme_weighted_sampler


class MultiClassMetrics:
    """Calculate metrics for multi-class damage assessment."""
    
    @staticmethod
    def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Overall pixel accuracy."""
        pred_classes = torch.argmax(pred, dim=1)
        correct = (pred_classes == target).float().sum()
        total = target.numel()
        return (correct / total).item()
    
    @staticmethod
    def per_class_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 5) -> Dict:
        """IoU for each class."""
        pred_classes = torch.argmax(pred, dim=1)
        
        ious = {}
        for c in range(num_classes):
            pred_c = (pred_classes == c)
            target_c = (target == c)
            
            intersection = (pred_c & target_c).float().sum()
            union = (pred_c | target_c).float().sum()
            
            iou = (intersection / (union + 1e-6)).item()
            ious[f'class_{c}_iou'] = iou
        
        # Mean IoU
        ious['mean_iou'] = np.mean([v for k, v in ious.items() if 'class' in k])
        
        return ious
    
    @staticmethod
    def per_class_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 5) -> Dict:
        """F1 score for each class."""
        pred_classes = torch.argmax(pred, dim=1)
        
        f1_scores = {}
        for c in range(num_classes):
            pred_c = (pred_classes == c)
            target_c = (target == c)
            
            tp = (pred_c & target_c).float().sum()
            fp = (pred_c & ~target_c).float().sum()
            fn = (~pred_c & target_c).float().sum()
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            
            f1_scores[f'class_{c}_f1'] = f1.item()
        
        # Mean F1
        f1_scores['mean_f1'] = np.mean([v for k, v in f1_scores.items() if 'class' in k])
        
        return f1_scores


class MultiClassTrainer:
    """Trainer for multi-class damage assessment."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints_multiclass',
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if use_wandb:
            try:
                import wandb
                wandb.init(project='geosense-multiclass')
                wandb.watch(model)
            except:
                print("Warning: wandb not available")
        
        self.best_val_iou = 0.0
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_acc = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} - Training')
        
        for batch in pbar:
            pre_img = batch['pre_image'].to(self.device)
            post_img = batch['post_image'].to(self.device)
            target = batch['mask'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits, metadata = self.model(pre_img, post_img)
            
            # Loss (now using Focal Loss!)
            loss = self.criterion(logits, target)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                acc = MultiClassMetrics.accuracy(logits, target)
            
            total_loss += loss.item()
            total_acc += acc
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
        
        metrics = {
            'train_loss': total_loss / len(self.train_loader),
            'train_acc': total_acc / len(self.train_loader)
        }
        
        return metrics
    
    def validate(self, epoch: int) -> Dict:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        all_ious = []
        all_f1s = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} - Validation')
        
        with torch.no_grad():
            for batch in pbar:
                pre_img = batch['pre_image'].to(self.device)
                post_img = batch['post_image'].to(self.device)
                target = batch['mask'].to(self.device)
                
                # Forward
                logits, metadata = self.model(pre_img, post_img)
                
                # Loss
                loss = self.criterion(logits, target)
                
                # Metrics
                acc = MultiClassMetrics.accuracy(logits, target)
                ious = MultiClassMetrics.per_class_iou(logits, target)
                f1s = MultiClassMetrics.per_class_f1(logits, target)
                
                total_loss += loss.item()
                total_acc += acc
                all_ious.append(ious['mean_iou'])
                all_f1s.append(f1s['mean_f1'])
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{ious["mean_iou"]:.4f}'})
        
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_acc': total_acc / len(self.val_loader),
            'val_mean_iou': np.mean(all_ious),
            'val_mean_f1': np.mean(all_f1s)
        }
        
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics)
            except:
                pass
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model_multiclass.pth')
            torch.save(checkpoint, best_path)
            print(f'✓ Saved best model with IoU: {metrics["val_mean_iou"]:.4f}')
    
    def train(self, num_epochs: int, save_every: int = 5):
        """Main training loop."""
        print("=" * 70)
        print("Multi-Class Damage Assessment Training")
        print("=" * 70)
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 70)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Print summary
            print(f'\nEpoch {epoch} Summary:')
            print(f'Train Loss: {train_metrics["train_loss"]:.4f} | '
                  f'Train Acc: {train_metrics["train_acc"]:.4f}')
            print(f'Val Loss: {val_metrics["val_loss"]:.4f} | '
                  f'Val Acc: {val_metrics["val_acc"]:.4f} | '
                  f'Val IoU: {val_metrics["val_mean_iou"]:.4f} | '
                  f'Val F1: {val_metrics["val_mean_f1"]:.4f}')
            
            # Check if best
            is_best = val_metrics['val_mean_iou'] > self.best_val_iou
            if is_best:
                self.best_val_iou = val_metrics['val_mean_iou']
            
            # Save
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)
        
        print("\n" + "=" * 70)
        print(f'Training completed! Best IoU: {self.best_val_iou:.4f}')
        print("=" * 70)


def main():
    """Main training function for multi-class model."""
    
    config = {
        'batch_size': 4,
        'num_epochs': 40,
        'learning_rate': 2e-5,  # Lower for stability
        'weight_decay': 1e-4,
        'image_size': 1024,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_wandb': True,  # Set to True if you have wandb
    }
    
    print(f"Using device: {config['device']}")
    
    # Import modules
    from xbd_dataset import XBDDataset, get_training_augmentation, get_validation_augmentation
    # from multiclass_model import create_multiclass_geosense_model
    from geosense_segformer import create_geosense_segformer
    
    # Create datasets with multi-class mode
    train_dataset = XBDDataset(
        pre_image_dir='/kaggle/input/xbd-organized-zipped/data/train/pre',
        post_image_dir='/kaggle/input/xbd-organized-zipped/data/train/post',
        pre_json_dir='/kaggle/input/xbd-organized-zipped/data/train/pre_labels',
        post_json_dir='/kaggle/input/xbd-organized-zipped/data/train/post_labels',
        image_size=(config['image_size'], config['image_size']),
        transform=get_training_augmentation(),
        mode='multiclass'
    )
    
    val_dataset = XBDDataset(
        pre_image_dir='/kaggle/input/xbd-organized-zipped/data/val/pre',
        post_image_dir='/kaggle/input/xbd-organized-zipped/data/val/post',
        pre_json_dir='/kaggle/input/xbd-organized-zipped/data/val/pre_labels',
        post_json_dir='/kaggle/input/xbd-organized-zipped/data/val/post_labels',
        image_size=(config['image_size'], config['image_size']),
        transform=get_validation_augmentation(),
        mode='multiclass'
    )
    
    # ===== CREATE WEIGHTED SAMPLER =====
    print("\n" + "=" * 70)
    print("Setting up weighted sampler for rare class oversampling...")
    print("=" * 70)
    # sampler = WeightedSampler.get_sampler_from_dataset(train_dataset)
    # EXTREME weighted sampler - heavily oversample rare damage
    sampler = get_extreme_weighted_sampler(train_dataset)
    
    # Dataloaders with weighted sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # ===== CREATE MULTI-CLASS MODEL =====
    # model = create_multiclass_geosense_model(
    #     dinov2_size='base',
    #     image_size=(config['image_size'], config['image_size']),
    #     patch_overlap=32,
    #     fusion_type='concat_diff',
    #     freeze_encoder=True  # CHANGED to False for better learning
    # )

    # ===== CREATE MULTI-CLASS MODEL WITH SEGFORMER HEAD =====
    model = create_geosense_segformer(
        dinov2_size='base',
        image_size=(config['image_size'], config['image_size']),
        patch_overlap=32,
        freeze_encoder=True
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel Parameters:')
    print(f'Total: {total_params:,}')
    print(f'Trainable: {trainable_params:,}')
    
    # ===== USE FOCAL LOSS WITH EXACT CLASS WEIGHTS FROM DIAGNOSTIC =====
    print("\n" + "=" * 70)
    print("Setting up Focal Loss with diagnostic-derived class weights")
    print("=" * 70)
    
    # EXACT WEIGHTS from your diagnostic output
    # These are better than [0.0080, 0.1727, ...]
    class_weights = [1.0, 22.0, 173.0, 137.0, 291.0]
    max_weight = max(class_weights)
    class_weights = [w / max_weight for w in class_weights]
    
    print(f"Class weights: {[f'{w:.4f}' for w in class_weights]}")
    print("\nExplanation:")
    print(f"  Class 0 (Background   ): {class_weights[0]:.4f} - downweighted (94% of pixels)")
    print(f"  Class 1 (No Damage    ): {class_weights[1]:.4f}")
    print(f"  Class 2 (Minor Damage ): {class_weights[2]:.4f} - heavily weighted (rare)")
    print(f"  Class 3 (Major Damage ): {class_weights[3]:.4f}")
    print(f"  Class 4 (Destroyed    ): {class_weights[4]:.4f} - heavily weighted (rarest)")
    
    # criterion = ImprovedCombinedLoss(
    #     ce_weight=0.5,
    #     focal_weight=1.0,
    #     dice_weight=2.0,
    #     class_weights=class_weights,
    #     gamma=2.5
    # )
    
    # AGGRESSIVE LOSS - forces model to learn damage classes
    criterion = get_aggressive_loss_config().to(config['device'])

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Trainer
    trainer = MultiClassTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config['device'],
        checkpoint_dir='checkpoints_multiclass',
        use_wandb=config['use_wandb']
    )
    
    # Train!
    trainer.train(num_epochs=config['num_epochs'], save_every=2)


if __name__ == '__main__':
    main()