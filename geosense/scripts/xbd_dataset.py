import os
import json
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
from shapely.geometry import shape
from typing import Dict, List, Tuple


class XBDDataset(Dataset):
    """
    Dataset class for XBD (xView2) building damage assessment dataset.
    Handles pre/post disaster image pairs with polygon annotations.
    """
    
    # Damage categories mapping
    DAMAGE_TYPES = {
        'no-damage': 0,
        'minor-damage': 1,
        'major-damage': 2,
        'destroyed': 3,
        'un-classified': 4
    }
    
    def __init__(
        self,
        pre_image_dir: str,
        post_image_dir: str,
        pre_json_dir: str,
        post_json_dir: str,
        image_size: Tuple[int, int] = (1024, 1024),
        transform=None,
        mode: str = 'multiclass'
    ):
        """
        Args:
            pre_image_dir: Directory containing pre-disaster TIFF images
            post_image_dir: Directory containing post-disaster TIFF images
            pre_json_dir: Directory containing pre-disaster JSON annotations
            post_json_dir: Directory containing post-disaster JSON annotations
            image_size: Target size for images (height, width)
            transform: Optional augmentations
            mode: 'binary' for change detection, 'multiclass' for damage levels
        """
        self.pre_image_dir = pre_image_dir
        self.post_image_dir = post_image_dir
        self.pre_json_dir = pre_json_dir
        self.post_json_dir = post_json_dir
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        
        self.image_pairs = self._get_image_pairs()
        
    def _get_image_pairs(self) -> List[str]:
        """Get matching pre/post image pairs."""
        pre_images = set(os.listdir(self.pre_image_dir))
        post_images = set(os.listdir(self.post_image_dir))
        
        pairs = []
        for pre_img in pre_images:
            if pre_img.endswith('.tif') or pre_img.endswith('.tiff'):
                base_name = pre_img.replace('_pre_disaster', '').replace('.tif', '').replace('.tiff', '')
                # Look for corresponding post image
                for post_img in post_images:
                    if base_name in post_img and ('post_disaster' in post_img or 'post' in post_img):
                        pairs.append((pre_img, post_img))
                        break
        
        return pairs
    
    def _load_tiff(self, path: str) -> np.ndarray:
        """Load TIFF image and convert to RGB numpy array."""
        with rasterio.open(path) as src:
            # Read all bands
            img = src.read()
            
            # Handle different channel configurations
            if img.shape[0] == 1:
                # Grayscale
                img = np.repeat(img, 3, axis=0)
            elif img.shape[0] > 3:
                # Take first 3 channels (RGB)
                img = img[:3]
            
            # Transpose to HWC format
            img = np.transpose(img, (1, 2, 0))
            
            # Normalize to 0-255 if needed
            if img.max() > 255:
                img = (img / img.max() * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
                
            return img
    
    def _load_json_annotations(self, json_path: str) -> Dict:
        """Load JSON annotation file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def _polygons_to_mask(
        self, 
        annotations: Dict, 
        img_shape: Tuple[int, int],
        is_post: bool = False
    ) -> np.ndarray:
        """
        Convert polygon annotations to segmentation mask.
        Uses the 'xy' coordinate format from XBD dataset.
        
        Args:
            annotations: JSON annotation dictionary
            img_shape: (height, width) of the image
            is_post: If True, creates multi-class mask with damage levels
        
        Returns:
            Segmentation mask as numpy array
        """
        if self.mode == 'multiclass' and is_post:
            # Multi-class mask: 0=background, 1=no-damage, 2=minor, 3=major, 4=destroyed
            mask = np.zeros(img_shape, dtype=np.uint8)
            
            # Get features from 'xy' coordinates (pixel coordinates)
            features = annotations.get('features', {}).get('xy', [])
            
            for feature in features:
                try:
                    # Parse WKT polygon format
                    wkt_string = feature['wkt']
                    # Extract coordinates from WKT string
                    coords_str = wkt_string.replace('POLYGON ((', '').replace('))', '')
                    coords_pairs = coords_str.split(', ')
                    
                    # Convert to numpy array of (x, y) coordinates
                    coords = []
                    for pair in coords_pairs:
                        x, y = pair.split(' ')
                        coords.append([float(x), float(y)])
                    
                    coords = np.array(coords, dtype=np.int32)
                    
                    # Get damage class from properties
                    damage_type = feature['properties'].get('subtype', 'no-damage')
                    
                    # Map damage type to class ID
                    damage_map = {
                        'no-damage': 1,
                        'minor-damage': 2,
                        'major-damage': 3,
                        'destroyed': 4,
                        'un-classified': 0  # background
                    }
                    class_id = damage_map.get(damage_type, 0)
                    
                    # Fill polygon
                    cv2.fillPoly(mask, [coords], class_id)
                    
                except Exception as e:
                    print(f"Error processing polygon: {e}")
                    continue
        
        elif self.mode == 'binary':
            # Binary mask for change detection
            mask = np.zeros(img_shape, dtype=np.uint8)
            
            features = annotations.get('features', {}).get('xy', [])
            
            for feature in features:
                try:
                    # Parse WKT polygon
                    wkt_string = feature['wkt']
                    coords_str = wkt_string.replace('POLYGON ((', '').replace('))', '')
                    coords_pairs = coords_str.split(', ')
                    
                    coords = []
                    for pair in coords_pairs:
                        x, y = pair.split(' ')
                        coords.append([float(x), float(y)])
                    
                    coords = np.array(coords, dtype=np.int32)
                    
                    if is_post:
                        # Only mark damaged buildings as 1
                        damage_type = feature['properties'].get('subtype', 'no-damage')
                        if damage_type in ['minor-damage', 'major-damage', 'destroyed']:
                            cv2.fillPoly(mask, [coords], 1)
                    else:
                        # All buildings in pre-image
                        cv2.fillPoly(mask, [coords], 1)
                        
                except Exception as e:
                    print(f"Error processing polygon: {e}")
                    continue
        
        else:
            # Default: building presence mask
            mask = np.zeros(img_shape, dtype=np.uint8)
            features = annotations.get('features', {}).get('xy', [])
            
            for feature in features:
                try:
                    wkt_string = feature['wkt']
                    coords_str = wkt_string.replace('POLYGON ((', '').replace('))', '')
                    coords_pairs = coords_str.split(', ')
                    
                    coords = []
                    for pair in coords_pairs:
                        x, y = pair.split(' ')
                        coords.append([float(x), float(y)])
                    
                    coords = np.array(coords, dtype=np.int32)
                    cv2.fillPoly(mask, [coords], 1)
                except Exception as e:
                    continue
        
        return mask
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - pre_image: Pre-disaster image tensor (C, H, W)
                - post_image: Post-disaster image tensor (C, H, W)
                - mask: Change detection mask or damage classification mask
                - pre_mask: Building mask from pre-disaster (optional)
        """
        pre_img_name, post_img_name = self.image_pairs[idx]
        
        # Load images
        pre_img_path = os.path.join(self.pre_image_dir, pre_img_name)
        post_img_path = os.path.join(self.post_image_dir, post_img_name)
        
        pre_img = self._load_tiff(pre_img_path)
        post_img = self._load_tiff(post_img_path)
        
        # Load annotations
        pre_json_name = pre_img_name.replace('.tif', '.json').replace('.tiff', '.json')
        post_json_name = post_img_name.replace('.tif', '.json').replace('.tiff', '.json')
        
        pre_json_path = os.path.join(self.pre_json_dir, pre_json_name)
        post_json_path = os.path.join(self.post_json_dir, post_json_name)
        
        pre_annotations = self._load_json_annotations(pre_json_path)
        post_annotations = self._load_json_annotations(post_json_path)
        
        # Create masks
        pre_mask = self._polygons_to_mask(pre_annotations, pre_img.shape[:2], is_post=False)
        post_mask = self._polygons_to_mask(post_annotations, post_img.shape[:2], is_post=True)
        
        # Resize images and masks
        pre_img = cv2.resize(pre_img, (self.image_size[1], self.image_size[0]))
        post_img = cv2.resize(post_img, (self.image_size[1], self.image_size[0]))
        pre_mask = cv2.resize(pre_mask, (self.image_size[1], self.image_size[0]), 
                             interpolation=cv2.INTER_NEAREST)
        post_mask = cv2.resize(post_mask, (self.image_size[1], self.image_size[0]), 
                              interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentations if specified
        if self.transform:
            augmented = self.transform(
                image=pre_img,
                image1=post_img,
                mask=post_mask,
                mask1=pre_mask
            )
            pre_img = augmented['image']
            post_img = augmented['image1']
            post_mask = augmented['mask']
            pre_mask = augmented['mask1']
        
        # Convert to tensors
        pre_img = torch.from_numpy(pre_img).permute(2, 0, 1).float() / 255.0
        post_img = torch.from_numpy(post_img).permute(2, 0, 1).float() / 255.0
        post_mask = torch.from_numpy(post_mask).long()
        pre_mask = torch.from_numpy(pre_mask).long()
        
        return {
            'pre_image': pre_img,
            'post_image': post_img,
            'mask': post_mask,  # Main target mask
            'pre_mask': pre_mask,
            'image_name': pre_img_name
        }


# Example usage and data augmentation
def get_training_augmentation():
    """Returns augmentation pipeline for training."""
    import albumentations as A
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
        ], p=0.2),
    ], additional_targets={'image1': 'image', 'mask1': 'mask'})


def get_validation_augmentation():
    """Returns augmentation pipeline for validation (none)."""
    import albumentations as A
    return A.Compose([], additional_targets={'image1': 'image', 'mask1': 'mask'})


# Example instantiation
if __name__ == "__main__":
    dataset = XBDDataset(
        pre_image_dir='path/to/pre_images',
        post_image_dir='path/to/post_images',
        pre_json_dir='path/to/pre_labels',
        post_json_dir='path/to/post_labels',
        image_size=(1024, 1024),
        transform=get_training_augmentation(),
        return_change_mask=True  # Binary change detection
    )
    
    # Test loading
    sample = dataset[0]
    print(f"Pre-image shape: {sample['pre_image'].shape}")
    print(f"Post-image shape: {sample['post_image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Unique mask values: {torch.unique(sample['mask'])}")
