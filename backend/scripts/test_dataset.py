import json
import numpy as np
import matplotlib.pyplot as plt
import cv2


def test_json_parsing():
    """Test parsing the XBD JSON format."""
    
    # Sample post-disaster JSON (with damage labels)
    sample_post_json = {
        "features": {
            "xy": [
                {
                    "properties": {
                        "feature_type": "building",
                        "subtype": "no-damage",
                        "uid": "217fb2f0-9671-4cb8-b70f-9eb11b594e05"
                    },
                    "wkt": "POLYGON ((99.05767498485287 0.002684676241459629, 103.7481098419098 11.93443783903123, 64.22730841957046 0.00184925240727189, 99.05767498485287 0.002684676241459629))"
                },
                {
                    "properties": {
                        "feature_type": "building",
                        "subtype": "destroyed",
                        "uid": "38e4ae71-4b44-48bb-9b46-452ae719c76a"
                    },
                    "wkt": "POLYGON ((871.7151114060493 279.7180641600485, 894.0028121310498 275.0305715454728, 890.7842478832384 336.9169556954659, 871.7151114060493 279.7180641600485))"
                },
                {
                    "properties": {
                        "feature_type": "building",
                        "subtype": "major-damage",
                        "uid": "54ea090f-28d5-460b-8e67-0b827d9447f4"
                    },
                    "wkt": "POLYGON ((375.7194321159964 295.3355384734762, 383.2371089492614 295.3355384713971, 376.2947337695092 330.928530595859, 375.7194321159964 295.3355384734762))"
                },
                {
                    "properties": {
                        "feature_type": "building",
                        "subtype": "minor-damage",
                        "uid": "test-minor"
                    },
                    "wkt": "POLYGON ((500 400, 550 400, 550 450, 500 450, 500 400))"
                }
            ]
        },
        "metadata": {
            "width": 1024,
            "height": 1024
        }
    }
    
    # Create mask
    img_shape = (1024, 1024)
    mask = np.zeros(img_shape, dtype=np.uint8)
    
    # Damage class mapping
    damage_map = {
        'no-damage': 1,
        'minor-damage': 2,
        'major-damage': 3,
        'destroyed': 4
    }
    
    # Color mapping for visualization
    color_map = {
        0: (128, 128, 128),  # Background - Gray
        1: (0, 255, 0),      # No damage - Green
        2: (255, 255, 0),    # Minor - Yellow
        3: (255, 165, 0),    # Major - Orange
        4: (255, 0, 0)       # Destroyed - Red
    }
    
    features = sample_post_json['features']['xy']
    
    for feature in features:
        # Parse WKT polygon
        wkt_string = feature['wkt']
        coords_str = wkt_string.replace('POLYGON ((', '').replace('))', '')
        coords_pairs = coords_str.split(', ')
        
        # Convert to numpy array
        coords = []
        for pair in coords_pairs:
            x, y = pair.split(' ')
            coords.append([float(x), float(y)])
        
        coords = np.array(coords, dtype=np.int32)
        
        # Get damage class
        damage_type = feature['properties'].get('subtype', 'no-damage')
        class_id = damage_map.get(damage_type, 0)
        
        # Fill polygon
        cv2.fillPoly(mask, [coords], class_id)
        
        print(f"✓ Parsed {damage_type}: {len(coords)} vertices, class_id={class_id}")
    
    # Create colored visualization
    colored_mask = np.zeros((*img_shape, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        colored_mask[mask == class_id] = color
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(mask, cmap='tab10', vmin=0, vmax=4)
    axes[0].set_title('Multi-Class Mask\n(Class IDs)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(colored_mask)
    axes[1].set_title('Color-Coded Mask\n🟢=No 🟡=Minor 🟠=Major 🔴=Destroyed', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_mask_parsing.png', dpi=150, bbox_inches='tight')
    print("\n✓ Test visualization saved to 'test_mask_parsing.png'")
    plt.show()
    
    # Print statistics
    unique, counts = np.unique(mask, return_counts=True)
    print("\n" + "=" * 60)
    print("Mask Statistics:")
    print("=" * 60)
    damage_names = {0: 'Background', 1: 'No Damage', 2: 'Minor', 3: 'Major', 4: 'Destroyed'}
    for class_id, count in zip(unique, counts):
        percentage = (count / mask.size) * 100
        print(f"{damage_names.get(class_id, 'Unknown'):15s}: {count:8,} pixels ({percentage:5.2f}%)")
    
    return mask


def test_full_dataset_loading():
    """Test loading actual dataset (if available)."""
    print("\n" + "=" * 60)
    print("Testing Full Dataset Loading")
    print("=" * 60)
    
    try:
        from xbd_dataset import XBDDataset, get_training_augmentation
        
        # Try to load dataset
        dataset = XBDDataset(
            pre_image_dir='geosense/data/train/pre',
            post_image_dir='geosense/data/train/post',
            pre_json_dir='geosense/data/train/pre_labels',
            post_json_dir='geosense/data/train/post_labels',
            image_size=(1024, 1024),
            transform=None,
            mode='multiclass'
        )
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  Total samples: {len(dataset)}")
        
        # Load first sample
        if len(dataset) > 0:
            sample = dataset[501]
            print(f"\n✓ First sample loaded:")
            print(f"  Pre-image shape: {sample['pre_image'].shape}")
            print(f"  Post-image shape: {sample['post_image'].shape}")
            print(f"  Mask shape: {sample['mask'].shape}")
            print(f"  Unique mask values: {np.unique(sample['mask'].numpy())}")
            
            # Visualize
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(sample['pre_image'].permute(1, 2, 0))
            axes[0].set_title('Pre-Disaster', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(sample['post_image'].permute(1, 2, 0))
            axes[1].set_title('Post-Disaster', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Color-coded mask
            mask_np = sample['mask'].numpy()
            colored = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
            colored[mask_np == 0] = (128, 128, 128)  # Gray
            colored[mask_np == 1] = (0, 255, 0)      # Green
            colored[mask_np == 2] = (255, 255, 0)    # Yellow
            colored[mask_np == 3] = (255, 165, 0)    # Orange
            colored[mask_np == 4] = (255, 0, 0)      # Red
            
            axes[2].imshow(colored)
            axes[2].set_title('Damage Mask (Color-Coded)', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig('test_dataset_sample.png', dpi=150, bbox_inches='tight')
            print("\n✓ Sample visualization saved to 'test_dataset_sample.png'")
            plt.show()
            
    except Exception as e:
        print(f"✗ Could not load dataset: {e}")
        print("  Make sure data is organized in: data/train/pre, data/train/post, etc.")


def main():
    """Run all tests."""
    print("=" * 60)
    print("XBD Dataset Format Test")
    print("=" * 60)
    
    # Test 1: JSON parsing
    print("\n1. Testing JSON Parsing...")
    mask = test_json_parsing()
    
    # Test 2: Full dataset loading
    print("\n2. Testing Full Dataset Loading...")
    test_full_dataset_loading()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\nIf you see colored masks, the dataset loader is working correctly!")
    print("Now you can train with: python multiclass_training.py")


if __name__ == '__main__':
    main()
