import os
import shutil
from pathlib import Path
from tqdm import tqdm
import glob

def organize_xbd_dataset(source_dir, output_dir):
    """
    Organize XBD dataset into the required structure.
    
    Source structure:
      source_dir/train/images/*_pre_disaster.tif
      source_dir/train/images/*_post_disaster.tif
      source_dir/train/labels/*_pre_disaster.json
      source_dir/train/labels/*_post_disaster.json
    
    Output structure:
      output_dir/train/pre/*_pre_disaster.tif
      output_dir/train/post/*_post_disaster.tif
      output_dir/train/pre_labels/*_pre_disaster.json
      output_dir/train/post_labels/*_post_disaster.json
    """
    
    print("=" * 70)
    print("ORGANIZING XBD DATASET")
    print("=" * 70)
    
    splits = {
        'train': 'tier1',
        'val': 'hold', 
        'test': 'test'
    }
    
    for split_out, split_in in splits.items():
        print(f"\n📂 Processing {split_out} split...")
        
        source_images = os.path.join(source_dir, split_in, 'images')
        source_labels = os.path.join(source_dir, split_in, 'labels')
        
        if not os.path.exists(source_images):
            print(f"  ⚠️ Skipping {split_out} - source not found: {source_images}")
            continue
        
        # Create output directories
        output_dirs = {
            'pre': os.path.join(output_dir, split_out, 'pre'),
            'post': os.path.join(output_dir, split_out, 'post'),
            'pre_labels': os.path.join(output_dir, split_out, 'pre_labels'),
            'post_labels': os.path.join(output_dir, split_out, 'post_labels')
        }
        
        for dir_path in output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Get all pre-disaster images
        pre_images = glob.glob(os.path.join(source_images, '*_pre_disaster.tif'))
        
        print(f"  Found {len(pre_images)} image pairs")
        
        for pre_img_path in tqdm(pre_images, desc=f"  Copying {split_out}"):
            base_name = os.path.basename(pre_img_path)
            disaster_name = base_name.replace('_pre_disaster.tif', '')
            
            pre_img = pre_img_path
            post_img = os.path.join(source_images, f"{disaster_name}_post_disaster.tif")
            pre_json = os.path.join(source_labels, f"{disaster_name}_pre_disaster.json")
            post_json = os.path.join(source_labels, f"{disaster_name}_post_disaster.json")
            
            if os.path.exists(pre_img):
                shutil.copy2(pre_img, os.path.join(output_dirs['pre'], base_name))
            
            if os.path.exists(post_img):
                shutil.copy2(post_img, os.path.join(output_dirs['post'], 
                            f"{disaster_name}_post_disaster.tif"))
            
            if os.path.exists(pre_json):
                shutil.copy2(pre_json, os.path.join(output_dirs['pre_labels'], 
                            f"{disaster_name}_pre_disaster.json"))
            
            if os.path.exists(post_json):
                shutil.copy2(post_json, os.path.join(output_dirs['post_labels'], 
                            f"{disaster_name}_post_disaster.json"))
        
        print(f"  ✅ {split_out} complete!")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            n_pre = len(os.listdir(os.path.join(split_dir, 'pre')))
            n_post = len(os.listdir(os.path.join(split_dir, 'post')))
            n_pre_labels = len(os.listdir(os.path.join(split_dir, 'pre_labels')))
            n_post_labels = len(os.listdir(os.path.join(split_dir, 'post_labels')))
            
            print(f"\n{split.upper()}:")
            print(f"  Pre images: {n_pre}")
            print(f"  Post images: {n_post}")
            print(f"  Pre labels: {n_pre_labels}")
            print(f"  Post labels: {n_post_labels}")
            
            if n_pre == n_post == n_pre_labels == n_post_labels:
                print(f"  ✅ All counts match!")
            else:
                print(f"  ⚠️ Warning: Counts don't match!")
    
    print("\n" + "=" * 70)
    print("✅ DATASET ORGANIZATION COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    source_directory = ""  # path to XBD dataset
    output_directory = "geosense/data"  # desired output path
    
    organize_xbd_dataset(source_directory, output_directory)
