from xbd_dataset import XBDDataset
import numpy as np

dataset = XBDDataset(
    '/kaggle/input/xbd-organized-zipped/data/train/pre', 
    '/kaggle/input/xbd-organized-zipped/data/train/post',
    '/kaggle/input/xbd-organized-zipped/data/train/pre_labels', 
    '/kaggle/input/xbd-organized-zipped/data/train/post_labels',
    image_size=(1024, 1024), mode='multiclass'
)

# Sample 200 images
all_classes = []
for i in range(min(200, len(dataset))):
    sample = dataset[i]
    unique, counts = np.unique(sample['mask'].numpy(), return_counts=True)
    for cls, cnt in zip(unique, counts):
        all_classes.extend([cls] * cnt)

unique, counts = np.unique(all_classes, return_counts=True)
total = len(all_classes)
print('Class Distribution:')
for cls, cnt in zip(unique, counts):
    print(f'  Class {cls}: {cnt/total*100:.1f}%')