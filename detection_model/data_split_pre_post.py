import os
import shutil

base_dir = base_dir = '' # Add base directory path
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

images = os.listdir(images_dir)
labels = os.listdir(labels_dir)

print(f"total images -> {len(images)}, total labels -> {len(labels)}")

images.sort()
labels.sort()

def make_cat_dirs(categories):
    for cat in categories:
        os.makedirs(os.path.join(os.getcwd(), cat), exist_ok=True)


def make_pairs(images, labels):
    paired_imgs = []
    paired_lbls = []
    idx = 0
    while idx < len(images) - 1:
        img1 = images[idx]
        img2 = images[idx + 1]
        lbl1 = labels[idx]
        lbl2 = labels[idx + 1]

        paired_imgs.append((img1,img2))
        paired_lbls.append((lbl1,lbl2))
        idx += 2

    print(f"total paired images -> {len(paired_imgs)}, total paired labels -> {len(paired_lbls)}")
    return paired_imgs, paired_lbls


def move_files(paired_imgs, paired_lbls):
    for i in range(len(paired_imgs)):
        post_img, pre_img = paired_imgs[i]
        post_lbl, pre_lbl = paired_lbls[i]

        shutil.move(os.path.join(images_dir, pre_img), os.path.join(os.getcwd(), 'pre_images', pre_img))
        shutil.move(os.path.join(images_dir, post_img), os.path.join(os.getcwd(), 'post_images', post_img))
        shutil.move(os.path.join(labels_dir, pre_lbl), os.path.join(os.getcwd(), 'pre_labels', pre_lbl))
        shutil.move(os.path.join(labels_dir, post_lbl), os.path.join(os.getcwd(), 'post_labels', post_lbl))


def check_split(categories):
    issue_cnt = 0
    issues = []
    for cat in categories:
        cat_dir = os.path.join(os.getcwd(), cat)
        files = os.listdir(cat_dir)
        print(f"total files in {cat} -> {len(files)}")
    
        if 'pre' in cat:
            for file in files:
                if 'post' in file:
                    issue_cnt += 1
                    issues.append(file)
        else:
            for file in files:
                if 'pre' in file:
                    issue_cnt += 1
                    issues.append(file)
                    
    return issue_cnt, issues


if __name__ == "__main__":
    categories = ['pre_images', 'post_images', 'pre_labels', 'post_labels']
    make_cat_dirs(categories)
    paired_imgs, paired_lbls = make_pairs(images, labels)
    move_files(paired_imgs, paired_lbls)
    issue_cnt, issues = check_split(categories)
    if issue_cnt == 0:
        print("Data split successful!")
    else:
        print(f"Data split has {issue_cnt} issues. Check files: {issues}")