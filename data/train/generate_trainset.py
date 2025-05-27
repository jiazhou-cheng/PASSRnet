import os
import numpy as np
from numpy import random
from PIL import Image
from glob import glob
import cv2

# Settings
mask_mode = 'fixed'  # 'fixed', 'random_rect', 'irregular'
image_dir = 'data/train/Flickr1024/'
output_dir = 'data/train/Flickr1024_patches_masked_' + mask_mode + '/'
patch_h, patch_w = 30*4, 90*4  # HR patch size
stride = 20*4

# create fixed rect mask at fixed location
def generate_fixed_centered_mask(patch_h, patch_w, mask_ratio=0.25):

    mask_h = round(patch_h * mask_ratio)
    mask_w = round(patch_w * mask_ratio)
    mask_x = round((patch_h - mask_h) / 2)
    mask_y = round((patch_w - mask_w) / 2)

    fixed_mask = np.ones((patch_h, patch_w), dtype=np.uint8)
    fixed_mask[mask_x:mask_x + mask_h, mask_y:mask_y + mask_w] = 0
    fixed_mask3 = np.stack([fixed_mask] * 3, axis=-1)  # For RGB images

    return fixed_mask, fixed_mask3

# create rect mask at random location
def generate_random_rect_mask(patch_h, patch_w, rect_h_ratio=0.25, rect_w_ratio=0.25):

    rect_h = int(patch_h * rect_h_ratio)
    rect_w = int(patch_w * rect_w_ratio)

    x0 = random.randint(rect_h, patch_h - rect_h)
    y0 = random.randint(rect_w, patch_w - rect_w)

    mask = np.ones((patch_h, patch_w), dtype=np.uint8)
    mask[x0:x0 + rect_h, y0:y0 + rect_w] = 0
    mask3 = np.stack([mask] * 3, axis=-1)  # For RGB images

    return mask, mask3

# create random watermark mask
def generate_random_irregular_mask(patch_h, patch_w, min_area_ratio=0.05, max_area_ratio=0.2):

    mask = np.ones((patch_h, patch_w), dtype=np.uint8)
    blob = np.zeros((patch_h, patch_w), dtype=np.uint8)
    num_blobs = np.random.randint(1, 5)  # Number of blobs to create

    for _ in range(num_blobs):
        center_x = np.random.randint(0, patch_w)
        center_y = np.random.randint(0, patch_h)
        max_radius = int(np.sqrt(patch_h * patch_w * np.random.uniform(min_area_ratio, max_area_ratio)) // 2)

        shape_type = np.random.choice(['ellipse', 'poly'])

        if shape_type == 'ellipse':
            axes = (
                np.random.randint(max_radius // 2, max_radius),
                np.random.randint(max_radius // 2, max_radius)
            )
            angle = np.random.randint(0, 360)
            cv2.ellipse(blob, (center_x, center_y), axes, angle, 0, 360, 255, -1)

        else:  # random polygon
            num_points = np.random.randint(3, 6)
            pts = np.array([
                [center_x + np.random.randint(-max_radius, max_radius),
                 center_y + np.random.randint(-max_radius, max_radius)]
                for _ in range(num_points)
            ], dtype=np.int32)
            pts = pts.clip([[0, 0]], [[patch_w - 1, patch_h - 1]])
            cv2.fillPoly(blob, [pts], 255)

    # Smooth shapes
    kernel = np.ones((5, 5), np.uint8)
    blob = cv2.dilate(blob, kernel, iterations=1)
    blob = cv2.erode(blob, kernel, iterations=1)

    mask[blob > 0] = 0
    mask3 = np.stack([mask] * 3, axis=-1)

    return mask, mask3

img_paths = sorted(glob(os.path.join(image_dir, '*.png')))

for idx_file in range(0, len(img_paths), 2):
    file_name = os.path.basename(img_paths[idx_file]).split('_')[0]

    img_0 = np.array(Image.open(img_paths[idx_file]))
    img_1 = np.array(Image.open(img_paths[idx_file + 1]))

    h = (img_0.shape[0] // patch_h) * patch_h
    w = (img_0.shape[1] // patch_w) * patch_w
    img_0 = img_0[:h, :w, :]
    img_1 = img_1[:h, :w, :]

    idx_patch = 1
    for x_hr in range(0, h - patch_h + 1, stride):
        for y_hr in range(0, w - patch_w + 1, stride):
            patch_0 = img_0[x_hr:x_hr+patch_h, y_hr:y_hr+patch_w, :]
            patch_1 = img_1[x_hr:x_hr+patch_h, y_hr:y_hr+patch_w, :]

            if mask_mode == 'fixed':
                mask, mask3 = generate_fixed_centered_mask(patch_h, patch_w)
            elif mask_mode == 'random_rect':
                mask, mask3 = generate_random_rect_mask(patch_h, patch_w)
            elif mask_mode == 'irregular':
                mask, mask3 = generate_random_irregular_mask(patch_h, patch_w)
            else:
                raise ValueError("Invalid mask mode. Choose from 'fixed', 'random_rect', or 'irregular'.")

            masked_0 = patch_0 * mask3
            masked_1 = patch_1 * mask3

            patch_folder = os.path.join(output_dir, f'{file_name}_{idx_patch:03d}')
            os.makedirs(patch_folder, exist_ok=True)

            Image.fromarray(patch_0).save(os.path.join(patch_folder, 'hr0.png'))
            Image.fromarray(patch_1).save(os.path.join(patch_folder, 'hr1.png'))
            Image.fromarray(masked_0).save(os.path.join(patch_folder, 'masked0.png'))
            Image.fromarray(masked_1).save(os.path.join(patch_folder, 'masked1.png'))
            Image.fromarray((1 - mask3) * 255).convert('L').save(os.path.join(patch_folder, 'mask.png'))

            idx_patch += 1
