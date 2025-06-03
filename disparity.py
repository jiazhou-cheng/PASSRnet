import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Set the parent directories
recon_dir = 'results/KITTI2012_patches_masked_fixed_mse_'
recon_dir_L = 'results_onlyL/KITTI2012_patches_masked_fixed'
recon_dir_R = 'results_onlyR/KITTI2012_patches_masked_fixed'
gt_dir = 'data/test/KITTI2012_patches_masked_fixed'  # Change to your actual GT folder

# Stereo matcher
stereo = cv.StereoBM_create(numDisparities=32, blockSize=15)

def compute_disparity(left_path, right_path):
    imgL = cv.imread(left_path, cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(right_path, cv.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        raise FileNotFoundError(f"Could not read {left_path} or {right_path}")
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disparity

def compute_error_map(disp1, disp2):
    valid = np.isfinite(disp1) & np.isfinite(disp2) & (disp1 > 0) & (disp2 > 0)
    error = np.abs(disp1 - disp2)
    error[~valid] = 0  # mask out invalid pixels
    return error, valid

rmse_list = []
error_sum = None
valid_mask_sum = None

for subfolder in sorted(os.listdir(recon_dir)):
    folder_path = os.path.join(recon_dir, subfolder)
    
    # Skip non-directories or non-matching names
    if not os.path.isdir(folder_path) or not subfolder.replace('_', '').isdigit():
        continue

    # File paths
    left_recon = os.path.join(folder_path, 'recon_L.png')
    right_recon = os.path.join(folder_path, 'recon_R.png')

    left_recon = os.path.join(recon_dir_L, subfolder, 'recon_L.png')
    right_recon = os.path.join(recon_dir_R, subfolder, 'recon_L.png')

    left_gt = os.path.join(gt_dir, subfolder, 'hr0.png')
    right_gt = os.path.join(gt_dir, subfolder, 'hr1.png')

    if not (os.path.exists(left_recon) and os.path.exists(right_recon) and
            os.path.exists(left_gt) and os.path.exists(right_gt)):
        print(f"Skipping {subfolder}: Missing one or more images.")
        continue

    # Compute disparities
    disp_recon = compute_disparity(left_recon, right_recon)
    disp_gt = compute_disparity(left_gt, right_gt)

    error_map, valid_mask = compute_error_map(disp_recon, disp_gt)
    rmse = np.sqrt(np.mean((error_map[valid_mask])**2)) if np.any(valid_mask) else np.nan
    rmse_list.append(rmse)

    # Accumulate errors and valid masks
    if error_sum is None:
        error_sum = error_map
        valid_sum = valid_mask.astype(np.int32)
    else:
        error_sum += error_map
        valid_sum += valid_mask.astype(np.int32)


# Compute average error map
mean_error_map = error_sum / np.maximum(valid_sum, 1)

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(mean_error_map, cmap='magma')
plt.colorbar(label='Mean Absolute Disparity Error (pixels)')
plt.title('Mean Disparity Error Map Baseline')
plt.axis('off')
plt.tight_layout()
plt.savefig('baseline stereo.png')

print(f"\nAverage RMSE across {len(rmse_list)} samples: {np.nanmean(rmse_list):.4f}")