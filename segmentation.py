import numpy as np
import cv2
import os
import nibabel as nib
from plotting_utils import plot_image_and_segmentation
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt

from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage import filters
from skimage.draw import disk, circle_perimeter 
# circle_perimeter if you only want boundary, disk is for full area
from skimage.morphology import dilation, disk


###############################################################################################
############################ Simple filling segmentation ######################################
###############################################################################################
# Filling the left ventricle from myocardium contours in a 2D slice and applying it 
# slice-by-slice to a 3D segmentation.

def fill_inside_myocardium(slice_seg):
    """
    Returns a new segmentation slice where the region enclosed by myocardium (label 2)
    is filled as label 3 (LV)
    """
    # Create myocardium binary mask
    myocardium_mask = np.ascontiguousarray((slice_seg == 2).astype(np.uint8))

    # Find contours of myocardium
    contours, _ = cv2.findContours(myocardium_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create empty mask for the LV and make it C-contiguous
    lv_inside = np.ascontiguousarray(np.zeros_like(slice_seg, dtype=np.uint8))

    # Fill only the inside of the myocardium contour
    cv2.drawContours(lv_inside, contours, -1, color=1, thickness=-1)

    # Remove myocardium from the filled mask
    lv_inside[slice_seg == 2] = 0

    # Create new segmentation
    new_seg = slice_seg.copy()
    new_seg[lv_inside == 1] = 3  # Label for LV

    return new_seg

def complete_lv_segmentation(segmentation_3d):
    """
    Completes the left ventricle segmentation by filling the region inside the myocardium 
    for each 2D slice in a 3D segmentation volume.
    """
    filled_seg = segmentation_3d.copy()
    for i in range(segmentation_3d.shape[2]):
        filled_seg[:, :, i] = fill_inside_myocardium(segmentation_3d[:, :, i])
    return filled_seg

def complete_and_save_all_test_segmentations(dataset_root):
    """
    Iterates through all test cases in the dataset directory, completes the LV segmentation 
    using the filling method and saves the updated segmentations with a '_simple_seg.nii' suffix.
    """
    for case_id in sorted(os.listdir(dataset_root)):
        case_path = os.path.join(dataset_root, case_id)
        if not os.path.isdir(case_path):
            continue

        for phase in ["ED", "ES"]:
            seg_filename = f"{case_id}_{phase}_seg.nii"
            seg_path = os.path.join(case_path, seg_filename)

            # Skip if file doesn't exist
            if not os.path.exists(seg_path):
                print(f"Missing: {seg_path}")
                continue

            # Load segmentation
            nii = nib.load(seg_path)
            seg_data = nii.get_fdata()

            # Complete segmentation
            completed_seg = complete_lv_segmentation(seg_data)

            # Convert to NIfTI and save with _complete_seg suffix
            completed_nii = nib.Nifti1Image(completed_seg.astype(np.uint8), affine=nii.affine, header=nii.header)
            save_path = os.path.join(case_path, f"{case_id}_{phase}_simple_seg.nii")
            nib.save(completed_nii, save_path)

#####################################################################################################
################################ Left Ventricle Segmentation ########################################
#####################################################################################################

# Initial tries with Hough (as in the first paper) VERSION 1
def estimate_seed_hough(phase1_slice, phase10_slice, radius_range=(15, 45), plot=False):
    """
    Returns:
    - seed : (x, y) integer coordinates of the estimated LV center (or None if not found)
    """
    # 1) Compute the magnitude of subtraction
    diff = np.abs(phase1_slice - phase10_slice)
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))  # Normalize to [0, 1]

    ################
    # Remove low-intensity background in the difference
    # e.g., any value < 10% of max is set to 0
    th = 0.1 * diff.max()
    diff[diff < th] = 0


    # 2) Edge detection on the subtraction image
    edges = canny(diff, sigma=2)

    # 3) Circular Hough Transform over the detected edges
    radii = np.arange(radius_range[0], radius_range[1], 2)
    hough_res = hough_circle(edges, radii)

    # Extract the most significant circle (peak)
    accums, cx, cy, out_radii = hough_circle_peaks(
        hough_res, radii, total_num_peaks=1
    )

    # If we can't detect any circle, return None
    if len(cx) == 0:
        print("[Hough] No circle detected. Try adjusting radius_range or check your slices.")
        return None

    # The seed is the center of the detected circle
    seed = (cx[0], cy[0])

    # Optionally, plot the result for debugging
    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(phase1_slice, cmap='gray')
        ax.scatter(cx[0], cy[0], color='cyan', s=50, label='Seed (Hough)')

        # Draw the detected circle perimeter
        circ_x, circ_y = circle_perimeter(int(cx[0]), int(cy[0]), int(out_radii[0]))
        circ_y = np.clip(circ_y, 0, diff.shape[0] - 1)
        circ_x = np.clip(circ_x, 0, diff.shape[1] - 1)
        ax.plot(circ_x, circ_y, 'r.', markersize=1, label='Detected Circle')

        ax.set_title("Initial LV Seed (Hough Transform) - Mid-Ventricular Slice")
        ax.legend()
        plt.show()

    return seed


# VERSION 2
def estimate_seed_hough_penalty(
    phase1_slice,
    phase10_slice,
    radius_range=(15, 45),
    plot=False,
    num_peaks=5,
    t_bright=0.5,
    t_dark=0.2,
    alpha=1.0
):

    # 1) Create the difference image and normalize to [0, 1]
    diff = np.abs(phase1_slice - phase10_slice).astype(np.float32)
    mn, mx = diff.min(), diff.max()
    if (mx - mn) < 1e-8:
        print("[Hough] Subtraction is nearly uniform, cannot detect edges.")
        return None
    diff = (diff - mn) / (mx - mn)

    # 2) Edge detection
    edges = canny(diff, sigma=2)

    # 3) Circular Hough transform
    radii_candidates = np.arange(radius_range[0], radius_range[1], 2)
    hough_res = hough_circle(edges, radii_candidates)

    # We collect multiple peak circles
    accums, cx, cy, out_radii = hough_circle_peaks(
        hough_res, radii_candidates,
        total_num_peaks=num_peaks
    )

    if len(cx) == 0:
        print("[Hough] No circle detected. Try adjusting radius_range or the slice.")
        return None

    # (A) For each circle, compute "penalized overlap" score:
    #     score = (# bright) - alpha * (# dark)

    best_score = -np.inf
    best_idx = 0

    for i in range(len(cx)):
        center_x = int(cx[i])
        center_y = int(cy[i])
        radius = int(out_radii[i])

        # Build a filled circle mask
        rr, cc = disk((center_y, center_x), radius, shape=diff.shape)

        # Count bright pixels
        bright_mask = diff[rr, cc] >= t_bright
        S_bright = np.sum(bright_mask)

        # Count dark pixels
        dark_mask = diff[rr, cc] <= t_dark
        S_dark = np.sum(dark_mask)

        score = S_bright - alpha * S_dark

        if score > best_score:
            best_score = score
            best_idx = i

    # The chosen circle
    chosen_x = cx[best_idx]
    chosen_y = cy[best_idx]
    chosen_radius = out_radii[best_idx]
    seed = (chosen_x, chosen_y)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(phase1_slice, cmap='gray', vmin=phase1_slice.min(), vmax=phase1_slice.max())
        ax.set_title("Chosen Circle with Penalized Overlap Score")

        # Mark the chosen circle
        ax.scatter(chosen_x, chosen_y, color='cyan', s=60, label='Chosen Seed')
        # If you want to just show the perimeter:
        # from skimage.draw import circle_perimeter
        # circ_x, circ_y = circle_perimeter(int(chosen_y), int(chosen_x), int(chosen_radius))
        # ax.plot(circ_y, circ_x, 'r.', markersize=1)

        # Alternatively, show a rough fill by plotting disk coords
        rr_chosen, cc_chosen = disk((chosen_y, chosen_x), int(chosen_radius), shape=diff.shape)
        ax.plot(cc_chosen, rr_chosen, 'r.', markersize=1, label='Chosen Circle')

        # Plot the other circles in a smaller marker to compare
        for i in range(len(cx)):
            if i != best_idx:
                ax.scatter(cx[i], cy[i], c='yellow', marker='x', s=30)

        ax.legend()
        plt.show()

    return seed

# VERSION 3

def estimate_seed_hough_pick_center(
    phase1_slice,
    phase10_slice,
    radius_range=(15, 45),
    plot=False,
    num_peaks=5
):
   
    # 1) Compute the subtraction + normalize
    diff = np.abs(phase1_slice - phase10_slice).astype(np.float32)
    mn, mx = diff.min(), diff.max()
    if (mx - mn) < 1e-8:
        print("[Hough] Subtraction is nearly uniform; no edges.")
        return None
    diff = (diff - mn) / (mx - mn)

    # 2) Edge detection
    edges = canny(diff, sigma=2)

    # 3) Circular Hough
    radii_candidates = np.arange(radius_range[0], radius_range[1], 2)
    hough_res = hough_circle(edges, radii_candidates)

    # We ask for multiple circle candidates
    accums, cx, cy, out_radii = hough_circle_peaks(
        hough_res, radii_candidates, total_num_peaks=num_peaks
    )

    if len(cx) == 0:
        print("[Hough] No circle detected.")
        return None

    # 'cx[i]' is x, 'cy[i]' is y in the image's row-col sense
    # Let's pick the circle whose center is closest to the image center
    h, w = diff.shape
    center_x = w / 2
    center_y = h / 2

    best_idx = None
    best_dist = float('inf')

    for i in range(len(cx)):
        # distance from center
        dist_center = np.hypot(cx[i] - center_x, cy[i] - center_y)
        if dist_center < best_dist:
            best_dist = dist_center
            best_idx = i

    chosen_x = cx[best_idx]   # column
    chosen_y = cy[best_idx]   # row
    chosen_radius = out_radii[best_idx]

    # We'll return the seed in (row, col) so it's straightforward to use for array indexing
    # seed = (int(chosen_y), int(chosen_x)) me parece q esta mal
    seed = (int(chosen_x), int(chosen_y))


    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(phase1_slice, cmap='gray')
        ax.set_title("Chosen Circle (Closest to Image Center)")

        # Plot all detected circles in yellow
        for i in range(len(cx)):
            # scatter() wants (x, y) → (cx[i], cy[i])
            ax.scatter(cx[i], cy[i], c='yellow', marker='x', s=30)

        # Mark the chosen circle in cyan
        ax.scatter(chosen_x, chosen_y, c='cyan', s=60, label='Chosen Seed')

        # Plot perimeter
        # circle_perimeter expects (row, col, radius) = (chosen_y, chosen_x, chosen_radius)
        rr, cc = circle_perimeter(int(chosen_y), int(chosen_x), int(chosen_radius))
        rr = np.clip(rr, 0, h - 1)
        cc = np.clip(cc, 0, w - 1)
        # plot() wants x=cc, y=rr
        ax.plot(cc, rr, 'r.', markersize=1, label='Chosen Circle')

        ax.legend()
        plt.show()

    return seed


##############################################################################################################
############################################ My step 1 #######################################################

def get_initial_seed(case_number, phase="ED", slice_idx=None, base_path='./DatasetLucianaMunhos/Dataset/Test/'):
    """
    Returns (seed_x, seed_y) as the center of the myocardium in the given slice.

    Parameters:
        case_number (str or int): e.g., "101"
        phase (str): either "ED" or "ES"
        slice_idx (int or None): index of the slice to use; if None, middle slice is chosen
        base_path (str): path to the Dataset/Test base folder

    Returns:
        (int, int): Coordinates (x, y) of the seed in the slice
    """
    case_number = str(case_number)
    folder_path = os.path.join(base_path, case_number)

    image_filename = f"{case_number}_{phase}.nii"
    seg_filename = f"{case_number}_{phase}_seg.nii"

    image_path = os.path.join(folder_path, image_filename)
    seg_path = os.path.join(folder_path, seg_filename)

    image = nib.load(image_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()

    num_slices = image.shape[2]

    if slice_idx is None:
        slice_idx = num_slices // 2
    elif not (0 <= slice_idx < num_slices):
        raise ValueError(f"slice_idx {slice_idx} out of range (0 to {num_slices-1})")

    seg_slice = seg[:, :, slice_idx]

    myocardium_mask = seg_slice == 2

    if not np.any(myocardium_mask):
        raise ValueError(f"No MYO found in slice {slice_idx} of {seg_filename}")

    y, x = center_of_mass(myocardium_mask)

    return int(round(x)), int(round(y))

#########################################################################################################
################################################ Step 2 #################################################

def region_growing_from_seed(image, seed, max_diff=0.2):
    """
    Performs region growing from a given seed point in a 2D image. Neighboring pixels are 
    added to the region if their intensity is within `max_diff` of the region's current mean. 
    Returns the binary mask of the grown region along with its mean and standard deviation.
    """

    h, w = image.shape
    visited = np.zeros_like(image, dtype=bool)
    mask = np.zeros_like(image, dtype=bool)

    x0, y0 = int(seed[0]), int(seed[1])
    region_values = [image[y0, x0]]
    threshold = max_diff

    stack = [(x0, y0)]
    visited[y0, x0] = True
    mask[y0, x0] = True

    while stack:
        x, y = stack.pop()
        current_mean = np.mean(region_values)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                xn, yn = x + dx, y + dy
                if 0 <= xn < w and 0 <= yn < h and not visited[yn, xn]:
                    intensity = image[yn, xn]
                    if abs(intensity - current_mean) <= threshold:
                        visited[yn, xn] = True
                        mask[yn, xn] = True
                        region_values.append(intensity)
                        stack.append((xn, yn))

    mean_intensity = image[mask].mean() if mask.any() else 0
    std_intensity = image[mask].std() if mask.any() else 0

    return mask, mean_intensity, std_intensity


##############################################################################################################
################################################## Step 3 ####################################################

def fit_plane_least_squares(mask, image):
    """
    Fits a 2D plane to the intensity values of the given image inside a binary mask 
    using least squares regression. Returns the fitted plane and its coefficients.
    """

    # Get coordinates of full-blood voxels
    y, x = np.nonzero(mask)  # rows (y), columns (x)
    z = image[y, x]  # corresponding intensities

    # Design matrix for plane: z = ax + by + c
    A = np.c_[x, y, np.ones_like(x)]
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs

    # Generate fitted plane over full image
    xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    plane = a * xx + b * yy + c

    # print("Min plane:", np.min(plane))
    # print("Max plane:", np.max(plane))
    # print("Mean plane:", np.mean(plane))

    return plane, coeffs

def correct_coil_bias(image, plane):
    """
    Applies coil bias correction by subtracting the fitted plane from the original image.
    """
    return image.astype(np.float32) - plane


def show_comparison(original, corrected, plane):
    """
    Displays side-by-side visualizations of the original image, fitted coil bias (plane), 
    and bias-corrected image for comparison.
    """

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Original Image")
    axs[1].imshow(plane, cmap='gray')
    axs[1].set_title("Fitted Coil Bias (Plane)")
    axs[2].imshow(corrected, cmap='gray')
    axs[2].set_title("Corrected Image")
    plt.show()


def step3(mask, image):
    """
    Performs coil bias correction on the input image by fitting a plane over the masked 
    region and subtracting it. Returns the corrected image, fitted plane, and coefficients.
    """
    plane, coeffs = fit_plane_least_squares(mask, image)
    corrected_image = correct_coil_bias(image, plane)
    #show_comparison(image, corrected_image, plane) # optional visualization
    return corrected_image, plane, coeffs

## Normalization of the corrected_image
def normalize_image(image):
    """
    Normalize the image to the range [0, 1].
    """
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val > 1e-8:
        normalized_image = (image - min_val) / (max_val - min_val)
    else:
        normalized_image = np.zeros_like(image)

    return normalized_image	

##############################################################################################################
################################################## Step 4 ####################################################

def estimate_myocmean(image, seed, thresholds, growth_limit=600, region_mask_seed = None):
    """
    Estimates the mean and standard deviation of the myocardium intensity by performing 
    region growing from a seed across multiple thresholds. Stops if region growth 
    explodes. Then approximates the myocardium as a ring around the final left ventricle 
    mask using morphological dilation.

    Returns the myocardium mean and std, list of region volumes, used thresholds, 
    growth ratios and the final LV mask before explosion.
    """

    volumes = []
    masks = []
    growth_ratios = []
    previous_volume = None
    jump_index = None

    for i, th in enumerate(thresholds):
        mask = region_grow_threshold(image, seed, th)

        # for the case when it looses the LV and takes the RV by accident (they touch each other with same intesity)
        if not mask[seed[1], seed[0]]:
            jump_index = i
            print(f"Threshold {th:.2f} EXCLUDED — seed no longer in mask. Stopping.")
            break  # Important: stop iteration here!

        volume = np.sum(mask)
        volumes.append(volume)
        masks.append(mask)

        #plt.imshow(mask, cmap='gray')
        # in the title say which image it is
        #plt.title(f"Threshold: {th:.2f} (Volume: {volume})")
        #plt.axis('off')
        #plt.show()

        if previous_volume is not None:
            growth = volume - previous_volume
            growth_ratios.append(growth)
            
            if growth > growth_limit and i > 1:
                print("Index of the explosion:", i)
                jump_index = i
                print(f"Growth exploded at threshold {th:.2f} (ΔV = {growth})")
                break

        previous_volume = volume

    volumes = np.array(volumes)

    print("Volumes:", volumes)

    # Decide which mask to use
    if jump_index is None:
        print("No significant growth detected.")
        final_mask = masks[0]
    elif jump_index == 0:
        # use the previous mask 
        final_mask = region_mask_seed
    else:
        final_mask = masks[jump_index - 1]

    # plt.imshow(final_mask, cmap='gray')
    # plt.title("Intermediate LV mask (before explosion)")
    # plt.axis('off')
    # plt.show()

    # # Estimate MYO by dilating and subtracting LV
    dilated_mask = dilation(final_mask, disk(5))
    myoc_mask = dilated_mask & ~final_mask

    myoc_mean = np.mean(image[myoc_mask])
    myoc_std = np.std(image[myoc_mask])

    return myoc_mean, myoc_std, volumes, thresholds[:len(volumes)], growth_ratios, final_mask


##############################################################################################################
################################################## Step 5 ####################################################

def region_grow_threshold(image, seed, threshold):
    """
    Performs region growing starting from a seed pixel, including only neighboring pixels 
    with intensity greater than or equal to the given threshold. Grows only to connected 
    pixels meeting the criterion, resulting in a binary mask of the region.
    """
    h, w = image.shape
    visited = np.zeros_like(image, dtype=bool)
    mask = np.zeros_like(image, dtype=bool)

    x0, y0 = int(seed[0]), int(seed[1]) 

    stack = [(x0, y0)]
    visited[y0, x0] = True
    mask[y0, x0] = True

    while stack:
        x, y = stack.pop()

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                xn, yn = x + dx, y + dy
                if 0 <= xn < w and 0 <= yn < h and not visited[yn, xn]:
                    intensity = image[yn, xn]
                    if intensity >= threshold:
                        visited[yn, xn] = True
                        mask[yn, xn] = True
                        stack.append((xn, yn))

    return mask


##############################################################################################################
############################################# Full segmentation ##############################################

def process_slice(case, slice_idx, vol, seg, phase="ED", base_path = './DatasetLucianaMunhos/Dataset/Test/'):
    """
    Processes a single MRI slice by selecting an initial seed inside the myocardium, 
    performing region growing, correcting coil bias, estimating myocardium intensity, 
    and generating the final left ventricle segmentation mask through adaptive thresholding.
    
    Returns the final binary mask for the LV region in that slice.
    """

    myocardium_mask = seg[:, :, slice_idx] == 2
    if not np.any(myocardium_mask):
        print(f"No myocardium found in slice {slice_idx} of {case}")
        empty_mask = np.zeros_like(vol[:, :, slice_idx], dtype=bool)
        return empty_mask

    img_slice = vol[:, :, slice_idx]

    # Step 1: Get seed
    best_seed = get_initial_seed(case_number=case, phase=phase, slice_idx=slice_idx, base_path= base_path)
    seed_x, seed_y = best_seed

    # # Display the seed on top of the slice
    # plt.imshow(img_slice, cmap='gray')
    # plt.scatter(best_seed[0], best_seed[1], color='red', s=50)
    # plt.title(f"Initial Seed from Myocardium - Case {case} Slice {slice_idx}")
    # plt.axis('off')
    # plt.show()
    
    # Step 2: Region growing
    region_mask, LVmean, LVstd = region_growing_from_seed(img_slice, best_seed, max_diff=25)

    # # plot the region mask on the slice
    # plt.imshow(img_slice, cmap='gray')
    # plt.imshow(region_mask, alpha=0.5, cmap='jet')
    # plt.title("Region Grown from Seed")
    # plt.axis('off')
    # plt.show()

    # Step 3: Correction
    corrected_image, plane, coeffs = step3(region_mask, img_slice)
    corrected_image = normalize_image(corrected_image) * 255
    seed_val = corrected_image[seed_y, seed_x]

    # Step 4: Threshold estimation : LVmean and lv_std in the corrected image
    LVmean = np.mean(corrected_image[region_mask]) 
    LVstd = np.std(corrected_image[region_mask])

    print("LVmean in corrected image:", LVmean)
    print("LVstd in corrected image:", LVstd)
    print("Seed value in corrected image:", seed_val)

    max_threshold = max(seed_val, LVmean)

    if max_threshold > 100:
        num = 35
    elif max_threshold > 50:
        num = 20
    else:
        num = 12

    thresholds = np.linspace(int(max_threshold), 0, num=num)

    # region_mask is the result of step 2, the region growing for the seed
    myoc_mean, myoc_std, volumes, ths, growth, final_mask = estimate_myocmean(corrected_image, best_seed, thresholds, region_mask_seed=region_mask) 
    #print("the quantity of thresholds: ", len(thresholds))
    #print("First threshold:", thresholds[0])

    # # Ground-truth myocardium stats just to compare
    # seg_slice = seg[:, :, slice_idx]
    # myoc_mean_given_mask = np.mean(corrected_image[seg_slice == 2])
    # myocardium_std = np.std(corrected_image[seg_slice == 2])
    # print("Ground-truth Myocardium mean intensity:", myoc_mean_given_mask)
    
    print("Myoc_mean: ", myoc_mean)
    print("Myoc_std: ", myoc_std)

    # Step 5: Final LV segmentation
    LVstd = np.std(corrected_image[final_mask])
    print("LV std deviation:", LVstd)

    secure_final_threshold_step5 = (myoc_mean+seed_val)/2
    #secure_final_threshold_step5 = myoc_mean + 0.5 * LVstd
    print("Initial threshold for final step:", secure_final_threshold_step5)
    
    continue_iterating = True
    while continue_iterating:
        print(f"Trying with threshold = {secure_final_threshold_step5}")
        final_lv_mask = region_grow_threshold(corrected_image, best_seed, secure_final_threshold_step5)
        actual_volume = np.sum(final_lv_mask)
        print(f"Volume: {actual_volume}")

        if actual_volume > 4000:
            print("Volume is too big. Increasing threshold.")
            secure_final_threshold_step5 += 2
        else:
            print("Volume is acceptable. Stopping iteration.")
            continue_iterating = False

    # Optional visualization
    plt.imshow(corrected_image, cmap='gray')
    plt.imshow(final_lv_mask, alpha=0.5, cmap='jet')
    plt.title(f"Final LV Mask - Case {case} Slice {slice_idx}")
    plt.axis('off')
    plt.show()

    return final_lv_mask


def segment_all_cases_left_ventricle(
    test_cases=range(101, 151), 
    phase="ED", 
    output_suffix="_complete_seg.nii", 
    base_path='./DatasetLucianaMunhos/Dataset/Test/'
):
    """
    Segments the left ventricle for all test cases and slices by calling `process_slice` 
    on each slice. Saves updated segmentation volumes with the LV filled in as label 3.
    Handles missing files and errors and supports both local and Colab paths.
    """

    for case_num in test_cases:
        case = str(case_num)
        print(f"Processing case {case}...")

        # Build paths
        vol_path = f"{base_path}{case}/{case}_{phase}.nii"
        seg_path = f"{base_path}{case}/{case}_{phase}_seg.nii"
        output_path = f"{base_path}{case}/{case}_{phase}{output_suffix}"

        if not os.path.exists(vol_path) or not os.path.exists(seg_path):
            print(f"Missing files for case {case}, skipping.")
            continue

        # Load volume and segmentation
        vol_nii = nib.load(vol_path)
        vol = vol_nii.get_fdata()
        affine = vol_nii.affine

        seg = nib.load(seg_path).get_fdata()
        num_slices = vol.shape[2]

        # Make a copy to modify
        complete_seg = seg.copy()

        for slice_idx in range(num_slices):
            try:
                lv_mask = process_slice(case, slice_idx, vol, seg, phase=phase, base_path=base_path)    
                # Only update where current seg is 0 (background)
                overwrite_mask = (lv_mask == 1) & (complete_seg[:, :, slice_idx] == 0)
                complete_seg[:, :, slice_idx][overwrite_mask] = 3  # Label 3 for LV
            except ValueError as e:
                message = str(e)
                if "out of range" in message:
                    print(f"Slice index error: {message}")
                elif "found in slice" in message:
                    print(f"No myocardium in slice: {message}")
                else:
                    print(f"Unexpected ValueError: {message}")
            except Exception as e:
                print(f"Error processing case {case}, slice {slice_idx}: {e}")

        nib.save(nib.Nifti1Image(complete_seg.astype(np.uint8), affine), output_path)
        print(f"Saved updated segmentation to {output_path}")
