import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def plot_image_and_segmentation(case_path, file_name, slice_idx=0, seg_suffix="_seg.nii"):
    """
    Plots a slice of a 3D medical image and its corresponding segmentation.

    Parameters:
    - case_path (str): Path to the folder containing the .nii files.
    - file_name (str): Name of the original image file (e.g., "101_ED.nii").
    - slice_idx (int): Slice index to visualize (default: 0).
    - seg_suffix (str): Suffix for the segmentation file (default: "_seg.nii").
    """

    img_path = os.path.join(case_path, file_name)
    seg_path = os.path.join(case_path, file_name.replace(".nii", seg_suffix))

    assert os.path.exists(img_path), f"Image file not found: {img_path}"
    assert os.path.exists(seg_path), f"Segmentation file not found: {seg_path}"

    img = nib.load(img_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img[:, :, slice_idx], cmap='gray')
    axes[0].set_title("Original image")
    axes[0].axis('off')

    axes[1].imshow(seg[:, :, slice_idx], cmap='gray', alpha=1)
    axes[1].set_title("Segmentation")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def plot_single_completion(case_index, dataset_root="Dataset/Test", slice_mode="center"):
    """
    Visualizes the incomplete vs. completed segmentation of a single cardiac MRI case 
    at a specified slice and for both ED and ES phases. The slice can be selected by index
    and by default is the center slice.
    """

    # Get sorted list of cases
    cases = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
    
    if case_index < 0 or case_index >= len(cases):
        raise ValueError(f"Invalid case index {case_index}. Must be between 0 and {len(cases)-1}.")
    
    case_id = cases[case_index]
    case_path = os.path.join(dataset_root, case_id)

    for phase in ["ED", "ES"]:
        # File paths
        incomplete_path = os.path.join(case_path, f"{case_id}_{phase}_seg.nii")
        complete_path = os.path.join(case_path, f"{case_id}_{phase}_complete_seg.nii")

        # Skip if any file is missing
        if not os.path.exists(incomplete_path) or not os.path.exists(complete_path):
            print(f"Skipping {case_id} {phase} — missing file(s)")
            continue

        # Load segmentations
        incomplete = nib.load(incomplete_path).get_fdata()
        complete = nib.load(complete_path).get_fdata()

        # Determine slice
        if slice_mode == "center":
            slice_idx = incomplete.shape[2] // 2
        elif slice_mode == "max":
            slice_idx = np.argmax([(incomplete[:, :, i] == 2).sum() for i in range(incomplete.shape[2])])
        else:
            slice_idx = int(slice_mode)

        # Plot side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"{case_id} - {phase} Phase - Slice {slice_idx}", fontsize=14)

        axes[0].imshow(incomplete[:, :, slice_idx], cmap='gray')
        axes[0].set_title("Original Incomplete Segmentation")
        axes[0].axis('off')

        axes[1].imshow(complete[:, :, slice_idx], cmap='gray')
        axes[1].set_title("Completed Segmentation (with LV)")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

def plot_class_distribution(y_train, y_test, figsize=(5, 2)):
    """
    Plots the class distribution of the original, training, and test datasets to help 
    assess class balance. Uses normalized histograms to show the proportion of samples 
    for each class across the three subsets.
    """

    fig, axs = plt.subplots(1, 3, sharey=True, figsize=figsize)
    fig.suptitle('Proportion of samples from each class')

    # y_all is the concatenation of y_train and y_test
    y_all = np.concatenate((y_train, y_test))

    axs[0].hist(y_all, weights=np.ones_like(y_all)/len(y_all))
    axs[0].set_xlabel('Original data-set')
    axs[1].hist(y_train, weights=np.ones_like(y_train)/len(y_train))
    axs[1].set_xlabel('Training set')
    axs[2].hist(y_test, weights=np.ones_like(y_test)/len(y_test))
    axs[2].set_xlabel('Test set')
    axs[0].set_ylabel('Proportion')

    plt.tight_layout()
    plt.show()