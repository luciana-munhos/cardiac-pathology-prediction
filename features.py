import numpy as np
import cv2
import os
import nibabel as nib
import pandas as pd

def volume(segmentation, num, voxel_vol = 1.0):
    """
    Calculate the volume of a 3D segmentation.
    """
    return np.sum(segmentation == num) * voxel_vol

def ejection_fraction(edv, esv):
    """
    Calculate the ejection fraction.
    Ejection fraction is defined as (EDV - ESV) / EDV.
    It is a measure of the percentage of blood pumped out of the heart with each beat.
    """
    return (edv - esv) / edv if edv != 0 else 0

def myocardium_thickness_stats(seg_3d):
    """
    Computes myocardium thickness statistics (label 2) slice-by-slice in the x-y plane.
    The thickness is estimated as area / total perimeter (external + internal contour),
    which is a common approximation in medical image analysis.

    Returns: mean, std, max, min of estimated thicknesses
    """
    thicknesses = []

    for i in range(seg_3d.shape[2]):
        slice_seg = seg_3d[:, :, i].astype(np.uint8)
        myo_mask = (slice_seg == 2).astype(np.uint8)

        if np.sum(myo_mask) == 0:
            continue

        # Use cv2.RETR_CCOMP to retrieve both outer and inner contours
        contours, _ = cv2.findContours(myo_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        total_perimeter = 0
        for contour in contours:
            total_perimeter += cv2.arcLength(contour, True)

        if total_perimeter == 0:
            continue

        area = np.sum(myo_mask)
        thickness = area / (total_perimeter + 1e-5)
        thicknesses.append(thickness)

    if not thicknesses:
        return 0.0, 0.0, 0.0, 0.0

    return np.mean(thicknesses), np.std(thicknesses), np.max(thicknesses), np.min(thicknesses)

def circularity(seg_3d, label):
    """
    Computes mean and max circularity of the given label slice-by-slice in the x-y plane.
    For label 2 (myocardium), uses both outer and inner contours.
    For other labels, uses only the outer contour.

    Circularity = 4 * pi * Area / Perimeter^2
    """
    circularities = []

    for i in range(seg_3d.shape[2]):
        slice_seg = seg_3d[:, :, i].astype(np.uint8)
        mask = (slice_seg == label).astype(np.uint8)

        if np.sum(mask) == 0:
            continue

        # Use both contours for myocardium, only external otherwise
        mode = cv2.RETR_CCOMP if label == 2 else cv2.RETR_EXTERNAL
        contours, _ = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)

        total_area = 0
        total_perimeter = 0
        for contour in contours:
            total_area += cv2.contourArea(contour)
            total_perimeter += cv2.arcLength(contour, True)

        if total_perimeter > 0:
            circ = 4 * np.pi * total_area / (total_perimeter ** 2)
            circularities.append(circ)

    if not circularities:
        return 0.0, 0.0

    return np.mean(circularities), np.max(circularities)

def compute_circumference_stats(seg_3d, label):
    """
    Computes mean and max circumference (perimeter) of a given label slice-by-slice.
    """
    circumferences = []

    for i in range(seg_3d.shape[2]):
        slice_seg = seg_3d[:, :, i].astype(np.uint8)
        mask = (slice_seg == label).astype(np.uint8)

        if np.sum(mask) == 0:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            circumferences.append(perimeter)

    if not circumferences:
        return 0.0, 0.0

    return np.mean(circumferences), np.max(circumferences)

def lvm_thickness_between_lvc_rvc(seg_3d):
    """
    Compute thickness of LVM (label 2) between LVC (label 3) and RVC (label 1)
    in slices where both are present. Returns mean and std of these measurements.
    """
    thicknesses = []

    for i in range(seg_3d.shape[2]):
        slice_seg = seg_3d[:, :, i].astype(np.uint8)
        lvc = (slice_seg == 3).astype(np.uint8)
        rvc = (slice_seg == 1).astype(np.uint8)
        lvm = (slice_seg == 2).astype(np.uint8)

        if np.sum(lvc) > 0 and np.sum(rvc) > 0 and np.sum(lvm) > 0:
            # Approximate LVM thickness between LVC and RVC
            projection = np.sum(lvm, axis=0)
            thickness = np.count_nonzero(projection)
            if thickness > 0:
                thicknesses.append(thickness)

    if not thicknesses:
        return 0.0, 0.0

    return np.mean(thicknesses), np.std(thicknesses)

def rvc_size_at_apical_lvm_slice(seg_3d):
    """
    Finds the most apical slice with LVM present and returns RVC and LVC area and their ratio.
    """
    apical_idx = -1
    for i in range(seg_3d.shape[2]):
        if np.any(seg_3d[:, :, i] == 2):  # LVM is label 2
            apical_idx = i

    if apical_idx == -1:
        return 0.0, 0.0, 0.0

    slice_apex = seg_3d[:, :, apical_idx]
    rvc_area = np.sum(slice_apex == 1)
    lvc_area = np.sum(slice_apex == 3)
    ratio = rvc_area / lvc_area if lvc_area > 0 else 0.0

    return rvc_area, lvc_area, ratio

def compute_mass(myo_volume_mm3, density_g_per_cm3=1.05):
    """
    Estimate the myocardial mass from the volume (in mm³) using myocardial tissue density.
    Default density for myocardium ≈ 1.05 g/cm³.
    """
    volume_cm3 = myo_volume_mm3 / 1000  # Convert mm³ to cm³
    mass_g = volume_cm3 * density_g_per_cm3
    return mass_g

def process_dataset(metadata_path, base_path, id_range, use_completed_seg=False, dropFeatures=False):
    """
    Extracts a full set of anatomical and functional features from cardiac MRI segmentations 
    (ED and ES phases) for each subject, including volumes, ejection fractions, myocardial 
    thickness, circularity metrics, apex ratios, BMI, and BSA-normalized values. 
    Optionally drops lower-priority features if dropFeatures is True.
    """

    # Read metadata
    metadata = pd.read_csv(metadata_path)

    for col in [
            # High Priority
            'EF_LV',                     # Ejection Fraction of Left Ventricle
            'EF_RV',                     # Ejection Fraction of Right Ventricle
            'LV_EDV_BSA',                # LV volume normalized by BSA
            'RV_EDV_BSA',                # RV volume normalized by BSA
            'MYO_ED_BSA',                # Myocardial volume normalized by BSA
            'Mass',                      # Estimated myocardial mass
            'RVC_LVC_Apex_Ratio',        # Ratio between RV and LV areas at apex
            'MYO_Thickness_Mean',        # Mean thickness of myocardium
            'MYO_Thickness_Std',         # Std of myocardial thickness
            'LVM_Thickness_Between_LVC_RVC_Mean',  # Mean thickness between LVC and RVC
            'LVM_Thickness_Between_LVC_RVC_Std',   # Std of thickness between LVC and RVC

            # Medium Priority 
            'MYO_Circ_Max',              # Max circumference of MYO
            'RV_Circ_Max',               # Max circumference of RV
            'MYO_Circ_Mean',              # Mean circumference of MYO
            'RV_Circ_Mean',              # Mean circumference of RV
            'MYO_Thickness_Max',         # Max myocardial thickness
            'MYO_Thickness_Min',         # Min myocardial thickness
            'BMI',                       # Body Mass Index 
            'RV_LV_EDV_Ratio',           # Volume ratio RV / LV
            'MYO_LV_EDV_Ratio',          # Volume ratio MYO / LV

            # Low Priority
            'LV_EDV',                    # Raw LV ED volume
            'LV_ESV',                    # Raw LV ES volume
            'RV_EDV',                    # Raw RV ED volume
            'RV_ESV',                    # Raw RV ES volume
            'MYO_ED',                    # Raw myocardial ED volume
            'MYO_ES',                    # Raw myocardial ES volume
            'BSA',                       # Body Surface Area
            'RVC_Apex_Area',             # RV area at apex
            'LVC_Apex_Area',              # LV area at apex

            'MYO_Circum_Mean',
            'MYO_Circum_Max',
            'RV_Circum_Mean',
            'RV_Circum_Max']:

        metadata[col] = 0.0

    # Loop through each subject
    for idx, subject_num in enumerate(id_range):
        subject_id = f"{subject_num:03d}" 
        subject_path = os.path.join(base_path, subject_id)

        #suffix = "_complete_seg.nii" if use_completed_seg else "_seg.nii"
        if "Train" in base_path:
            suffix = "_seg.nii"
        else:
            if use_completed_seg:
                suffix = "_complete_seg.nii"
            else:
                suffix = "_simple_seg.nii"


        try:
            seg_ed = nib.load(os.path.join(subject_path, f"{subject_id}_ED{suffix}")).get_fdata()
            seg_es = nib.load(os.path.join(subject_path, f"{subject_id}_ES{suffix}")).get_fdata()
        except FileNotFoundError:
            print(f"Missing segmentation for subject {subject_id}, skipping.")
            continue

        # get voxel volume from the header from the .nii file
        voxel_vol = np.prod(nib.load(os.path.join(subject_path, f"{subject_id}_ED{suffix}")).header.get_zooms())
        
        # Calculate volumes
        lv_edv = volume(seg_ed, 3, voxel_vol)
        lv_esv = volume(seg_es, 3, voxel_vol)
        rv_edv = volume(seg_ed, 1, voxel_vol)
        rv_esv = volume(seg_es, 1, voxel_vol)
        myo_ed = volume(seg_ed, 2, voxel_vol)
        myo_es = volume(seg_es, 2, voxel_vol)
        ef_lv = ejection_fraction(lv_edv, lv_esv)
        ef_rv = ejection_fraction(rv_edv, rv_esv)

        metadata.at[idx, 'LV_EDV'] = lv_edv
        metadata.at[idx, 'LV_ESV'] = lv_esv
        metadata.at[idx, 'EF_LV'] = ef_lv
        metadata.at[idx, 'RV_EDV'] = rv_edv
        metadata.at[idx, 'RV_ESV'] = rv_esv
        metadata.at[idx, 'EF_RV'] = ef_rv
        metadata.at[idx, 'MYO_ED'] = myo_ed
        metadata.at[idx, 'MYO_ES'] = myo_es

        # Thickness features (slice-based)
        mean_t, std_t, max_t, min_t = myocardium_thickness_stats(seg_ed)
        metadata.at[idx, 'MYO_Thickness_Mean'] = mean_t
        metadata.at[idx, 'MYO_Thickness_Std'] = std_t
        metadata.at[idx, 'MYO_Thickness_Max'] = max_t
        metadata.at[idx, 'MYO_Thickness_Min'] = min_t

        # Shape features (circularity & circumference)
        myo_circ_mean, myo_circ_max = circularity(seg_ed, 2)  # myocardium
        rv_circ_mean, rv_circ_max = circularity(seg_ed, 1)    # right ventricle cavity

        metadata.at[idx, 'MYO_Circ_Mean'] = myo_circ_mean
        metadata.at[idx, 'MYO_Circ_Max'] = myo_circ_max
        metadata.at[idx, 'RV_Circ_Mean'] = rv_circ_mean
        metadata.at[idx, 'RV_Circ_Max'] = rv_circ_max

        # Circumference ESTA MAL 
        myo_circum_mean, myo_circum_max = compute_circumference_stats(seg_ed, 3) #############################################
        rv_circum_mean, rv_circum_max = compute_circumference_stats(seg_ed, 1)
        metadata.at[idx, 'MYO_Circum_Mean'] = myo_circum_mean
        metadata.at[idx, 'MYO_Circum_Max'] = myo_circum_max
        metadata.at[idx, 'RV_Circum_Mean'] = rv_circum_mean
        metadata.at[idx, 'RV_Circum_Max'] = rv_circum_max

         # Thickness between LVC and RVC
        mean_bt, std_bt = lvm_thickness_between_lvc_rvc(seg_ed)
        metadata.at[idx, 'LVM_Thickness_Between_LVC_RVC_Mean'] = mean_bt
        metadata.at[idx, 'LVM_Thickness_Between_LVC_RVC_Std'] = std_bt

        # Apex features
        rvc_apex, lvc_apex, apex_ratio = rvc_size_at_apical_lvm_slice(seg_ed)
        metadata.at[idx, 'RVC_Apex_Area'] = rvc_apex
        metadata.at[idx, 'LVC_Apex_Area'] = lvc_apex
        metadata.at[idx, 'RVC_LVC_Apex_Ratio'] = apex_ratio

        # Mass
        mass = compute_mass(myo_ed)
        metadata.at[idx, 'Mass'] = mass

        # Calculate BMI 
        height = metadata.at[idx, 'Height']
        weight = metadata.at[idx, 'Weight']
        bmi = weight / ((height / 100) ** 2)
        metadata.at[idx, 'BMI'] = bmi

        # Normalize by BSA
        bsa = np.sqrt((weight * height) / 3600)
        metadata.at[idx, 'BSA'] = bsa
        metadata.at[idx, 'LV_EDV_BSA'] = lv_edv / bsa
        metadata.at[idx, 'RV_EDV_BSA'] = rv_edv / bsa
        metadata.at[idx, 'MYO_ED_BSA'] = myo_ed / bsa

        # Volume ratios
        metadata.at[idx, 'RV_LV_EDV_Ratio'] = rv_edv / lv_edv if lv_edv > 0 else 0
        metadata.at[idx, 'MYO_LV_EDV_Ratio'] = myo_ed / lv_edv if lv_edv > 0 else 0

    # Clean up metadata
    if 'Id' in metadata.columns:
        metadata = metadata.drop(columns=['Id'])

    if dropFeatures:
        columns_to_drop = ['LV_EDV', 'RV_EDV', 'MYO_ED', 'BSA', 'Weight', 'Height'] 
        # Drop them
        metadata = metadata.drop(columns=columns_to_drop)

    return metadata.to_numpy()


def process_data_best_features(metadata_path, base_path, id_range, use_completed_seg=False):
    """
    Extracts a selected subset of 14 key features from cardiac MRI segmentations 
    based on their relevance for classification, focusing on volumes, ratios, 
    circularity, and myocardial thickness metrics for a more compact feature set.

    Obs: the selected features were based on the feature importance from a Random
    Forest model that I executed one time, but then I notice that this set changes.
    To improve the model, it would be better to pass the 14 features as a parameter.
    """

    # Read metadata
    metadata = pd.read_csv(metadata_path)

    # Initialize selected features with 0.0
    for col in [
        'EF_LV', 'MYO_LV_EDV_Ratio', 'RV_LV_EDV_Ratio', 'LV_ESV', 'EF_RV', 'RV_ESV',
        'MYO_Circum_Mean', 'MYO_Thickness_Max', 'MYO_Circ_Mean', 'MYO_Circum_Max',
        'LV_EDV', 'LV_EDV_BSA', 'RV_EDV_BSA', 'RVC_LVC_Apex_Ratio'
    ]:
        metadata[col] = 0.0

    for idx, subject_num in enumerate(id_range):
        subject_id = f"{subject_num:03d}"
        subject_path = os.path.join(base_path, subject_id)

        if "Train" in base_path:
            suffix = "_seg.nii"
        else:
            if use_completed_seg:
                suffix = "_complete_seg.nii"
            else:
                suffix = "_simple_seg.nii"

        try:
            seg_ed = nib.load(os.path.join(subject_path, f"{subject_id}_ED{suffix}")).get_fdata()
            seg_es = nib.load(os.path.join(subject_path, f"{subject_id}_ES{suffix}")).get_fdata()
        except FileNotFoundError:
            print(f"Missing segmentation for subject {subject_id}, skipping.")
            continue

        lv_edv = volume(seg_ed, 3)
        lv_esv = volume(seg_es, 3)
        rv_edv = volume(seg_ed, 1)
        rv_esv = volume(seg_es, 1)
        myo_ed = volume(seg_ed, 2)

        metadata.at[idx, 'LV_EDV'] = lv_edv
        metadata.at[idx, 'LV_ESV'] = lv_esv
        metadata.at[idx, 'EF_LV'] = ejection_fraction(lv_edv, lv_esv)
        metadata.at[idx, 'RV_EDV_BSA'] = rv_edv / (np.sqrt((metadata.at[idx, 'Weight'] * metadata.at[idx, 'Height']) / 3600)) if lv_edv > 0 else 0
        metadata.at[idx, 'RV_ESV'] = rv_esv
        metadata.at[idx, 'EF_RV'] = ejection_fraction(rv_edv, rv_esv)
        metadata.at[idx, 'MYO_LV_EDV_Ratio'] = myo_ed / lv_edv if lv_edv > 0 else 0
        metadata.at[idx, 'RV_LV_EDV_Ratio'] = rv_edv / lv_edv if lv_edv > 0 else 0

        myo_circum_mean, myo_circum_max = compute_circumference_stats(seg_ed, 3)
        metadata.at[idx, 'MYO_Circum_Mean'] = myo_circum_mean
        metadata.at[idx, 'MYO_Circum_Max'] = myo_circum_max

        myo_circ_mean, _ = circularity(seg_ed, 2)
        metadata.at[idx, 'MYO_Circ_Mean'] = myo_circ_mean

        _, max_t, = myocardium_thickness_stats(seg_ed)[2:4]
        metadata.at[idx, 'MYO_Thickness_Max'] = max_t

        metadata.at[idx, 'LV_EDV_BSA'] = lv_edv / (np.sqrt((metadata.at[idx, 'Weight'] * metadata.at[idx, 'Height']) / 3600)) if lv_edv > 0 else 0

        _, _, apex_ratio = rvc_size_at_apical_lvm_slice(seg_ed)
        metadata.at[idx, 'RVC_LVC_Apex_Ratio'] = apex_ratio

    if 'Id' in metadata.columns:
        metadata = metadata.drop(columns=['Id'])

    return metadata.to_numpy()