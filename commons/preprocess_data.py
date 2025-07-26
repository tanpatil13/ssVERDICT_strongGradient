import numpy as np
import nibabel as nib
import os


def get_matched_files(data_dir, pattern):
    """
    Returns the file in the directory that matches the given regex pattern.
    """
    for f in os.listdir(data_dir):
        if pattern.match(f):
            return os.path.join(data_dir, f)
    return None


def load_nifti_image(path):
    """
    Loads the nifti image from the given path and returns the image data.
    """
    imgnii = nib.load(path)
    imgdata = np.rot90(imgnii.get_fdata())
    return imgdata


def calculate_actual_bvals(x_bvec, y_bvec, z_bvec):
    """
    Calculate the actual b-values from the gradient matrices.
    b-values are calculated as the square root of the sum of squares of the gradient components for each voxel.
    """
    assert x_bvec.shape == y_bvec.shape == z_bvec.shape, "The gradient matrix must have the same shape"

    num_vox_x, num_vox_y, num_slices, num_vols = x_bvec.shape
    ac_bvals = np.zeros_like(x_bvec)

    for vol in range(num_vols):
        for slice in range(num_slices):
            x = x_bvec[:, :, slice, vol]
            y = y_bvec[:, :, slice, vol]
            z = z_bvec[:, :, slice, vol]

            ac_bvals[:, :, slice, vol] = np.sqrt(x**2 + y**2 + z**2)

    return ac_bvals


def get_image_mask(data):
    """
    Returns a mask of the image data where the values are greater than 0.
    """
    mask = np.zeros_like(data[:, :, :, 0])
    mask[data[:, :, :, 0] != 0] = 1
    return mask


def flatten_mask_data(data, mask=None):
    """
    Flattens the 4D (num_vox_x, num_vox_y, num_slices, num_vols) data array into a 2D (num_vox_x * num_vox_y * num_slices, num_vols)
    array where each column corresponds to a volume.
    If mask is True, it filters out zero values from the flattened data.
    """
    num_vox_x, num_vox_y, num_slices, num_vols = data.shape
    flattened_data = np.zeros((num_vox_x * num_vox_y * num_slices, num_vols))
    flattened_masked_data = []
    
    for vol in range(num_vols):
        flattened_data[:, vol] = data[:, :, :, vol].flatten()
        if mask is not None:
            flattened_mask = mask.flatten()
            flattened_masked_data.append(flattened_data[:, vol][flattened_mask == 1])  # Only keep non-zero values
        else:
            flattened_masked_data.append(flattened_data[:, vol])

    return np.array(flattened_masked_data).T


def calculate_closest_idx(th_data, ac_data):
    """
    Calculate the index of the closest value in ac_data (actual data) for each value in th_data (theoritical data).
    For e.g., for each b-value in ac_data, find the index of the closest b-value in th_data to obtain a consistent mapping.
    """
    closest_idx = np.zeros(ac_data.shape, dtype=int)
    for i in range(ac_data.shape[0]):
        closest_idx[i] = np.argmin(np.abs(th_data - ac_data[i]))
    return closest_idx


def preprocess_images(split_data_dir, grad_dataset_dir,
                      image_file_pattern, x_bvec_file_pattern, y_bvec_file_pattern, z_bvec_file_pattern,
                      th_bvals):
    """
    Preprocesses the images in the given split data directory.
    It loads the images, calculates the actual b-values, normalizes the images, flattens them,
    averages the image signals over the same b-values and returns the processed data for model input.
    """

    preprocessed_image_data = []

    for subject_type, subject_ids in split_data_dir.items():
        for subject_id in subject_ids:

            nii_gz_path = os.path.join(grad_dataset_dir, subject_type, subject_id)

            image_file = get_matched_files(nii_gz_path, image_file_pattern)
            x_bvec_file = get_matched_files(nii_gz_path, x_bvec_file_pattern)
            y_bvec_file = get_matched_files(nii_gz_path, y_bvec_file_pattern)
            z_bvec_file = get_matched_files(nii_gz_path, z_bvec_file_pattern)

            if not all([image_file, x_bvec_file, y_bvec_file, z_bvec_file]):
                print(f"Missing files for {subject_type} {subject_id}")
                continue

            image_data = load_nifti_image(image_file)
            x_bvec_data = load_nifti_image(x_bvec_file)
            y_bvec_data = load_nifti_image(y_bvec_file)
            z_bvec_data = load_nifti_image(z_bvec_file)

            image_dim = image_data.shape
            image_mask = get_image_mask(image_data)

            ac_bvals = calculate_actual_bvals(x_bvec_data, y_bvec_data, z_bvec_data)
            flattened_ac_bvals = flatten_mask_data(ac_bvals)

            b0_indices = []
            for i in range(ac_bvals.shape[2]):
                if(ac_bvals[:, :, :, i].all() == 0):
                    b0_indices.append(i)

            image_data_flattened_masked = flatten_mask_data(image_data, mask=image_mask)

            mean_ac_bvals = np.mean(flattened_ac_bvals, axis=0)
            ac_th_bvals_map = calculate_closest_idx(th_bvals, mean_ac_bvals)
            avg_image_data = np.zeros((image_data_flattened_masked.shape[0], len(th_bvals)))
            for i in range(len(th_bvals)):
                avg_image_data[:, i] = np.mean(image_data_flattened_masked[:, ac_th_bvals_map == i], axis=1)
            
            avg_image_data = np.clip(avg_image_data, 0, None)

            norm_avg_image_data = np.zeros_like(avg_image_data)

            for i in range(avg_image_data.shape[1]):

                norm_avg_image_data[:, i] = avg_image_data[:, i] / (avg_image_data[:, 0] + 1e-6)  # Normalize by the first b-value (b0)

            norm_avg_image_data = np.clip(norm_avg_image_data, None, 1-(1e-6))
            preprocessed_image_data.extend(norm_avg_image_data)

    preprocessed_image_data = np.array(preprocessed_image_data)

    return ac_bvals, preprocessed_image_data, image_dim, image_mask