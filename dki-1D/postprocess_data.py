import numpy as np
import nibabel as nib
import matplotlib, matplotlib.pyplot as plt
from commons.preprocess_data import get_matched_files

def plot_param_maps(D_k_map, K_map, zslice, th_gradient_strength, timestamp, target_dir, control_type, control_id=None, cmap='jet'):
    """
    Plots the parameter maps for D_k, K, f_VASC, and R.
    Saves the plots as PNG files and the parameter maps as NIfTI files.
    Parameters:
    - D_k_map: 3D numpy array for D_k parameter map.
    - K_map: 3D numpy array for K parameter map.
    - zslice: Integer index for the z-slice to visualize.
    - th_gradient_strength: String indicating the gradient strength.
    - timestamp: String for the timestamp to use in file names.
    - target_dir: Directory where the output files will be saved.
    - control_type: String indicating the type of control used.
    - control_id: Optional string indicating the control ID for specific patient.
    - cmap: Colormap to use for the plots.
    """
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax = ax.flatten()

    x_limit = (50, 110)
    y_limit = (105, 45)

    if control_id == "1":
        x_limit = (60, 120)
        y_limit = (110, 50)
    elif control_id == "3":
        x_limit = (50, 110)
        y_limit = (120, 60)
    elif control_id == "5":
        x_limit = (60, 120)
        y_limit = (110, 50)

    D_k_plot = ax[0].imshow(D_k_map[:, :, zslice], cmap=cmap)
    ax[0].set_xlim(x_limit[0], x_limit[1])
    ax[0].set_ylim(y_limit[0], y_limit[1])
    ax[0].set_title('D_k')
    ax[0].axis('off')
    plt.colorbar(D_k_plot, ax=ax[0], fraction=0.046, pad=0.04)
    # D_k_plot.set_clim(0, 1)

    K_plot = ax[1].imshow(K_map[:, :, zslice], cmap=cmap)
    plt.colorbar(K_plot, ax=ax[1], fraction=0.046, pad=0.04)
    # K_plot.set_clim(0, 1)
    ax[1].set_xlim(x_limit[0], x_limit[1])
    ax[1].set_ylim(y_limit[0], y_limit[1])
    ax[1].set_title('K')
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

    fig.savefig(target_dir + '/model_output_directory/' + timestamp + f'/ssDKI_1D_param_maps_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.png', dpi=300, bbox_inches='tight')

    Dksave = nib.Nifti1Image(D_k_map, np.eye(4))
    nib.save(Dksave, target_dir + '/model_output_directory/' + timestamp + f'/ssDKI_1D_D_k_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.nii.gz')

    Ksave = nib.Nifti1Image(K_map, np.eye(4))
    nib.save(Ksave, target_dir + '/model_output_directory/' + timestamp + f'/ssDKI_1D_K_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.nii.gz')

def generate_param_maps(D_k_pred, K_pred, image_mask, th_gradient_strength, timestamp, target_dir, zslice, control_type, control_id=None, prostate_mask_dir=None, prostate_mask_file_pattern=None):
    """
    Generates parameter voxel array from the predicted values of f_IC, f_EES, and R by normalizing and constraining them.
    Generates parameter maps by reshaping the flattened voxel arrays back to the original image dimensions using the image mask.
    Parameters:
    - D_k_pred: Flattened array of predicted D_k values.
    - K_pred: Flattened array of predicted K values.
    - image_mask: 3D numpy array representing the mask of the image.
    - th_gradient_strength: String indicating the gradient strength.
    - timestamp: String for the timestamp to use in file names.
    - target_dir: Directory where the output files will be saved.
    - zslice: Integer index for the z-slice to visualize.
    - control_type: String indicating the type of control used.
    - control_id: Optional string indicating the control ID for specific patient.
    - prostate_mask_dir: Directory containing the prostate mask file.
    - prostate_mask_file_pattern: Regex pattern for the prostate mask file.
    """

    D_k = np.array(D_k_pred)
    K = np.array(K_pred)

    mask_vox = image_mask.flatten()

    D_k_vox = np.zeros_like(mask_vox)
    D_k_vox[mask_vox == 1] = np.squeeze(D_k)
    D_k_map = D_k_vox.reshape(image_mask.shape)

    K_vox = np.zeros_like(mask_vox)
    K_vox[mask_vox == 1] = np.squeeze(K)
    K_map = K_vox.reshape(image_mask.shape)

    cmap = matplotlib.colormaps.get_cmap('jet').copy()
    if prostate_mask_file_pattern and prostate_mask_dir:
        prostate_mask_file = get_matched_files(prostate_mask_dir, prostate_mask_file_pattern)
        prostate_mask = nib.load(prostate_mask_file).get_fdata()
        prostate_mask = np.repeat(prostate_mask[:, :, np.newaxis], image_mask.shape[2], axis=2)

        D_k_map = np.where(prostate_mask, D_k_map, 0)
        D_k_map = np.ma.masked_where(D_k_map == 0, D_k_map)

        K_map = np.where(prostate_mask, K_map, 0)
        K_map = np.ma.masked_where(K_map == 0, K_map)

        cmap.set_bad(color='white')

    plot_param_maps(D_k_map, K_map, zslice, th_gradient_strength, timestamp, target_dir, control_type, control_id, cmap)

    return D_k_map, K_map