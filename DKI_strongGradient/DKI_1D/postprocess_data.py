import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def plot_param_maps(D_k_map, K_map, zslice, timestamp, control_type):
    """
    Plots the parameter maps for D_k, K, f_VASC, and R.
    Saves the plots as PNG files and the parameter maps as NIfTI files.
    Parameters:
    - D_k_map: 3D numpy array for D_k parameter map.
    - K_map: 3D numpy array for K parameter map.
    - r_map: 3D numpy array for R parameter map.
    - f_vasc_map: 3D numpy array for f_VASC parameter map.
    - cell_map: 3D numpy array for cell parameter map.
    - zslice: Integer index for the z-slice to visualize.
    - th_gradient_strength: String indicating the gradient strength.
    - timestamp: String for the timestamp to use in file names.
    - control_type: String indicating the type of control used.
    """
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax = ax.flatten()

    D_k_plot = ax[0].imshow(D_k_map[:, :, zslice], cmap='jet') 
    ax[0].set_xlim(50, 110)
    ax[0].set_ylim(105, 45)
    ax[0].set_title('D_k')
    ax[0].axis('off')
    plt.colorbar(D_k_plot, ax=ax[0], fraction=0.046, pad=0.04)
    # D_k_plot.set_clim(0, 1)

    K_plot = ax[1].imshow(K_map[:, :, zslice], cmap='jet')
    plt.colorbar(K_plot, ax=ax[1], fraction=0.046, pad=0.04)
    # K_plot.set_clim(0, 1)
    ax[1].set_xlim(50, 110)
    ax[1].set_ylim(105, 45)
    ax[1].set_title('K')
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

    fig.savefig(timestamp + f'/ssDKI_1D_param_maps_{control_type}_{timestamp}.png', dpi=300, bbox_inches='tight')

    Dksave = nib.Nifti1Image(D_k_map, np.eye(4))
    nib.save(Dksave, timestamp + f'/ssDKI_1D_D_k_{control_type}_{timestamp}.nii.gz')

    Ksave = nib.Nifti1Image(K_map, np.eye(4))
    nib.save(Ksave, timestamp + f'/ssDKI_1D_K_{control_type}_{timestamp}.nii.gz')

def generate_param_maps(D_k_pred, K_pred, image_mask, timestamp, zslice, control_type):
    """
    Generates parameter voxel array from the predicted values of f_IC, f_EES, and R by normalizing and constraining them.
    Generates parameter maps by reshaping the flattened voxel arrays back to the original image dimensions using the image mask.
    Parameters:
    - D_k_pred: Flattened array of predicted D_k values.
    - K_pred: Flattened array of predicted K values.
    - image_mask: 3D numpy array representing the mask of the image.
    - timestamp: String for the timestamp to use in file names.
    - control_type: String indicating the type of control used.
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

    plot_param_maps(D_k_map, K_map, zslice, timestamp, control_type)

    return D_k_map, K_map