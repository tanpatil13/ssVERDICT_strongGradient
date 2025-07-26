import numpy as np
import nibabel as nib
import matplotlib, matplotlib.pyplot as plt
from commons.preprocess_data import get_matched_files

def plot_param_maps(MD_map, FA_map, MK_map, AK_map, RK_map, zslice, th_gradient_strength, timestamp, control_type,control_id=None, cmap='jet'):
    """
    Plots the parameter maps for D_k, K, f_VASC, and R.
    Saves the plots as PNG files and the parameter maps as NIfTI files.
    Parameters:
    - MD_map: 3D numpy array for Mean Diffusivity parameter map.
    - FA_map: 3D numpy array for Fractional Anisotropy parameter map.
    - MK_map: 3D numpy array for Mean Kurtosis parameter map.
    - AK_map: 3D numpy array for Axial Kurtosis parameter map.
    - RK_map: 3D numpy array for Radial Kurtosis parameter map.
    - th_gradient_strength: String indicating the gradient strength.
    - zslice: Integer index for the z-slice to visualize.
    - timestamp: String for the timestamp to use in file names.
    - control_type: String indicating the type of control used.
    - control_id: Optional string indicating the control ID for specific patient.
    - cmap: Colormap to use for the plots.
    """
    
    fig, ax = plt.subplots(3, 2, figsize=(20, 15))
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

    MD_plot = ax[0].imshow(MD_map[:, :, zslice], cmap=cmap)
    plt.colorbar(MD_plot, ax=ax[0], fraction=0.046, pad=0.04)
    # MD_plot.set_clim(0, 2)
    ax[0].set_xlim(x_limit[0], x_limit[1])
    ax[0].set_ylim(y_limit[0], y_limit[1])
    ax[0].set_title('Mean Diffusivity (MD)')
    ax[0].axis('off')

    FA_plot = ax[1].imshow(FA_map[:, :, zslice], cmap=cmap)
    plt.colorbar(FA_plot, ax=ax[1], fraction=0.046, pad=0.04)
    # FA_plot.set_clim(0, 1)
    ax[1].set_xlim(x_limit[0], x_limit[1])
    ax[1].set_ylim(y_limit[0], y_limit[1])
    ax[1].set_title('Fractional Anisotropy (FA)')
    ax[1].axis('off')

    MK_plot = ax[2].imshow(MK_map[:, :, zslice], cmap=cmap)
    plt.colorbar(MK_plot, ax=ax[2], fraction=0.046, pad=0.04)
    # MK_plot.set_clim(0, 5)
    ax[2].set_xlim(x_limit[0], x_limit[1])
    ax[2].set_ylim(y_limit[0], y_limit[1])
    ax[2].set_title('Mean Kurtosis (MK)')
    ax[2].axis('off')

    AK_plot = ax[3].imshow(AK_map[:, :, zslice], cmap=cmap)
    plt.colorbar(AK_plot, ax=ax[3], fraction=0.046, pad=0.04)
    # AK_plot.set_clim(0, 5)
    ax[3].set_xlim(x_limit[0], x_limit[1])
    ax[3].set_ylim(y_limit[0], y_limit[1])
    ax[3].set_title('Axial Kurtosis (AK)')
    ax[3].axis('off')

    RK_plot = ax[4].imshow(RK_map[:, :, zslice], cmap=cmap)
    plt.colorbar(RK_plot, ax=ax[4], fraction=0.046, pad=0.04)
    RK_plot.set_clim(0, 4)
    ax[4].set_xlim(x_limit[0], x_limit[1])
    ax[4].set_ylim(y_limit[0], y_limit[1])
    ax[4].set_title('Radial Kurtosis (RK)')
    ax[4].axis('off')

    ax[5].axis('off')

    plt.tight_layout()
    plt.show()

    fig.savefig(timestamp + f'/ssDKI_3D_param_maps_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.png', dpi=300, bbox_inches='tight')

    MDsave = nib.Nifti1Image(MD_map, np.eye(4))
    nib.save(MDsave, timestamp + f'/ssDKI_3D_MD_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.nii.gz')

    FAsave = nib.Nifti1Image(FA_map, np.eye(4))
    nib.save(FAsave, timestamp + f'/ssDKI_3D_FA_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.nii.gz')

    MKsave = nib.Nifti1Image(MK_map, np.eye(4))
    nib.save(MKsave, timestamp + f'/ssDKI_3D_MK_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.nii.gz')

    AKsave = nib.Nifti1Image(AK_map, np.eye(4))
    nib.save(AKsave, timestamp + f'/ssDKI_3D_AK_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.nii.gz')

    RKsave = nib.Nifti1Image(RK_map, np.eye(4))
    nib.save(RKsave, timestamp + f'/ssDKI_3D_RK_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.nii.gz')


def generate_param_maps(MD_pred, FA_pred, MK_pred, AK_pred, RK_pred, image_mask, th_gradient_strength, timestamp, zslice, control_type, control_id=None, prostate_mask_dir=None, prostate_mask_file_pattern=None):
    """
    Generates parameter voxel array from the predicted values of f_IC, f_EES, and R by normalizing and constraining them.
    Generates parameter maps by reshaping the flattened voxel arrays back to the original image dimensions using the image mask.
    Parameters:
    - f_ic_pred: Flattened array of predicted f_IC values.
    - f_ees_pred: Flattened array of predicted f_EES values.
    - r_pred: Flattened array of predicted R values.
    - image_mask: 3D numpy array representing the mask of the image.
    - th_gradient_strength: String indicating the gradient strength.
    - timestamp: String for the timestamp to use in file names.
    - control_type: String indicating the type of control used.
    - control_id: Optional string indicating the control ID for specific patient.
    - prostate_mask_dir: Directory containing the prostate mask file.
    - prostate_mask_file_pattern: Regex pattern for the prostate mask file.
    """

    MD = np.array(MD_pred)
    FA = np.array(FA_pred)
    MK = np.array(MK_pred)
    AK = np.array(AK_pred)
    RK = np.array(RK_pred)

    mask_vox = image_mask.flatten()

    MD_vox = np.zeros_like(mask_vox)
    MD_vox[mask_vox == 1] = np.squeeze(MD)
    MD_map = MD_vox.reshape(image_mask.shape)

    FA_vox = np.zeros_like(mask_vox)
    FA_vox[mask_vox == 1] = np.squeeze(FA)
    FA_map = FA_vox.reshape(image_mask.shape)

    MK_vox = np.zeros_like(mask_vox)
    MK_vox[mask_vox == 1] = np.squeeze(MK)
    MK_map = MK_vox.reshape(image_mask.shape)

    AK_vox = np.zeros_like(mask_vox)
    AK_vox[mask_vox == 1] = np.squeeze(AK)
    AK_map = AK_vox.reshape(image_mask.shape)

    RK_vox = np.zeros_like(mask_vox)
    RK_vox[mask_vox == 1] = np.squeeze(RK)
    RK_map = RK_vox.reshape(image_mask.shape)

    cmap = matplotlib.colormaps.get_cmap('jet').copy()
    if prostate_mask_file_pattern and prostate_mask_dir:
        prostate_mask_file = get_matched_files(prostate_mask_dir, prostate_mask_file_pattern)
        prostate_mask = nib.load(prostate_mask_file).get_fdata()

        MD_map = np.where(prostate_mask, MD_map, 0)
        MD_map = np.ma.masked_where(MD_map == 0, MD_map)

        FA_map = np.where(prostate_mask, FA_map, 0)
        FA_map = np.ma.masked_where(FA_map == 0, FA_map)

        MK_map = np.where(prostate_mask, MK_map, 0)
        MK_map = np.ma.masked_where(MK_map == 0, MK_map)

        AK_map = np.where(prostate_mask, AK_map, 0)
        AK_map = np.ma.masked_where(AK_map == 0, AK_map)

        RK_map = np.where(prostate_mask, RK_map, 0)
        RK_map = np.ma.masked_where(RK_map == 0, RK_map)

        cmap.set_bad(color='white')

    plot_param_maps(MD_map, FA_map, MK_map, AK_map, RK_map, zslice, th_gradient_strength, timestamp, control_type, control_id, cmap)

    return MD_map, FA_map, MK_map, AK_map, RK_map

def computeEigenDecomposition(D):
    D = D.squeeze()
    D_mat =  np.array([[D[0], D[3], D[4]],
                       [D[3], D[1], D[5]],
                       [D[4], D[5], D[2]]])
    eigenvalues, eigenvectors = np.linalg.eig(D_mat)
    
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    return eigenvalues, eigenvectors, sorted_idx

def compute_MD(D):
    Dxx, Dyy, Dzz = D[:, 0], D[:, 1], D[:, 2]
    return (Dxx + Dyy + Dzz) / 3.0

def compute_FA(D):
    FA = np.zeros(D.shape[0])
    for i in range(D.shape[0]):
        trace_D = D[i, 0] + D[i, 1] + D[i, 2]
        eigenvalues, _, _ = computeEigenDecomposition(D[i, :])
        numerator = 3 * sum((eigenvalues - trace_D / 3) ** 2)
        denominator = 2 * sum(eigenvalues ** 2)
        FA[i] = np.sqrt(numerator / (denominator + 1e-10))  # avoid division by 0
    return FA

def compute_KurtosisMetrics(D, W):
    MK = (W[:, 0] + W[:, 1] + W[:, 2] + 2 * (W[:, 9] + W[:, 10] + W[:, 11]))/5

    MD = np.zeros(D.shape[0])
    AK = np.zeros(W.shape[0])
    RK = np.zeros(W.shape[0])
    for i in range(D.shape[0]):
        eigenvalues, eigenvectors, sorted_idx = computeEigenDecomposition(D[i, :])

        MD[i] = (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]) / 3.0

        D_par = eigenvalues[0]
        D_perp = (eigenvalues[1] + eigenvalues[2]) / 2.0

        W_par = W[i, sorted_idx[0]]
        AK[i] = (W_par * MD[i]**2) / (D_par**2 + 1e-10)

        W_perp =(3.0/8) * (W[i, sorted_idx[1]] + W[i, sorted_idx[2]] + 2 * W[i, 14 - 3 - sorted_idx[0]])
        RK[i] = (W_perp * MD[i]**2) / (D_perp**2 + 1e-10)
        
    return MD, MK, AK, RK