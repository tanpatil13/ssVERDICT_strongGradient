import numpy as np
import nibabel as nib
import matplotlib, matplotlib.pyplot as plt
from commons.preprocess_data import get_matched_files

def plot_param_maps(f_ic_map, f_ees_map, d_ees_map, r_map, f_vasc_map, cell_map, zslice, th_gradient_strength, timestamp, target_dir, control_type, control_id=None, cmap='jet'):
    """
    Plots the parameter maps for f_IC, f_EES, f_VASC, and R.
    Saves the plots as PNG files and the parameter maps as NIfTI files.
    Parameters:
    - f_ic_map: 3D numpy array for f_IC parameter map.
    - f_ees_map: 3D numpy array for f_EES parameter map.
    - r_map: 3D numpy array for R parameter map.
    - f_vasc_map: 3D numpy array for f_VASC parameter map.
    - cell_map: 3D numpy array for cell parameter map.
    - zslice: Integer index for the z-slice to visualize.
    - th_gradient_strength: String indicating the gradient strength.
    - timestamp: String for the timestamp to use in file names.
    - target_dir: Directory where the output files will be saved.
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

    f_ic_plot = ax[0].imshow(f_ic_map[:, :, zslice], cmap=cmap)
    plt.colorbar(f_ic_plot, ax=ax[0], fraction=0.046, pad=0.04)
    # f_ic_plot.set_clim(0, 1) 
    ax[0].set_xlim(x_limit[0], x_limit[1])
    ax[0].set_ylim(y_limit[0], y_limit[1])
    ax[0].set_title('f_IC')
    ax[0].axis('off')

    f_ees_plot = ax[1].imshow(f_ees_map[:, :, zslice], cmap=cmap)
    plt.colorbar(f_ees_plot, ax=ax[1], fraction=0.046, pad=0.04)
    # f_ees_plot.set_clim(0, 1)
    ax[1].set_xlim(x_limit[0], x_limit[1])
    ax[1].set_ylim(y_limit[0], y_limit[1])
    ax[1].set_title('f_EES')
    ax[1].axis('off')

    f_vasc_plot = ax[2].imshow(f_vasc_map[:, :, zslice], cmap=cmap)
    plt.colorbar(f_vasc_plot, ax=ax[2], fraction=0.046, pad=0.04)
    # f_vasc_plot.set_clim(0, 0.2)
    ax[2].set_xlim(x_limit[0], x_limit[1])
    ax[2].set_ylim(y_limit[0], y_limit[1])
    ax[2].set_title('f_VASC')
    ax[2].axis('off')

    r_plot = ax[3].imshow(r_map[:, :, zslice], cmap=cmap)
    plt.colorbar(r_plot, ax=ax[3], fraction=0.046, pad=0.04)
    # r_plot.set_clim(0, 15)
    ax[3].set_xlim(x_limit[0], x_limit[1])
    ax[3].set_ylim(y_limit[0], y_limit[1])
    ax[3].set_title('R')
    ax[3].axis('off')

    d_ees = ax[4].imshow(d_ees_map[:, :, zslice], cmap=cmap)
    plt.colorbar(d_ees, ax=ax[4], fraction=0.046, pad=0.04)
    ax[4].set_xlim(x_limit[0], x_limit[1])
    ax[4].set_ylim(y_limit[0], y_limit[1])
    ax[4].set_title('d_EES')
    ax[4].axis('off')

    ax[5].axis('off')

    plt.tight_layout()
    plt.show()

    fig.savefig(target_dir + '/model_output_directory/' + timestamp + f'/ssVERDICT_NN_param_maps_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.png', dpi=300, bbox_inches='tight')

    ficsave = nib.Nifti1Image(f_ic_map, np.eye(4))
    nib.save(ficsave, target_dir + '/model_output_directory/' + timestamp + f'/ssVERDICT_NN_f_ic_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.nii.gz')

    feessave = nib.Nifti1Image(f_ees_map, np.eye(4))
    nib.save(feessave, target_dir + '/model_output_directory/' + timestamp + f'/ssVERDICT_NN_f_ees_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.nii.gz')

    fvascsave = nib.Nifti1Image(f_vasc_map, np.eye(4))
    nib.save(fvascsave, target_dir + '/model_output_directory/' + timestamp + f'/ssVERDICT_NN_f_vasc_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.nii.gz')

    deessave = nib.Nifti1Image(d_ees_map, np.eye(4))
    nib.save(deessave, target_dir + '/model_output_directory/' + timestamp + f'/ssVERDICT_NN_d_ees_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.nii.gz')

    rsave = nib.Nifti1Image(r_map, np.eye(4))
    nib.save(rsave, target_dir + '/model_output_directory/' + timestamp + f'/ssVERDICT_NN_r_{control_type}_{control_id}_{th_gradient_strength}_{timestamp}.nii.gz')

def generate_param_maps(f_ic_pred, f_ees_pred, d_ees_pred, r_pred, image_mask, th_gradient_strength, timestamp, target_dir, zslice, control_type, control_id=None, prostate_mask_dir=None, prostate_mask_file_pattern=None):
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
    - target_dir: Directory where the output files will be saved.
    - zslice: Integer index for the z-slice to visualize.
    - control_type: String indicating the type of control used.
    - control_id: Optional string indicating the control ID for specific patient.
    - prostate_mask_dir: Directory containing the prostate mask file.
    - prostate_mask_file_pattern: Regex pattern for the prostate mask file.
    """

    f_ic = np.array(f_ic_pred)
    f_ees = np.array(f_ees_pred)
    d_ees = np.array(d_ees_pred)
    r = np.array(r_pred)

    f_vasc = 1 - f_ic - f_ees
    f_vasc = f_vasc/(f_ic + f_ees + f_vasc)
    A = f_vasc
    normA = A - min(A)
    f_vasc = 0.2 * (normA/max(normA))       # constraining fvasc
    f_ees = f_ees/(f_ic + f_ees + f_vasc)
    f_ic = f_ic/(f_ic + f_ees + f_vasc)
    cell = f_ic/r**3

    mask_vox = image_mask.flatten()

    f_ic_vox = np.zeros_like(mask_vox)
    f_ic_vox[mask_vox == 1] = np.squeeze(f_ic)
    f_ic_map = f_ic_vox.reshape(image_mask.shape)

    f_ees_vox = np.zeros_like(mask_vox)
    f_ees_vox[mask_vox == 1] = np.squeeze(f_ees)
    f_ees_map = f_ees_vox.reshape(image_mask.shape)

    f_vasc_vox = np.zeros_like(mask_vox)
    f_vasc_vox[mask_vox == 1] = np.squeeze(f_vasc)
    f_vasc_map = f_vasc_vox.reshape(image_mask.shape)

    d_ees_vox = np.zeros_like(mask_vox)
    d_ees_vox[mask_vox == 1] = np.squeeze(d_ees)
    d_ees_map = d_ees_vox.reshape(image_mask.shape)

    r_vox = np.zeros_like(mask_vox)
    r_vox[mask_vox == 1] = np.squeeze(r)
    r_map = r_vox.reshape(image_mask.shape)

    cell_vox = np.zeros_like(mask_vox)
    cell_vox[mask_vox == 1] = np.squeeze(cell)
    cell_map = cell_vox.reshape(image_mask.shape)

    cmap = matplotlib.colormaps.get_cmap('jet').copy()
    if prostate_mask_file_pattern and prostate_mask_dir:
        prostate_mask_file = get_matched_files(prostate_mask_dir, prostate_mask_file_pattern)
        prostate_mask = nib.load(prostate_mask_file).get_fdata()
        prostate_mask = np.repeat(prostate_mask[:, :, np.newaxis], image_mask.shape[2], axis=2)

        f_ic_map = np.where(prostate_mask, f_ic_map, 0)
        f_ic_map = np.ma.masked_where(f_ic_map == 0, f_ic_map)

        f_ees_map = np.where(prostate_mask, f_ees_map, 0)
        f_ees_map = np.ma.masked_where(f_ees_map == 0, f_ees_map)

        f_vasc_map = np.where(prostate_mask, f_vasc_map, 0)
        f_vasc_map = np.ma.masked_where(f_vasc_map == 0, f_vasc_map)

        d_ees_map = np.where(prostate_mask, d_ees_map, 0)
        d_ees_map = np.ma.masked_where(d_ees_map == 0, d_ees_map)

        r_map = np.where(prostate_mask, r_map, 0)
        r_map = np.ma.masked_where(r_map == 0, r_map)

        cell_map = np.where(prostate_mask, cell_map, 0)
        cell_map = np.ma.masked_where(cell_map == 0, cell_map)

        cmap.set_bad(color='white')

    plot_param_maps(f_ic_map, f_ees_map, d_ees_map, r_map, f_vasc_map, cell_map, zslice, th_gradient_strength, timestamp, target_dir, control_type, control_id, cmap)

    return f_ic_map, f_ees_map, f_vasc_map, d_ees_map, r_map, cell_map