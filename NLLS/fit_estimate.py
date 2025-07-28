import os
import numpy as np
from lmfit import Parameters, Minimizer
from model import verdictResiduals
from commons.preprocess_data import preprocess_images
from postprocess_data import generate_param_maps

def get_prostate_mask_path(grad_dataset_dir, prostate_mask_dir):
    for subject_type, subject_ids in prostate_mask_dir.items():
        for subject_id in subject_ids:
            return os.path.join(grad_dataset_dir, subject_type, subject_id)

def perform_fit(grad_dataset_dir, patient_data_dir,
                image_file_pattern, x_bvec_file_pattern, y_bvec_file_pattern, z_bvec_file_pattern, 
                Delta, delta, gradient_strength, prostate_mask_file_pattern,
                th_bvals, th_gradient_strength, timestamp, target_dir):
    """
    Preprocess the images, train the NLLS fitted VERDICT model, and test it.
    Generate the estimated parameter maps for f_ic, f_ees, f_vasc, and r for the specified patients.
    Args:
        grad_dataset_dir: Directory containing the gradient dataset.
        patient_test_data_dir: Dictionary containing patient test data directories.
        image_file_pattern: Regex pattern for the specified gradient images.
        x_bvec_file_pattern: Regex pattern for the specified gradient x b-vectors.
        y_bvec_file_pattern: Regex pattern for the specified gradient y b-vectors.
        z_bvec_file_pattern: Regex pattern for the specified gradient z b-vectors.
        Delta, delta, gradient_strength: Delta, delta, and gradient strength for the specified gradient.
        prostate_mask_file_pattern: Regex pattern for the prostate mask file.
        th_bvals: List of b-values in ms/µm^2.
        th_gradient_strength: Theoretical maximum gradient strength in mT/m.
        timestamp: Timestamp for saving model checkpoints and plots.
        target_dir: Directory to save the outputs.
    """
    
    _,  patient_preprocessed_image_data, _, patient_image_mask = preprocess_images(patient_data_dir,
                                                                grad_dataset_dir, image_file_pattern, x_bvec_file_pattern, 
                                                                y_bvec_file_pattern, z_bvec_file_pattern, th_bvals)

    num_itr = 1
    img_dim = patient_preprocessed_image_data.shape
    f_ic_pred = np.zeros(img_dim[0])
    f_ees_pred = np.zeros(img_dim[0])
    r_pred = np.zeros(img_dim[0])
    resnorm = np.zeros(img_dim[0])
    is_success = np.zeros(img_dim[0], dtype=bool)

    Delta = np.array(Delta)
    delta = np.array(delta)
    gradient_strength = np.array(gradient_strength)

    th_bvals_scaled = [th_bvals[i]/1000 for i in range(len(th_bvals))]  # Convert to ms/µm^2

    for i in range(img_dim[0]):
        params = Parameters()
        params.add('f_ic', value=0.5, min=0.001, max=0.999)
        params.add('f_ees', value=0.5, min=0.001, max=0.999)
        params.add('r', value=7.5, min=0.001, max=14.999)

        for k in range(num_itr):
            fitter = Minimizer(verdictResiduals, params, fcn_args=(patient_preprocessed_image_data[i, :], th_bvals_scaled, Delta, delta, gradient_strength))
            results = fitter.minimize(method='leastsq')
            f_ic_pred[i] = results.params['f_ic'].value
            f_ees_pred[i] = results.params['f_ees'].value
            r_pred[i] = results.params['r'].value
            resnorm[i] = results.chisqr
            is_success[i] = results.success

    param_data = np.column_stack((f_ic_pred, f_ees_pred, r_pred, resnorm, is_success))

    estimates_save_dir_path = "fitting_estimates"
    if target_dir != "" and timestamp != "":
        estimates_save_dir_path = target_dir + f"/model_output_directory/{timestamp}/fitting_estimates"
    if not os.path.exists(estimates_save_dir_path):
        os.makedirs(estimates_save_dir_path)
    np.savetxt(f"{estimates_save_dir_path}/nlls_fit_patient_{th_gradient_strength}_{timestamp}.csv", param_data, delimiter=',', header='f_ic,f_ees,r,resnorm,is_success', comments='')

    patient_prostate_mask_path = get_prostate_mask_path(grad_dataset_dir, patient_data_dir)
    patient_f_ic_map, patient_f_ees_map, patient_r_map, patient_f_vasc_map, _ = generate_param_maps(f_ic_pred, f_ees_pred, r_pred, patient_image_mask, th_gradient_strength, timestamp, target_dir, 6, "patient", "3", patient_prostate_mask_path, prostate_mask_file_pattern)