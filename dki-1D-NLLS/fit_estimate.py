import os
import numpy as np
from lmfit import Parameters, Minimizer
from model import dkiResiduals
from commons.preprocess_data import preprocess_images
from postprocess_data import generate_param_maps

def get_prostate_mask_path(grad_dataset_dir, prostate_mask_dir):
    for subject_type, subject_ids in prostate_mask_dir.items():
        for subject_id in subject_ids:
            return os.path.join(grad_dataset_dir, subject_type, subject_id)

def perform_fit(grad_dataset_dir, patient_data_dir, healthy_data_dir,
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
                                                                y_bvec_file_pattern, z_bvec_file_pattern, prostate_mask_file_pattern, th_bvals)

    _,  healthy_preprocessed_image_data, _, healthy_image_mask = preprocess_images(healthy_data_dir,
                                                                grad_dataset_dir, image_file_pattern, x_bvec_file_pattern, 
                                                                y_bvec_file_pattern, z_bvec_file_pattern, prostate_mask_file_pattern, th_bvals)

    num_itr = 1

    patient_img_dim = patient_preprocessed_image_data.shape
    patient_D_k_pred = np.zeros(patient_img_dim[0])
    patient_K_pred = np.zeros(patient_img_dim[0])
    patient_resnorm = np.zeros(patient_img_dim[0])
    patient_is_success = np.zeros(patient_img_dim[0], dtype=bool)

    healthy_img_dim = healthy_preprocessed_image_data.shape
    healthy_D_k_pred = np.zeros(healthy_img_dim[0])
    healthy_K_pred = np.zeros(healthy_img_dim[0])
    healthy_resnorm = np.zeros(healthy_img_dim[0])
    healthy_is_success = np.zeros(healthy_img_dim[0], dtype=bool)

    Delta = np.array(Delta)
    delta = np.array(delta)
    gradient_strength = np.array(gradient_strength)

    th_bvals_scaled = [th_bvals[i]/1000 for i in range(len(th_bvals))]  # Convert to ms/µm^2

    print(f"Running the DKI NLLS fitting on the patient data for gradient strength: {th_gradient_strength} ...")

    for i in range(patient_img_dim[0]):
        patient_params = Parameters()
        patient_params.add('D_k', value=1.35, min=0.2, max=2.5)
        patient_params.add('K', value=1.75, min=0, max=3.5)

        for k in range(num_itr):
            patient_fitter = Minimizer(dkiResiduals, patient_params, fcn_args=(patient_preprocessed_image_data[i, :], th_bvals_scaled))
            patient_results = patient_fitter.minimize(method='leastsq')
            patient_D_k_pred[i] = patient_results.params['D_k'].value
            patient_K_pred[i] = patient_results.params['K'].value
            patient_resnorm[i] = patient_results.chisqr
            patient_is_success[i] = patient_results.success

    patient_param_data = np.column_stack((patient_D_k_pred, patient_K_pred, patient_resnorm, patient_is_success))

    print(f"Running the DKI NLLS fitting on the healthy control data for gradient strength: {th_gradient_strength} ...")

    for i in range(healthy_img_dim[0]):
        healthy_params = Parameters()
        healthy_params.add('D_k', value=1.35, min=0.2, max=2.7)
        healthy_params.add('K', value=1.75, min=0, max=3.5)

        for k in range(num_itr):
            healthy_fitter = Minimizer(dkiResiduals, healthy_params, fcn_args=(healthy_preprocessed_image_data[i, :], th_bvals_scaled))
            healthy_results = healthy_fitter.minimize(method='leastsq')
            healthy_D_k_pred[i] = healthy_results.params['D_k'].value
            healthy_K_pred[i] = healthy_results.params['K'].value
            healthy_resnorm[i] = healthy_results.chisqr
            healthy_is_success[i] = healthy_results.success

    healthy_param_data = np.column_stack((healthy_D_k_pred, healthy_K_pred, healthy_resnorm, healthy_is_success))

    estimates_save_dir_path = "fitting_estimates"
    if target_dir != "" and timestamp != "":
        estimates_save_dir_path = target_dir + f"/model_output_directory/{timestamp}/fitting_estimates"
    if not os.path.exists(estimates_save_dir_path):
        os.makedirs(estimates_save_dir_path)
    np.savetxt(f"{estimates_save_dir_path}/dki_nlls_fit_patient_3_{th_gradient_strength}_{timestamp}.csv", patient_param_data, delimiter=',', header='D_k,K,resnorm,is_success', comments='')
    np.savetxt(f"{estimates_save_dir_path}/dki_nlls_fit_patient_5_{th_gradient_strength}_{timestamp}.csv", healthy_param_data, delimiter=',', header='D_k,K,resnorm,is_success', comments='')

    patient_prostate_mask_path = get_prostate_mask_path(grad_dataset_dir, patient_data_dir)
    patient_D_k_map, patient_K_map = generate_param_maps(patient_D_k_pred, patient_K_pred, patient_image_mask, th_gradient_strength, timestamp, target_dir, 6, "patient", "3", patient_prostate_mask_path, prostate_mask_file_pattern)

    healthy_prostate_mask_path = get_prostate_mask_path(grad_dataset_dir, healthy_data_dir)
    healthy_D_k_map, healthy_K_map = generate_param_maps(healthy_D_k_pred, healthy_K_pred, healthy_image_mask, th_gradient_strength, timestamp, target_dir, 7, "patient", "5", healthy_prostate_mask_path, prostate_mask_file_pattern)