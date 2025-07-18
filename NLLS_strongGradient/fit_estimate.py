import numpy as np
from lmfit import Parameters, Minimizer
from model import verdictResiduals
from preprocess_data import preprocess_images
from postprocess_data import generate_param_maps

def perform_fit(grad_dataset_dir, patient_data_dir,
                image_G300_file_pattern, x_bvec_G300_file_pattern, y_bvec_G300_file_pattern, z_bvec_G300_file_pattern, 
                image_G80_file_pattern, x_bvec_G80_file_pattern, y_bvec_G80_file_pattern, z_bvec_G80_file_pattern,
                image_G40_file_pattern, x_bvec_G40_file_pattern, y_bvec_G40_file_pattern, z_bvec_G40_file_pattern,
                Delta_G300, delta_G300, gradient_strength_G300,
                Delta_G80, delta_G80, gradient_strength_G80,
                Delta_G40, delta_G40, gradient_strength_G40,
                th_bvals, timestamp):
    
    _,  patient_preprocessed_image_G300_data, _, patient_image_G300_mask = preprocess_images(patient_data_dir,
                                                                grad_dataset_dir, image_G300_file_pattern, x_bvec_G300_file_pattern, 
                                                                y_bvec_G300_file_pattern, z_bvec_G300_file_pattern, th_bvals)
    _,  patient_preprocessed_image_G80_data, _, patient_image_G80_mask = preprocess_images(patient_data_dir,
                                                                    grad_dataset_dir, image_G80_file_pattern, x_bvec_G80_file_pattern,
                                                                    y_bvec_G80_file_pattern, z_bvec_G80_file_pattern, th_bvals)
    _,  patient_preprocessed_image_G40_data, _, patient_image_G40_mask = preprocess_images(patient_data_dir,
                                                                    grad_dataset_dir, image_G40_file_pattern, x_bvec_G40_file_pattern,
                                                                    y_bvec_G40_file_pattern, z_bvec_G40_file_pattern, th_bvals)
    patient_preprocessed_image_data = np.concatenate((patient_preprocessed_image_G300_data, patient_preprocessed_image_G80_data, patient_preprocessed_image_G40_data), axis=1)
    
    num_itr = 1
    img_dim = patient_preprocessed_image_data.shape
    f_ic_pred = np.zeros(img_dim[0])
    f_ees_pred = np.zeros(img_dim[0])
    r_pred = np.zeros(img_dim[0])
    resnorm = np.zeros(img_dim[0])
    is_success = np.zeros(img_dim[0], dtype=bool)

    Delta = np.array(Delta_G300.tolist() + Delta_G80.tolist() + Delta_G40.tolist())
    delta = np.array(delta_G300.tolist() + delta_G80.tolist() + delta_G40.tolist())
    gradient_strength = np.array(gradient_strength_G300.tolist() + gradient_strength_G80.tolist() + gradient_strength_G40.tolist())

    th_bvals_new = [th_bvals[i]/1000 for i in range(len(th_bvals))]  # Convert to ms/µm^2
    th_bvals_all_grad = th_bvals_new * 3  # Repeat for 300, 80, and 40 mT/m gradient strengths

    for i in range(img_dim[0]):
        params = Parameters()
        params.add('f_ic', value=0.5, min=0.001, max=0.999)
        params.add('f_ees', value=0.5, min=0.001, max=0.999)
        params.add('r', value=7.5, min=0.001, max=14.999)

        for k in range(num_itr):
            fitter = Minimizer(verdictResiduals, params, fcn_args=(patient_preprocessed_image_data[i, :], th_bvals_all_grad, Delta, delta, gradient_strength))
            results = fitter.minimize(method='leastsq')
            f_ic_pred[i] = results.params['f_ic'].value
            f_ees_pred[i] = results.params['f_ees'].value
            r_pred[i] = results.params['r'].value
            resnorm[i] = results.chisqr
            is_success[i] = results.success

    param_data = np.column_stack((f_ic_pred, f_ees_pred, r_pred, resnorm, is_success))
    np.savetxt(f"fitting_estimates/nlls_fit_patient_{timestamp}.csv", param_data, delimiter=',', header='f_ic,f_ees,r,resnorm,is_success', comments='')

    patient_f_ic_map, patient_f_ees_map, patient_r_map, patient_f_vasc_map, _ = generate_param_maps(f_ic_pred, f_ees_pred, r_pred, patient_image_G300_mask, 6, timestamp, "patient")