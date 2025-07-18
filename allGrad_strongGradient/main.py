import sys
import numpy as np
import torch
import re
from train_validate import perform_training_inference

def get_diffusion_parameters(th_bvals, gamma, th_gradient_strength):
    """
    Return the Delta and delta tensors based on the provided maximum theoritical gradient strength.
    Compute the actual gradient strengths based on the theoritical b-values, Delta, and delta values.
    th_bvals: List of theoretical b-values in ms.
    gamma: Gyromagnetic ratio for hydrogen in rad/ms/mT.
    th_gradient_strength: Theoretical maximum gradient strength in mT/m.
    """
    # Assign Delta and delta for 80mT/m gradient strength
    if (th_gradient_strength == 'G80'):
        Delta = torch.FloatTensor([32]*len(th_bvals))  # in ms
        delta = torch.FloatTensor([16]*len(th_bvals))  # in ms

    # Assign Delta and delta for 40mT/m gradient strength
    elif (th_gradient_strength == 'G40'):
        Delta = torch.FloatTensor([48]*len(th_bvals))  # in ms
        delta = torch.FloatTensor([26]*len(th_bvals))  # in ms

    # Assign Delta and delta for 300mT/m (default) gradient strength
    else:
        Delta = torch.FloatTensor([25]*len(th_bvals))  # in ms
        delta = torch.FloatTensor([5]*len(th_bvals))  # in ms
    
    gradient_strength = torch.FloatTensor([np.sqrt(th_bvals[i]/1000)/(gamma*delta[i]*np.sqrt(Delta[i]-delta[i]/3)) for i, _ in enumerate(th_bvals)])  # in mT/m

    return Delta, delta, gradient_strength

def main(timestamp):

    train_data_dir = {
        'healthy_control_data': ['240616-301', '171221-602'],
        'patient_data': ['060622-601', '170622-601', '170622-602']
    }
    val_data_dir = {
        'healthy_control_data': ['070322-601'],
        'patient_data': ['230622-601']
    }

    healthy_test_data_dir = {
        'healthy_control_data': ['100622-601']
    }
    patient_test_data_dir = {
        'patient_data': ['200722-601']
    }

    grad_dataset_dir = "../strong_gradient_dataset/"
    file_extension_pattern = r'\.nii\.gz'

    # File patterns for 300mT/m gradient data
    G300_file_pattern = r'.+Delta25_delta5_TE54_MS_b3k_TR3500.+sub_denoisedMPPCA_gibbsCorrRPG_padded_depadded_TD'
    image_G300_file_pattern = re.compile(rf'{G300_file_pattern}{file_extension_pattern}')
    x_bvec_G300_file_pattern = re.compile(rf'{G300_file_pattern}_mod_x{file_extension_pattern}')
    y_bvec_G300_file_pattern = re.compile(rf'{G300_file_pattern}_mod_y{file_extension_pattern}')
    z_bvec_G300_file_pattern = re.compile(rf'{G300_file_pattern}_mod_z{file_extension_pattern}')

    # File patterns for 80mT/m gradient data
    G80_file_pattern = r'.+Delta32_delta16_TE70_MS_b3k_TR3500.+sub_denoisedMPPCA_gibbsCorrRPG_padded_depadded_TD'
    image_G80_file_pattern = re.compile(rf'{G80_file_pattern}{file_extension_pattern}')
    x_bvec_G80_file_pattern = re.compile(rf'{G80_file_pattern}_mod_x{file_extension_pattern}')
    y_bvec_G80_file_pattern = re.compile(rf'{G80_file_pattern}_mod_y{file_extension_pattern}')
    z_bvec_G80_file_pattern = re.compile(rf'{G80_file_pattern}_mod_z{file_extension_pattern}')

    # File patterns for 40mT/m gradient data
    G40_file_pattern = r'.+Delta48_delta26_TE95_MS_b3k_TR3500.+sub_denoisedMPPCA_gibbsCorrRPG_padded_depadded_TD'
    image_G40_file_pattern = re.compile(rf'{G40_file_pattern}{file_extension_pattern}')
    x_bvec_G40_file_pattern = re.compile(rf'{G40_file_pattern}_mod_x{file_extension_pattern}')
    y_bvec_G40_file_pattern = re.compile(rf'{G40_file_pattern}_mod_y{file_extension_pattern}')
    z_bvec_G40_file_pattern = re.compile(rf'{G40_file_pattern}_mod_z{file_extension_pattern}')

    th_bvals = [1e-3, 50, 500, 1500, 2000, 3000]
    gamma = 2.675987e2  # rad/ms/mT, gyromagnetic ratio for hydrogen
    Delta_G300, delta_G300, gradient_strength_G300 = get_diffusion_parameters(th_bvals, gamma, 'G300')
    Delta_G80, delta_G80, gradient_strength_G80 = get_diffusion_parameters(th_bvals, gamma, 'G80')
    Delta_G40, delta_G40, gradient_strength_G40 = get_diffusion_parameters(th_bvals, gamma, 'G40')


if __name__ == "__main__":
    """
    Entry point for the script. Expects a timestamp argument to be passed.
    Raises a ValueError if the timestamp argument is not provided.
    """
    if len(sys.argv) < 2:
        raise ValueError("Timestamp argument required")
    main(sys.argv[1])
