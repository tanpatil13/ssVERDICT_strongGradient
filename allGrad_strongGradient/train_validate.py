import numpy as np
import torch, torch.nn as nn
from torchinfo import summary
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
from preprocess_data import preprocess_images
from autoencoder_model import ssVERDICT_NN
from postprocess_data import generate_param_maps


def set_random_state(seed):
    """
    Set random seed/state for better reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_dataloaders(train_preprocessed_image_data, val_preprocessed_image_data, healthy_test_preprocessed_image_data, patient_test_preprocessed_image_data, batch_size=256):
    """
    Create dataloaders for training, validation, and testing datasets.
    """
    
    set_random_state(13)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=torch.from_numpy(train_preprocessed_image_data.astype(np.float32)),
        batch_size=batch_size,
        shuffle=True,
        # num_workers=4
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=torch.from_numpy(val_preprocessed_image_data.astype(np.float32)),
        batch_size=batch_size,
        shuffle=False,
        # num_workers=4
    )

    healthy_test_dataloader = torch.utils.data.DataLoader(
        dataset=torch.from_numpy(healthy_test_preprocessed_image_data.astype(np.float32)),
        batch_size=batch_size,
        shuffle=False,
        # num_workers=4
    )
    patient_test_dataloader = torch.utils.data.DataLoader(
        dataset=torch.from_numpy(patient_test_preprocessed_image_data.astype(np.float32)),
        batch_size=batch_size,
        shuffle=False,
        # num_workers=4
    )
    return train_dataloader, val_dataloader, healthy_test_dataloader, patient_test_dataloader

def train_single_epoch(model, train_dataloader, criterion, optimizer, device='cuda'):
    """
    Train the model for a single epoch.
    Args:
        model: The self-supervised autoencoder model.
        train_dataloader: DataLoader for the training dataset.
        criterion: Loss function.
        optimizer: Optimizer for updating model parameters.
        device: Device to run the model on (default is 'cuda').
    Returns:
        epoch_train_loss: Average training loss for the epoch.
        all_f_ic_pred: List of predicted f_ic values for the training dataset.
        all_f_ees_pred: List of predicted f_ees values for the training dataset.
        all_r_pred: List of predicted r values for the training dataset.
    """

    train_loss = 0.0
    all_f_ic_pred, all_f_ees_pred, all_r_pred = [], [], []

    model.train()

    for S_train in tqdm(train_dataloader, desc="Training", unit="batch"):
        S_train = S_train.to(device)

        S_pred, f_ic_pred, f_ees_pred, r_pred = model(S_train)
        loss = criterion(S_pred, S_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        all_f_ic_pred.extend(f_ic_pred.detach().cpu().numpy())
        all_f_ees_pred.extend(f_ees_pred.detach().cpu().numpy())
        all_r_pred.extend(r_pred.detach().cpu().numpy())
        
    epoch_train_loss = train_loss / len(train_dataloader)
    print(f"Epoch Train Loss: {epoch_train_loss}")
    print("\n")

    return epoch_train_loss, all_f_ic_pred, all_f_ees_pred, all_r_pred

def val_test_model(split_type, model, val_test_dataloader, criterion, device='cuda'):
    """
    Validate or test the model on the validation or test dataset.
    Args:
        split_type: Type of split ('Validation' or 'Test').
        model: The self-supervised autoencoder model.
        val_test_dataloader: DataLoader for the validation or test dataset.
        criterion: Loss function.
        device: Device to run the model on (default is 'cuda').
    Returns:
        epoch_val_test_loss: Average validation or test loss for the epoch.
        all_f_ic_pred: List of predicted f_ic values for the validation or test dataset.
        all_f_ees_pred: List of predicted f_ees values for the validation or test dataset.
        all_r_pred: List of predicted r values for the validation or test dataset.
    """

    val_test_loss = 0.0
    all_f_ic_pred, all_f_ees_pred, all_r_pred = [], [], []

    model.eval()

    with torch.no_grad():
        for S_val_test in tqdm(val_test_dataloader, desc=f"{split_type} Evaluation", unit="batch"):
            S_val_test = S_val_test.to(device)

            S_pred, f_ic_pred, f_ees_pred, r_pred = model(S_val_test)
            loss = criterion(S_pred, S_val_test)

            val_test_loss += loss.item()
            all_f_ic_pred.extend(f_ic_pred.detach().cpu().numpy())
            all_f_ees_pred.extend(f_ees_pred.detach().cpu().numpy())
            all_r_pred.extend(r_pred.detach().cpu().numpy())
            
        epoch_val_test_loss = val_test_loss / len(val_test_dataloader)
        print(f"Epoch {split_type} Loss: {epoch_val_test_loss}")
        print("\n")

    return epoch_val_test_loss, all_f_ic_pred, all_f_ees_pred, all_r_pred

def train_model(model, train_dataloader, val_dataloader, criterion, optmizier, num_epochs='30', device='cuda', timestamp="", scheduler=None):
    """
    Train the self-supervised autoencoder model.
    Validate the model on the validation dataset after each epoch.
    Save the best model checkpoints based on validation loss, along with saving the model every 5 epochs.
    Args:
        model: The self-supervised autoencoder model.
        train_dataloader: DataLoader for the training dataset.
        val_dataloader: DataLoader for the validation dataset.
        criterion: Loss function.
        optmizier: Optimizer for updating model parameters.
        num_epochs: Number of epochs to train the model.
        device: Device to run the model on (default is 'cuda').
        timestamp: Timestamp for saving model checkpoints.
    Returns:
        train_losses: List of training losses for each epoch.
        val_losses: List of validation losses for each epoch.
        best_train_param_estimates: Dictionary containing the best training parameter estimates.
        best_val_param_estimates: Dictionary containing the best validation parameter estimates.
        best_checkpoint_path: Path to the best model checkpoint.
    """

    train_losses, val_losses = [], []
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\n------------------------")

        epoch_train_loss, train_f_ic_pred, train_f_ees_pred, train_r_pred = train_single_epoch(model, train_dataloader, criterion, optmizier, device)
        train_losses.append(epoch_train_loss)

        epoch_val_loss, val_f_ic_pred, val_f_ees_pred, val_r_pred = val_test_model("Validation", model, val_dataloader, criterion, device)
        val_losses.append(epoch_val_loss)

        if epoch_train_loss < best_train_loss:
            best_train_loss = epoch_train_loss
            best_train_f_ic_pred, best_train_f_ees_pred, best_train_r_pred = train_f_ic_pred, train_f_ees_pred, train_r_pred
            best_train_param_estimates = {
                'f_ic': best_train_f_ic_pred,
                'f_ees': best_train_f_ees_pred,
                'r': best_train_r_pred
            }

        model_save_dir_path = "model_save_directory"
        if not os.path.exists(model_save_dir_path):
            os.makedirs(model_save_dir_path)

        best_checkpoint_path = f"{model_save_dir_path}/ssVERDICT_checkpoint_best_epoch_{timestamp}.pth"
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_f_ic_pred, best_val_f_ees_pred, best_val_r_pred = val_f_ic_pred, val_f_ees_pred, val_r_pred
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optmizier.state_dict()
            }, best_checkpoint_path)
            best_val_param_estimates = {
                'f_ic': best_val_f_ic_pred,
                'f_ees': best_val_f_ees_pred,
                'r': best_val_r_pred
            }
            best_epoch = epoch + 1
        
        if scheduler is not None:
            scheduler.step()
            
    print(f"Best Epoch: {best_epoch}")

    return train_losses, val_losses, best_train_param_estimates, best_val_param_estimates, best_checkpoint_path

def plot_loss_curves(train_losses, val_losses, timestamp):
    """
    Plot the training and validation loss curves.
    Args:
        train_losses: List of training losses for each epoch.
        val_losses: List of validation losses for each epoch.
        th_gradient_strength: Theoretical gradient strength.
        timestamp: Timestamp for saving the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.savefig(timestamp + f'/train_val_loss_curves_{timestamp}.png', dpi=300, bbox_inches='tight')

def perform_training_inference(grad_dataset_dir, train_data_dir, val_data_dir, healthy_test_data_dir, patient_test_data_dir,
                                image_G300_file_pattern, x_bvec_G300_file_pattern, y_bvec_G300_file_pattern, z_bvec_G300_file_pattern, 
                                image_G80_file_pattern, x_bvec_G80_file_pattern, y_bvec_G80_file_pattern, z_bvec_G80_file_pattern,
                                image_G40_file_pattern, x_bvec_G40_file_pattern, y_bvec_G40_file_pattern, z_bvec_G40_file_pattern,
                                Delta_G300, delta_G300, gradient_strength_G300,
                                Delta_G80, delta_G80, gradient_strength_G80,
                                Delta_G40, delta_G40, gradient_strength_G40,
                                th_bvals, timestamp):
    """
    Preprocess the images, create dataloaders, train the self-supervised 1-D DKI autoencoder model, validate and test it.
    Generate the estimated parameter maps for D_k and K for both healthy controls and patients.
    Args:
        train_data_dir: Dictionary containing training data directories.
        val_data_dir: Dictionary containing validation data directories.
        grad_dataset_dir: Directory containing the gradient dataset.
        healthy_test_data_dir: Dictionary containing healthy control test data directories.
        patient_test_data_dir: Dictionary containing patient test data directories.
        image_G300_file_pattern: Regex pattern for 300mT/m gradient images.
        x_bvec_G300_file_pattern: Regex pattern for 300mT/m gradient x b-vectors.
        y_bvec_G300_file_pattern: Regex pattern for 300mT/m gradient y b-vectors.
        z_bvec_G300_file_pattern: Regex pattern for 300mT/m gradient z b-vectors.
        image_G80_file_pattern: Regex pattern for 80mT/m gradient images.
        x_bvec_G80_file_pattern: Regex pattern for 80mT/m gradient x b-vectors.
        y_bvec_G80_file_pattern: Regex pattern for 80mT/m gradient y b-vectors.
        z_bvec_G80_file_pattern: Regex pattern for 80mT/m gradient z b-vectors.
        image_G40_file_pattern: Regex pattern for 40mT/m gradient images.
        x_bvec_G40_file_pattern: Regex pattern for 40mT/m gradient x b-vectors.
        y_bvec_G40_file_pattern: Regex pattern for 40mT/m gradient y b-vectors.
        z_bvec_G40_file_pattern: Regex pattern for 40mT/m gradient z b-vectors.
        Delta_G300, delta_G300, gradient_strength_G300: Delta, delta, and gradient strength for 300mT/m gradient.
        Delta_G80, delta_G80, gradient_strength_G80: Delta, delta, and gradient strength for 80mT/m gradient.
        Delta_G40, delta_G40, gradient_strength_G40: Delta, delta, and gradient strength for 40mT/m gradient.
        th_bvals: List of b-values in ms/µm^2.
        timestamp: Timestamp for saving model checkpoints and plots.
    """
    
    _, train_preprocessed_image_G300_data, _, train_image_G300_mask = preprocess_images(train_data_dir,
                                                                grad_dataset_dir, image_G300_file_pattern, x_bvec_G300_file_pattern, 
                                                                y_bvec_G300_file_pattern, z_bvec_G300_file_pattern, th_bvals)
    _, train_preprocessed_image_G80_data, _, train_image_G80_mask = preprocess_images(train_data_dir,
                                                                    grad_dataset_dir, image_G80_file_pattern, x_bvec_G80_file_pattern, 
                                                                    y_bvec_G80_file_pattern, z_bvec_G80_file_pattern, th_bvals)
    _, train_preprocessed_image_G40_data, _, train_image_G40_mask = preprocess_images(train_data_dir,
                                                                    grad_dataset_dir, image_G40_file_pattern, x_bvec_G40_file_pattern, 
                                                                    y_bvec_G40_file_pattern, z_bvec_G40_file_pattern, th_bvals)
    train_preprocessed_image_data = np.concatenate((train_preprocessed_image_G300_data, train_preprocessed_image_G80_data, train_preprocessed_image_G40_data), axis=1)


    _, val_preprocessed_image_G300_data, _, val_image_G300_mask = preprocess_images(val_data_dir,
                                                                grad_dataset_dir, image_G300_file_pattern, x_bvec_G300_file_pattern, 
                                                                y_bvec_G300_file_pattern, z_bvec_G300_file_pattern, th_bvals)
    _, val_preprocessed_image_G80_data, _, val_image_G80_mask = preprocess_images(val_data_dir,
                                                                    grad_dataset_dir, image_G80_file_pattern, x_bvec_G80_file_pattern, 
                                                                    y_bvec_G80_file_pattern, z_bvec_G80_file_pattern, th_bvals)
    _, val_preprocessed_image_G40_data, _, val_image_G40_mask = preprocess_images(val_data_dir,
                                                                    grad_dataset_dir, image_G40_file_pattern, x_bvec_G40_file_pattern, 
                                                                    y_bvec_G40_file_pattern, z_bvec_G40_file_pattern, th_bvals)
    val_preprocessed_image_data = np.concatenate((val_preprocessed_image_G300_data, val_preprocessed_image_G80_data, val_preprocessed_image_G40_data), axis=1)

    _, healthy_test_preprocessed_image_G300_data, _, healthy_test_image_G300_mask = preprocess_images(healthy_test_data_dir,
                                                                grad_dataset_dir, image_G300_file_pattern, x_bvec_G300_file_pattern, 
                                                                y_bvec_G300_file_pattern, z_bvec_G300_file_pattern, th_bvals)
    _, healthy_test_preprocessed_image_G80_data, _, healthy_test_image_G80_mask = preprocess_images(healthy_test_data_dir,
                                                                    grad_dataset_dir, image_G80_file_pattern, x_bvec_G80_file_pattern, 
                                                                    y_bvec_G80_file_pattern, z_bvec_G80_file_pattern, th_bvals)
    _, healthy_test_preprocessed_image_G40_data, _, healthy_test_image_G40_mask = preprocess_images(healthy_test_data_dir,
                                                                    grad_dataset_dir, image_G40_file_pattern, x_bvec_G40_file_pattern, 
                                                                    y_bvec_G40_file_pattern, z_bvec_G40_file_pattern, th_bvals)
    healthy_test_preprocessed_image_data = np.concatenate((healthy_test_preprocessed_image_G300_data, healthy_test_preprocessed_image_G80_data, healthy_test_preprocessed_image_G40_data), axis=1)

    _,  patient_test_preprocessed_image_G300_data, _, patient_test_image_G300_mask = preprocess_images(patient_test_data_dir,
                                                                grad_dataset_dir, image_G300_file_pattern, x_bvec_G300_file_pattern, 
                                                                    y_bvec_G300_file_pattern, z_bvec_G300_file_pattern, th_bvals)
    _,  patient_test_preprocessed_image_G80_data, _, patient_test_image_G80_mask = preprocess_images(patient_test_data_dir,
                                                                    grad_dataset_dir, image_G80_file_pattern, x_bvec_G80_file_pattern, 
                                                                    y_bvec_G80_file_pattern, z_bvec_G80_file_pattern, th_bvals)
    _,  patient_test_preprocessed_image_G40_data, _, patient_test_image_G40_mask = preprocess_images(patient_test_data_dir,
                                                                    grad_dataset_dir, image_G40_file_pattern, x_bvec_G40_file_pattern, 
                                                                    y_bvec_G40_file_pattern, z_bvec_G40_file_pattern, th_bvals)
    patient_test_preprocessed_image_data = np.concatenate((patient_test_preprocessed_image_G300_data, patient_test_preprocessed_image_G80_data, patient_test_preprocessed_image_G40_data), axis=1)

    num_epochs = 40
    lr = 1e-3
    nparams = 3
    batch_size = 256

    Delta = torch.FloatTensor(Delta_G300.tolist() + Delta_G80.tolist() + Delta_G40.tolist())
    delta = torch.FloatTensor(delta_G300.tolist() + delta_G80.tolist() + delta_G40.tolist())
    gradient_strength = torch.FloatTensor(gradient_strength_G300.tolist() + gradient_strength_G80.tolist() + gradient_strength_G40.tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataloader, val_dataloader, healthy_test_dataloader, patient_test_dataloader = create_dataloaders(train_preprocessed_image_data, val_preprocessed_image_data, 
                                                                            healthy_test_preprocessed_image_data, patient_test_preprocessed_image_data, batch_size)

    th_bvals_new = [th_bvals[i]/1000 for i in range(len(th_bvals))]  # Convert to ms/µm^2
    th_bvals_all_grad = th_bvals_new * 3  # Repeat for 300, 80, and 40 mT/m gradient strengths
    model = ssVERDICT_NN(th_bvals_all_grad, Delta.to(device), delta.to(device), gradient_strength.to(device), nparams, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

    print("Model Summary: ")
    print(summary(model, input_size=(batch_size, len(th_bvals_all_grad)), device=device.type))

    train_losses, val_losses, best_train_param_estimates, best_val_param_estimates, best_checkpoint_path = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optmizier=optimizer,
        num_epochs=num_epochs,
        device=device,
        scheduler=scheduler
    )

    plot_loss_curves(train_losses, val_losses, timestamp)

    healthy_test_loss, healthy_test_f_ic_pred, healthy_test_f_ees_pred, healthy_test_r_pred = val_test_model("Test", model, healthy_test_dataloader, criterion, device)
    print(f"Healthy Control Test Loss: {healthy_test_loss}")

    healthy_test_f_ic_map, healthy_test_f_ees_map, healthy_test_r_map, _ = generate_param_maps(healthy_test_f_ic_pred, healthy_test_f_ees_pred, healthy_test_r_pred, healthy_test_image_G300_mask, timestamp, 7, "healthy")

    patient_test_loss, patient_test_f_ic_pred, patient_test_f_ees_pred, patient_test_r_pred = val_test_model("Test", model, patient_test_dataloader, criterion, device)
    print(f"Patient Test Loss: {patient_test_loss}")

    patient_test_f_ic_map, patient_test_f_ees_map, patient_test_r_map, _ = generate_param_maps(patient_test_f_ic_pred, patient_test_f_ees_pred, patient_test_r_pred, patient_test_image_G300_mask, timestamp, 8, "patient")