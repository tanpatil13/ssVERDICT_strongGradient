import numpy as np
import torch, torch.nn as nn


class ssVERDICT_NN(nn.Module):
    """
    ssVERDICT neural network model for ultra-strong gradient diffusion MRI data.
    Autoencoder architecture:
        - Encoder: a series of linear layers with PReLU activations and dropout for regularization
            - Input layer: accepts normalized and averaged image signals for different b-values -> input_dim
            - Linear (input_dim -> 10)
            - PReLU activation
            - Linear (10 -> 10)
            - PReLU activation
            - Linear (10 -> 10)
            - PReLU activation
            - Dropout (0.2)
            - Linear (10 -> nparams) -> latent dimension layer
            - Softplus activation to ensure positive outputs
        - Latent dimension: estimated parameters for the ssVERDICT model (f_ic, f_ees, r)
        - Decoder: VERDICT equations for three different biophysical compartments
            - Vacular component (S_vasc) -> astrosticks compartment
            - Intracellular component (S_ic) -> sphere compartment
            - Extracellular component (S_ees) -> ball compartment
            - Output: combined signal from all compartments (X)
        - Minimize the mean squared error (MSE) between the predicted (X) and actual (input) signals
    """
    
    def __init__(self, b_values, Delta, delta, gradient_strength, nparams, device):
        """
        Initialize the ssVERDICT_Net model.
        """
        super(ssVERDICT_NN, self).__init__()
        
        self.b_values = b_values
        self.Delta = Delta
        self.delta = delta
        self.gradient_strength = gradient_strength
        self.nparams = nparams
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(len(b_values), 10),
            nn.PReLU(),
            nn.Linear(10, 10),
            nn.PReLU(),
            nn.Linear(10, 10),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, nparams),
            nn.Softplus()
        )

    def forward(self, X):
        """
        Forward pass of the ssVERDICT_NN model.
        """
        params = self.encoder(X)

        # constrain parameters to biophysically-realistic ranges
        f_ic = torch.clamp(params[:,0].unsqueeze(1), min=0.001, max=0.999)
        f_ees = torch.clamp(params[:,1].unsqueeze(1), min=0.001, max=0.999)
        r = torch.clamp(params[:,2].unsqueeze(1), min=0.001, max=14.999)
        
        # sphere GPD approximation
        SPHERE_TRASCENDENTAL_ROOTS = np.r_[
        2.081575978, 5.940369990, 9.205840145,
        12.40444502, 15.57923641, 18.74264558, 21.89969648,
        25.05282528, 28.20336100, 31.35209173, 34.49951492,
        37.64596032, 40.79165523, 43.93676147, 47.08139741,
        50.22565165, 53.36959180, 56.51327045, 59.65672900,
        62.80000055, 65.94311190, 69.08608495, 72.22893775,
        75.37168540, 78.51434055, 81.65691380, 84.79941440,
        87.94185005, 91.08422750, 94.22655255, 97.36883035
        ]

        d_ees = 2
        d_ic = 2
        d_vasc = 8

        device = self.device
        
        alpha = torch.FloatTensor(SPHERE_TRASCENDENTAL_ROOTS).to(device) / (r)  # Ensure alpha is on the same device as x
        alpha2 = alpha ** 2
        alpha2D = alpha2 * d_ic
        alpha = alpha.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)
        alpha2D = alpha2D.unsqueeze(1)

        gamma = 2.675987e2
        first_factor = -2*(gamma*self.gradient_strength)**2 / 2

        delta = self.delta.unsqueeze(0).unsqueeze(2)
        Delta = self.Delta.unsqueeze(0).unsqueeze(2)
        b_values = torch.FloatTensor(self.b_values).to(device)
        
        summands = (alpha ** (-4) / (alpha2 * (r.unsqueeze(2))**2 - 2) * (
                            2 * delta - (
                            2 +
                            torch.exp(-alpha2D * (Delta - delta)) -
                            2 * torch.exp(-alpha2D * delta) -
                            2 * torch.exp(-alpha2D * Delta) +
                            torch.exp(-alpha2D * (Delta + delta))
                        ) / (alpha2D)
                    )
                )
        
        pi_tensor = torch.FloatTensor([torch.pi]).to(device)
        
        S_vasc = (1 - f_ic - f_ees) * ((torch.sqrt(pi_tensor) * torch.erf(torch.sqrt(b_values * d_vasc))) /
                (2 * torch.sqrt(b_values * d_vasc)))                              # astrosticks compartment
        S_ic = f_ic * torch.exp(first_factor * torch.sum(summands, 2))            # sphere compartment
        S_ees = f_ees * torch.exp(-b_values * d_ees)                              # ball compartment       
        X = S_vasc + S_ic + S_ees

        return X, f_ic, f_ees, r