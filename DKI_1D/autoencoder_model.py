import torch
import torch.nn as nn

class ssDKI_NN(nn.Module):
    """
    1-D DKI neural network model for ultra-strong gradient diffusion MRI data.
    Autoencoder architecture:
        - Encoder: a series of linear layers with PReLU activations and dropout for regularization
            - Input layer: accepts normalized and averaged image signals for different b-values -> input_dim
            - Input dimension: len(b_values)
            - Linear (input_dim -> 32)
            - PReLU activation
            - Linear (32 -> 64)
            - PReLU activation
            - Linear (64 -> 128)
            - PReLU activation
            - Linear (128 -> 64)
            - PReLU activation
            - Linear (64 -> 32)
            - PReLU activation
            - Dropout (0.2)
            - Linear (32 -> nparams) -> latent dimension layer
            - Sigmoid activation to ensure outputs in [0, 1], later to be rescaled into biophysically realistic ranges
        - Latent dimension: estimated parameters for the 1-D DKI model (D_k, K)
        - Decoder: 1-D DKI equation
            - Output: normalized signal S_pred for the given b-values
        - Minimize the mean squared error (MSE) between the predicted (S_pred) and actual (input) signals
    """

    def __init__(self, b_values, nparams, device):
        """
        Initialize the ssDKI_NN model.
        """
        super(ssDKI_NN, self).__init__()
        
        self.b_values = b_values
        self.nparams = nparams
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(len(b_values), 32),
            nn.PReLU(),
            nn.Linear(32, 64),
            nn.PReLU(),
            nn.Linear(64, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, nparams),
            nn.Sigmoid()
        )

    def forward(self, X):
        """
        Forward pass of the ssVERDICT_NN model.
        """
        params = self.encoder(X)

        D_min, D_max = 0.2, 2.5  # in µm^2/ms
        D_k = (D_min + (D_max - D_min) * params[:, 0]).unsqueeze(1)

        K_max = 3.5
        K = (K_max * params[:, 1]).unsqueeze(1)

        device = self.device
        b_values = torch.FloatTensor(self.b_values).to(device)
    
        exponent = (-b_values * D_k) + (b_values**2 * D_k**2 * K)/6
        S_pred = torch.exp(exponent)

        return S_pred, D_k, K