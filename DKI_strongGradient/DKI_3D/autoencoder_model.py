import torch
import torch.nn as nn

def dki_forward(D, W, bvals, gradient_vectors):
    """
    Predict normalized signal using full 3D DKI model.
    """
    n = gradient_vectors  # [B, N, 3]
    b = bvals.unsqueeze(0)

    n1, n2, n3 = n[..., 0], n[..., 1], n[..., 2]

    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = D.T
    Dn = (
        n1**2 * Dxx[:, None] +
        n2**2 * Dyy[:, None] +
        n3**2 * Dzz[:, None] +
        2 * n1 * n2 * Dxy[:, None] +
        2 * n1 * n3 * Dxz[:, None] +
        2 * n2 * n3 * Dyz[:, None]
    )

    terms = torch.stack([
        n1**4, n2**4, n3**4,
        4 * n1**3 * n2, 4 * n1**3 * n3,
        4 * n2**3 * n1, 4 * n2**3 * n3,
        4 * n3**3 * n1, 4 * n3**3 * n2,
        6 * n1**2 * n2**2, 6 * n1**2 * n3**2, 6 * n2**2 * n3**2,
        12 * n1**2 * n2 * n3, 12 * n1 * n2**2 * n3, 12 * n1 * n2 * n3**2
    ], dim=2)

    Kn = torch.sum(terms * W.unsqueeze(1), dim=2)

    MD = (Dxx[:, None] + Dyy[:, None] + Dzz[:, None])/3

    Dn = torch.clamp(Dn, min=1e-6)
    Kn = torch.clamp(Kn, min=0.0)

    exponent = -b * Dn + (b**2 * MD**2 * Kn) / 6
    S = torch.exp(exponent)

    return S


class ssDKI_3D_NN(nn.Module):
    
    def __init__(self, b_values, nparams, device):
        """
        Initialize the ssVERDICT_Net model.
        """
        super(ssDKI_3D_NN, self).__init__()
        
        self.b_values = b_values
        self.nparams = nparams
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(len(b_values), 32),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, nparams),
            nn.Sigmoid()
        )

    def forward(self, input_signal, gradient_vectors):
        params = self.encoder(input_signal)  # [B, 21]

        D_min, D_max = 0.2, 2.5  # in µm^2/ms
        D_tensor = D_min + (D_max - D_min) * params[:, :6]

        W_max = 3.5
        W_tensor = W_max * params[:, 6:]

        device = self.device
        b_values = torch.FloatTensor(self.b_values).to(device)
        gradient_vectors = torch.FloatTensor(gradient_vectors).to(device)
        S_pred = dki_forward(D_tensor, W_tensor, b_values, gradient_vectors)

        return S_pred, D_tensor, W_tensor