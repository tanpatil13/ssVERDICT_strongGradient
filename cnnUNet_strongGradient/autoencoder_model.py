import numpy as np
import torch, torch.nn as nn

class ConvLayers(nn.Module):
    """(Conv2d => BatchNorm => ReLU) x 4"""
    def __init__(self, in_channels, out_channels):
        super(ConvLayers, self).__init__()
        hidden_channels = (in_channels + out_channels) // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels), 
            nn.PReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels), 
            nn.PReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), 
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), 
            nn.PReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class ssVERDICT_UNet(nn.Module):
    def __init__(self, b_values, Delta, delta, gradient_strength, nparams, device, batch_size):
        super(ssVERDICT_UNet, self).__init__()

        self.latent_dim = 10
        self.b_values = b_values
        self.Delta = Delta
        self.delta = delta
        self.gradient_strength = gradient_strength
        self.nparams = nparams
        self.device = device
        self.batch_size = batch_size

        self.down1 = ConvLayers(len(b_values), 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ConvLayers(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = ConvLayers(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = ConvLayers(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvLayers(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.conv4 = ConvLayers(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = ConvLayers(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ConvLayers(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = ConvLayers(128, 64)

        self.final = nn.Conv2d(64, nparams, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        bn = self.bottleneck(self.pool4(d4))
        # bn = self.bottleneck(self.pool3(d3))

        u4 = self.up4(bn)
        u4 = self.conv4(torch.cat([u4, d4], dim=1))

        u3 = self.up3(u4)
        # u3 = self.up3(bn)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        params = self.final(u1)
        params = self.final_activation(params)


        f_ic_min, f_ic_max = 0.001, 0.999
        f_ees_min, f_ees_max = 0.001, 0.999
        r_min, r_max = 0.001, 14.999

        f_ic = f_ic_min + (f_ic_max - f_ic_min) * params[:, 0, :, :]
        f_ees = f_ees_min + (f_ees_max - f_ees_min) * params[:, 1, :, :]
        r = r_min + (r_max - r_min) * params[:, 2, :, :]

        # Get the spatial dimensions of the image
        img_dim = f_ic.shape
        f_ic_ = f_ic.reshape(img_dim[0], img_dim[1]*img_dim[2]).unsqueeze(1)
        f_ees_ = f_ees.reshape(img_dim[0], img_dim[1]*img_dim[2]).unsqueeze(1)
        r_ = r.reshape(img_dim[0], img_dim[1]*img_dim[2]).unsqueeze(1)
    
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
        alpha = torch.FloatTensor(SPHERE_TRASCENDENTAL_ROOTS).view(1, len(SPHERE_TRASCENDENTAL_ROOTS), 1).to(device) / (r_)  # Ensure alpha is on the same device as x
        alpha2 = alpha ** 2
        alpha2D = alpha2 * d_ic
        alpha = alpha.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)
        alpha2D = alpha2D.unsqueeze(1)

        gamma = 2.675987e2
        first_factor = -2*(gamma*self.gradient_strength)**2 / 2
        first_factor = first_factor.unsqueeze(0).unsqueeze(2)  # Reshape to match the dimensions of the summands

        delta = self.delta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        Delta = self.Delta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        b_values = torch.FloatTensor(self.b_values).to(device)
        
        summands = (alpha ** (-4) / (alpha2 * (r_.unsqueeze(2))**2 - 2) * (
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
        
        S_vasc = (1 - f_ic_ - f_ees_) * ((torch.sqrt(pi_tensor) * torch.erf(torch.sqrt(b_values * d_vasc))) /
                (2 * torch.sqrt(b_values * d_vasc))).unsqueeze(0).unsqueeze(2)                                  # astrosticks compartment
        S_ic = f_ic_ * torch.exp(first_factor * torch.sum(summands, 2))                                         # sphere compartment
        S_ees = f_ees_ * torch.exp(-b_values * d_ees).unsqueeze(0).unsqueeze(2)                                 # ball compartment       
        S_pred = S_vasc + S_ic + S_ees

        S_pred = S_pred.reshape(img_dim[0], S_pred.size(1), img_dim[1], img_dim[2])

        return S_pred, f_ic, f_ees, r