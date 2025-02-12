import torch
import torch.nn as nn
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # ✅ Use ResNet34 as encoder (as seen in checkpoint keys)
        self.encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])  # Remove FC layers

        # ✅ Ensure decoder matches saved model layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # Matches decoder.0
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Matches decoder.2
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # Matches decoder.4
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1),  # Matches decoder.6
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ✅ Function to load the trained model safely
def load_model(model_path="Notebook/best_model.pth", device="cpu"):
    model = UNet()
    
    # ✅ Use `weights_only=True` to avoid security warnings
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint)
    model.eval()  # Set model to evaluation mode
    return model
