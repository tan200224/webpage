import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

IMAGE_CHANNEL = 1
Z_DIM = 512
G_HIDDEN = 256
X_DIM = 256
D_HIDDEN = 64

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class GAN(nn.Module):

    def __init__(self):
        super(GAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
    
    def generate(self, image):
        with torch.no_grad():
            # Extract features from the input mask
            batch_size = image.size(0)
            # Flatten the image to create a latent vector
            latent = torch.flatten(image, start_dim=1)
            # Reshape to match generator input shape
            latent = latent.view(batch_size, -1, 1, 1)
            # Generate output using the generator
            output = self.generator(latent)
            # Ensure output has the right shape for the application (4 channels)
            if output.size(1) != 4:
                # If not 4 channels, duplicate the single channel
                output = output.repeat(1, 4, 1, 1)
            return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input layer: Upsample from Z_DIM to 4x4 spatial resolution
            nn.ConvTranspose2d(4196, G_HIDDEN * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 16),
            nn.ReLU(True),
            # 1st hidden layer: Upsample to 8x8
            nn.ConvTranspose2d(G_HIDDEN * 16, G_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            # 2nd hidden layer: Upsample to 16x16
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
            # 3rd hidden layer: Upsample to 32x32
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            # 4th hidden layer: Upsample to 64x64
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True),
            # 5th hidden layer: Upsample to 128x128
            nn.ConvTranspose2d(G_HIDDEN, G_HIDDEN // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN // 2),
            nn.ReLU(True),
            # Output layer: Upsample to 256x256
            nn.ConvTranspose2d(G_HIDDEN // 2, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)  

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(in_channels, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Adaptive Average Pooling to reduce to 1x1
            nn.AdaptiveAvgPool2d(1),
            # Output layer
            nn.Conv2d(D_HIDDEN * 8, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)