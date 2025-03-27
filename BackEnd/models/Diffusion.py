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
import random


Lambda = 25.0  # @param {'type':'number'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=None, seed_torch=True):
  """
  Function that controls randomness. NumPy and random modules must be imported.

  Args:
    seed : Integer
      A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
      If `True` sets the random seed for pytorch tensors, so pytorch module
      must be imported. Default is `True`.

  Returns:
    Nothing.
  """
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')


# In case that `DataLoader` is used
def seed_worker(worker_id):
  """
  DataLoader will reseed workers following randomness in
  multi-process data loading algorithm.

  Args:
    worker_id: integer
      ID of subprocess to seed. 0 means that
      the data will be loaded in the main process
      Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details

  Returns:
    Nothing
  """
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

# Inform the user if the notebook uses GPU or CPU.

def set_device():
  """
  Set the device. CUDA if available, CPU otherwise

  Args:
    None

  Returns:
    Nothing
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("WARNING: For this notebook to perform best, "
        "if possible, in the menu under `Runtime` -> "
        "`Change runtime type.`  select `GPU` ")
  else:
    print("GPU is enabled in this notebook.")

  return device


def marginal_prob_std(t, Lambda, device='cpu'):

  t = t.to(device)
  std = torch.sqrt((Lambda**(2 * t) - 1.) / 2. / np.log(Lambda))
  return std


def diffusion_coeff(t, Lambda, device='cpu'):

  diff_coeff = Lambda**t
  return diff_coeff.to(device)


class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights (frequencies) during initialization.
    # These weights (frequencies) are fixed during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    # Cosine(2 pi freq x), Sine(2 pi freq x)
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps.
  Allow time repr to input additively from the side of a convolution layer.
  """
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    # this broadcast the 2d tensor to 4d, add the same value across space.
    return self.dense(x)[..., None, None]

class Diffusion(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, channels=[32, 64, 128, 256, 512], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()

    self.marginal_prob_std = lambda t: marginal_prob_std(t, Lambda=Lambda, device=DEVICE)

    # Gaussian random feature embedding layer for time
    self.time_embed = nn.Sequential(
          GaussianFourierProjection(embed_dim=embed_dim),
          nn.Linear(embed_dim, embed_dim)
          )
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=2, bias=False, padding=1)
    self.t_mod1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

    self.conv1a = nn.Conv2d(channels[0], channels[0], 3, stride=1, bias=False, padding=1)
    self.t_mod1a = Dense(embed_dim, channels[0])
    self.gnorm1a = nn.GroupNorm(4, num_channels=channels[0])

    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False, padding=1)
    self.t_mod2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    
    self.conv2a = nn.Conv2d(channels[1], channels[1], 3, stride=1, bias=False, padding=1)
    self.t_mod2a = Dense(embed_dim, channels[1])
    self.gnorm2a = nn.GroupNorm(32, num_channels=channels[1])

    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False, padding=1)
    self.t_mod3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    
    self.conv3a = nn.Conv2d(channels[2], channels[2], 3, stride=1, bias=False, padding=1)
    self.t_mod3a = Dense(embed_dim, channels[2])
    self.gnorm3a = nn.GroupNorm(32, num_channels=channels[2])

    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False, padding=1)
    self.t_mod4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
    
    self.conv4a = nn.Conv2d(channels[3], channels[3], 3, stride=1, bias=False, padding=1)
    self.t_mod4a = Dense(embed_dim, channels[3])
    self.gnorm4a = nn.GroupNorm(32, num_channels=channels[3])

    self.conv5 = nn.Conv2d(channels[3], channels[4], 3, stride=2, bias=False, padding=1)
    self.t_mod5 = Dense(embed_dim, channels[4])
    self.gnorm5 = nn.GroupNorm(32, num_channels=channels[4])

    self.conv5a = nn.Conv2d(channels[4], channels[4], 3, stride=1, bias=False, padding=1)
    self.t_mod5a = Dense(embed_dim, channels[4])
    self.gnorm5a = nn.GroupNorm(32, num_channels=channels[4])
    

    # Decoding layers where the resolution increases
   
    self.tconv5b = nn.Conv2d(channels[4], channels[4], 3, stride=1, bias=False, padding=1)     #  + channels[2]
    self.t_mod6b = Dense(embed_dim, channels[4])
    self.tgnorm5b = nn.GroupNorm(32, num_channels=channels[4])
    
    self.tconv5 = nn.ConvTranspose2d(2*channels[4], channels[3], 3, stride=2, bias=False, padding=1, output_padding=1)
    self.t_mod6 = Dense(embed_dim, channels[3])
    self.tgnorm5 = nn.GroupNorm(32, num_channels=channels[3])
    
    self.tconv4b = nn.Conv2d(2*channels[3], channels[3], 3, stride=1, bias=False, padding=1)     #  + channels[2]
    self.t_mod7b = Dense(embed_dim, channels[3])
    self.tgnorm4b = nn.GroupNorm(32, num_channels=channels[3])

    self.tconv4 = nn.ConvTranspose2d(2*channels[3], channels[2], 3, stride=2, bias=False, padding=1, output_padding=1)     #  + channels[2]
    self.t_mod7 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    
    self.tconv3b = nn.Conv2d(2*channels[2], channels[2], 3, stride=1, bias=False, padding=1)     #  + channels[2]
    self.t_mod8b = Dense(embed_dim, channels[2])
    self.tgnorm3b = nn.GroupNorm(32, num_channels=channels[2])
    
    self.tconv3 = nn.ConvTranspose2d(2*channels[2], channels[1], 3, stride=2, bias=False, padding=1, output_padding=1)     #  + channels[2]
    self.t_mod8 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    
    self.tconv2b = nn.Conv2d(2*channels[1], channels[1], 3, stride=1, bias=False, padding=1)     #  + channels[1]
    self.t_mod9b = Dense(embed_dim, channels[1])
    self.tgnorm2b = nn.GroupNorm(32, num_channels=channels[1])

    self.tconv2 = nn.ConvTranspose2d(2*channels[1], channels[0], 3, stride=2, bias=False, padding=1, output_padding=1)     #  + channels[1]
    self.t_mod9 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

    self.tconv1b = nn.Conv2d(2*channels[0], channels[0], 3, stride=1, bias=False, padding=1)     #  + channels[1]
    self.t_mod10b = Dense(embed_dim, channels[0])
    self.tgnorm1b = nn.GroupNorm(32, num_channels=channels[0])

    self.tconv1 = nn.ConvTranspose2d(2*channels[0], channels[0], 3, stride=2, bias=False, padding=1, output_padding=1)     #  + channels[1]
    self.t_mod10 = Dense(embed_dim, channels[0])
    self.tgnorm1 = nn.GroupNorm(32, num_channels=channels[0])
    
    self.tconv0 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1, padding=1, output_padding=0)

    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    # A restricted version of the `marginal_prob_std` function, after specifying a Lambda.
    self.marginal_prob_std = marginal_prob_std

  def forward(self, x, t, y=None):

    # Obtain the Gaussian random feature embedding for t
    embed = self.act(self.time_embed(t))
    # Encoding path, downsampling
    ## Incorporate information from t
    
    h1 = self.conv1(x)  + self.t_mod1(embed)
    h1 = self.act(self.gnorm1(h1))
    
    h1a = self.conv1a(h1) + self.t_mod1a(embed)
    h1a = self.act(self.gnorm1a(h1a))

    #  2nd conv
    h2 = self.conv2(h1a) + self.t_mod2(embed)
    h2 = self.act(self.gnorm2(h2))

    h2a = self.conv2a(h2) + self.t_mod2a(embed)
    h2a = self.act(self.gnorm2a(h2a))

      # 3rd conv
    h3 = self.conv3(h2a) + self.t_mod3(embed)
    h3 = self.act(self.gnorm3(h3))

    h3a = self.conv3a(h3) + self.t_mod3a(embed)
    h3a = self.act(self.gnorm3a(h3a))

    # 4th conv
    h4 = self.conv4(h3a) + self.t_mod4(embed)
    h4 = self.act(self.gnorm4(h4))

    h4a = self.conv4a(h4) + self.t_mod4a(embed)
    h4a = self.act(self.gnorm4a(h4a))

    # 5th conv 256 -> 512
    h5 = self.conv5(h4a) + self.t_mod5(embed)
    h5 = self.act(self.gnorm5(h5))

    h5a = self.conv5a(h5) + self.t_mod5a(embed)
    h5a = self.act(self.gnorm5a(h5a))

    # -----------------------------------
    # Decoding path up sampling
    h5b = self.tconv5b(h5a) + self.t_mod6b(embed)
    h5b = self.act(self.tgnorm5b(h5b))
    
    h = self.tconv5(torch.cat([h5, h5b], dim=1)) + self.t_mod6(embed)
    h = self.act(self.tgnorm5(h))
    

    h4b = self.tconv4b(torch.cat([h, h4a], dim=1)) + self.t_mod7b(embed)
    h4b = self.act(self.tgnorm4b(h4b))

    h = self.tconv4(torch.cat([h4,h4b], dim=1)) + self.t_mod7(embed)
    h = self.act(self.tgnorm4(h))

    h3b = self.tconv3b(torch.cat([h,h3a], dim=1)) + self.t_mod8b(embed)
    h3b = self.act(self.tgnorm3b(h3b))

    h = self.tconv3(torch.cat([h3, h3b], dim=1)) + self.t_mod8(embed)
    h = self.act(self.tgnorm3(h))

    h2b = self.tconv2b(torch.cat([h, h2a], dim=1)) + self.t_mod9b(embed)
    h2b = self.act(self.tgnorm2b(h2b))

    h = self.tconv2(torch.cat([h2b, h2], dim=1)) + self.t_mod9(embed)
    h = self.act(self.tgnorm2(h))

    h1b = self.tconv1b(torch.cat([h, h1a], dim=1)) + self.t_mod10b(embed)
    h1b = self.act(self.tgnorm1b(h))

    h = self.tconv1(torch.cat([h1b, h1], dim=1)) + self.t_mod10(embed)
    h = self.act(self.tgnorm1(h))

    h = self.tconv0(h)
    

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]

    return h
  


  def generate(self, image):
    """
    Generate an output from the input mask using the diffusion model.
    This method is compatible with the ModelService interface.
    
    Args:
      image: Input tensor of shape [batch_size, 4, 256, 256]
      
    Returns:
      output: Generated output tensor with shape [batch_size, 4, 256, 256]
    """
    with torch.no_grad():
      # Extract the actual mask from the input (3rd channel)
      mask = image[:, 2:3, :, :]
      
      # Set a fixed timestep for generation (typically 0 for best quality)
      batch_size = image.size(0)
      t = torch.ones(batch_size, device=image.device) * 0.5
      
      # Pass through the diffusion model
      out = self.forward(mask, t)
      
      # Ensure output has 4 channels like the expected format
      if out.size(1) != 4:
        # If fewer channels (e.g., only 1), duplicate to 4 channels
        out = out.repeat(1, 4, 1, 1)
      
      return out
