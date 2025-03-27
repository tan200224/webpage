import os
import io
import base64
import numpy as np
from PIL import Image
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from matplotlib import pyplot as plt
import torch.nn as nn
import torchvision.transforms.functional as TF

from models.VAE import VAE
from models.GAN import GAN
from models.Diffusion import Diffusion

# Define model paths - using relative paths and checking if models exist
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "weights")

# Ensure model weights directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Define default model paths
DEFAULT_VAE_MODEL_PATH = os.path.join(MODELS_DIR, "vae_model.pt")
DEFAULT_GAN_MODEL_PATH = os.path.join(MODELS_DIR, "gan_model.pt")
DEFAULT_DIFFUSION_MODEL_PATH = os.path.join(MODELS_DIR, "diffusion_model.pt")

# Define paths to models, allow overriding with environment variables
ORIGINAL_VAE_MODEL_PATH = r"D:\LumenResearchDataBase\Project\selfCoding\mask2pic_64model_47.pt"
ORIGINAL_GAN_MODEL_PATH = os.environ.get('GAN_MODEL_PATH', DEFAULT_GAN_MODEL_PATH) 
ORIGINAL_DIFFUSION_MODEL_PATH = os.environ.get('DIFFUSION_MODEL_PATH', DEFAULT_DIFFUSION_MODEL_PATH)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_info = {
            'source': None,
            'type': None
        }
        self.model_type = None
        
        # Initialize with the default VAE model for backward compatibility
        self.load_model('vae')

    def load_model(self, model_type):

        self.model_type = model_type

        try:
            if model_type == 'vae':
                self.model = VAE().to(self.device)
                if os.path.exists(ORIGINAL_VAE_MODEL_PATH):
                    checkpoint = torch.load(ORIGINAL_VAE_MODEL_PATH)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model_info['source'] = 'original'
                else:
                    print(f"Warning: VAE model file not found at {ORIGINAL_VAE_MODEL_PATH}. Using untrained model.")
                    self.model_info['source'] = 'untrained'
                self.model_info['type'] = 'vae'


            elif model_type == 'gan':
                self.model = GAN().to(self.device)
                if os.path.exists(ORIGINAL_GAN_MODEL_PATH):
                    checkpoint = torch.load(ORIGINAL_GAN_MODEL_PATH)
                    self.model.load_state_dict(checkpoint)
                    self.model_info['source'] = 'original'
                else:
                    print(f"Warning: GAN model file not found at {ORIGINAL_GAN_MODEL_PATH}. Using untrained model.")
                    self.model_info['source'] = 'untrained'
                self.model_info['type'] = 'gan'



            elif model_type == 'diffusion':
                self.model = Diffusion().to(self.device)
                if os.path.exists(ORIGINAL_DIFFUSION_MODEL_PATH):
                    checkpoint = torch.load(ORIGINAL_DIFFUSION_MODEL_PATH)
                    self.model.load_state_dict(checkpoint)
                    self.model_info['source'] = 'original'
                else:
                    print(f"Warning: Diffusion model file not found at {ORIGINAL_DIFFUSION_MODEL_PATH}. Using untrained model.")
                    self.model_info['source'] = 'untrained'
                self.model_info['type'] = 'diffusion'
            else:
                raise ValueError(f"Invalid model type: {model_type}")
            
            # Set model to evaluation mode
            self.model.cuda()
           
            print(f"Successfully loaded model: {model_type}")
            
        except Exception as e:
            print(f"Error loading model {model_type}: {e}")
            raise

    def generate(self, image, device=None):
        """
        Generate an output using the currently loaded model.
        
        Args:
            image (torch.Tensor): Input tensor of shape [batch_size, 4, 256, 256]
            device (torch.device, optional): Device to run generation on. Defaults to self.device.
            
        Returns:
            torch.Tensor: Generated output tensor with the same shape
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model first.")
        
        # Use the provided device or fall back to self.device
        device_to_use = device if device is not None else self.device
        
        # Ensure the model and input are on the same device
        model = self.model.to(device_to_use)
        image = image.to(device_to_use)
        
        with torch.no_grad():
            # For VAE models, generate typically just needs the image
            if self.model_type == 'vae':
                output = model.generate(image)
            # For models that might need the device parameter explicitly
            elif self.model_type in ['gan', 'diffusion']:
                # Check if the model's generate method accepts a device parameter
                import inspect
                sig = inspect.signature(model.generate)
                if 'device' in sig.parameters:
                    output = model.generate(image, device_to_use)
                else:
                    output = model.generate(image)
            else:
                output = model.generate(image)
                
        return output

if __name__ == '__main__':
    model_service = ModelService()
    model_service.load_model('vae')
    print(f"Model loaded: {model_service.model_type}")

