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


app = Flask(__name__)
CORS(app)

# Define model paths
ORIGINAL_MODEL_PATH = os.path.join('public', 'models', 'vae.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define VAE model structure for state dict loading
def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTranspose, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        base = 128
        
        self.encoder = nn.Sequential(
            Conv(4, base, 3, stride=2, padding=1),
            Conv(base, 2*base, 3, padding=1),
            Conv(2*base, 2*base, 3, stride=2, padding=1),
            Conv(2*base, 2*base, 3, padding=1),
            Conv(2*base, 2*base, 3, stride=2, padding=1),
            Conv(2*base, 4*base, 3, padding=1),
            Conv(4*base, 4*base, 3, stride=2, padding=1),
            Conv(4*base, 4*base, 3, padding=1),
            Conv(4*base, 4*base, 3, stride=2, padding=1),
            nn.Conv2d(4*base, 64*base, 8),
            nn.LeakyReLU()
        )
        
        self.encoder_mu = nn.Conv2d(64*base, 32*base, 1)
        self.encoder_logvar = nn.Conv2d(64*base, 32*base, 1)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(32*base, 64*base, 1),
            ConvTranspose(64*base, 4*base, 8),
            Conv(4*base, 4*base, 3, padding=1),
            ConvTranspose(4*base, 4*base, 4, stride=2, padding=1),
            Conv(4*base, 4*base, 3, padding=1),
            ConvTranspose(4*base, 4*base, 4, stride=2, padding=1),
            Conv(4*base, 2*base, 3, padding=1),
            ConvTranspose(2*base, 2*base, 4, stride=2, padding=1),
            Conv(2*base, 2*base, 3, padding=1),
            ConvTranspose(2*base, 2*base, 4, stride=2, padding=1),
            Conv(2*base, base, 3, padding=1),
            ConvTranspose(base, base, 4, stride=2, padding=1),
            nn.Conv2d(base, 4, 3, padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Initialize the model
model = None
model_info = {
    'source': None,
    'type': None
}

print(f"PyTorch version: {torch.__version__}")
# Use a safer way to get Flask version
try:
    import flask
    print(f"Flask version: {flask.__version__}")
except (ImportError, AttributeError):
    print("Could not determine Flask version")

# Try to load the model, with fallbacks
try:
    # First, try to load the original model
    checkpoint = torch.load(ORIGINAL_MODEL_PATH, map_location=device)
    model = VAE()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if model is not None:
        model_info['source'] = 'original'
        model_info['type'] = 'full_model'
    
    # If we still don't have a model, create one on the fly
    if model is None:
        print("Creating a new model instance...")
        model = VAE()
        model.eval()
        model_info['source'] = 'generated'
        model_info['type'] = 'empty_model'
    
    # Test the model
    print("Testing model inference...")
    test_input = torch.zeros((1, 4, 256, 256), device=device)
    with torch.no_grad():
        try:
            mu, sigma = model.encode(test_input)
            z = mu + sigma * 0.05
            test_output = model.decode(z)
            if isinstance(test_output, tuple):
                print(f"Model output is a tuple with shapes: {[o.shape for o in test_output]}")
            else:
                print(f"Model output shape: {test_output.shape}")
            print(f"Using model: source={model_info['source']}, type={model_info['type']}")
        except Exception as e:
            print(f"Model inference test failed: {e}")
            print("Creating a new model as a fallback...")
            model = VAE()
            model.eval()
            model_info['source'] = 'generated'
            model_info['type'] = 'empty_model'
    
except Exception as e:
    print(f"Error during model loading process: {e}")
    # Create an empty model as a last resort
    try:
        model = VAE()
        model.eval()
        model_info['source'] = 'generated'
        model_info['type'] = 'empty_model'
    except Exception as e2:
        print(f"Failed to create empty model: {e2}")
        model = None

def preprocess_image(image_data):
    """Convert base64 image data to a tensor for model input"""
    try:
        # Strip the base64 prefix if it exists
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Open as PIL Image
        image = Image.open(io.BytesIO(image_bytes))     
        image.show()
        
        # Convert to grayscale if the image is not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 256x256 if needed
        if image.size != (256, 256):
            image = image.resize((256, 256))    


        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Normalize to [0, 1]
        image_np = image_np / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0)
        tensor = torch.cat((tensor, tensor, tensor, tensor), dim=1)
        
        return tensor.to(device)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def tensor_to_base64(tensor):
    """Convert a tensor to a base64 encoded image"""
    try:
        # The model outputs 4 channel image, we need to convert it properly
        # If tensor is a tuple, get the first element
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        
        # Make sure we're working with the tensor on CPU
        tensor = tensor.cpu().detach()
        
        # Get the first channel (or average the channels if needed)
        if tensor.shape[1] == 4:  # If it has 4 channels
            # Use the first channel for simplicity
            image_tensor = tensor[:, 0:1, :, :]
        else:
            image_tensor = tensor
        
        # Remove batch dimension and convert to numpy
        image_np = image_tensor.squeeze(0).squeeze(0).numpy()
        
        # Scale to [0, 255] and convert to uint8
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        
        # Create a PIL Image directly
        pil_img = Image.fromarray(image_np, mode='L')
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        
        # Convert to base64
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error converting tensor to base64: {e}")
        print(f"Tensor shape: {tensor.shape}, type: {tensor.dtype}")
        # Try an alternate approach for debugging
        try:
            # Create a solid gray image as a fallback
            fallback_img = Image.new('L', (256, 256), 128)
            buffer = io.BytesIO()
            fallback_img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{img_str}"
        except:
            return None

@app.route('/api/generate', methods=['POST'])
def generate():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs for details.'}), 500
    
    try:
        data = request.get_json(silent=True)  # Safe way to get JSON
        if data is None:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image']
        model_type = data.get('model', 'vae')
        
        # Preprocess the image
        mask = preprocess_image(image_data)
        if mask is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        # Run the model
        with torch.no_grad():
            mu1, sigma1 = model.encode(mask)
            alpha = 0.01
            z = mu1 + alpha * sigma1
            output = model.decode(z)
            output = output[0, 0]
            print(output.shape)
        
        # Convert back to base64
        result_img = tensor_to_base64(output)
        if result_img is None:
            return jsonify({'error': 'Failed to convert result to image'}), 500
        
        return jsonify({
            'result': result_img,
            'model': model_type,
            'model_info': model_info
        })
    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({'error': str(e)}), 500

# Add a health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        import flask
        flask_version = flask.__version__
    except (ImportError, AttributeError):
        flask_version = "unknown"
        
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'flask_version': flask_version,
        'torch_version': torch.__version__,
        'model_info': model_info if model is not None else None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 