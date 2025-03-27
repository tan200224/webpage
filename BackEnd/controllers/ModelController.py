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

from service.ModelService import ModelService

# Initialize the model service
model_service = ModelService()

ModelController = Flask(__name__)
CORS(ModelController)

print(f"PyTorch version: {torch.__version__}")
# Use a safer way to get Flask version
try:
    import flask
    print(f"Flask version: {flask.__version__}")
except (ImportError, AttributeError):
    print("Could not determine Flask version")

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
        
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 256x256 if needed
        if image.size != (256, 256):
            image = image.resize((256, 256))    
        
        # Convert to numpy array
        image_np = np.array(image)

        # Convert to tensor
        tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0)
        blank = torch.zeros((1, 1, 256, 256))
        tensor = torch.cat((blank, blank, tensor, tensor), dim=1)
        
        return tensor.to(model_service.device)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def tensor_to_base64(tensor):
    """Convert a tensor to a base64 encoded image"""
    try:
        # The model outputs 4 channel image, we need to convert it properly
        if isinstance(tensor, tuple):
            tensor = tensor[0]
            
        # Ensure the tensor is on CPU and detached from computation graph
        tensor = tensor.detach().cpu()
        
        # Take the first channel (or handle multi-channel output)
        if tensor.dim() == 4:  # [batch, channels, height, width]
            # If it's a batch, just use the first image
            if tensor.size(0) > 1:
                tensor = tensor[0]
                
            # If we have multiple channels, use the first one
            if tensor.size(0) > 1:
                tensor = tensor[0]

        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.imsave(buffer, tensor.numpy(), cmap='gray', format='PNG')
        
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
        except Exception as inner_e:
            print(f"Error creating fallback image: {inner_e}")
            return None

@ModelController.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json(silent=True)  # Safe way to get JSON
        if data is None:
            return jsonify({'error': 'Invalid JSON data'}), 400
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        

        image_data = data['image']
        model_type = data.get('model', 'vae')  # Default to 'vae' if not specified
        
        print(f"Client requested model: {model_type}")
        
        # Load the model based on the type
        try:
            model_service.load_model(model_type)
            print(f"Model loaded successfully: {model_type}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return jsonify({'error': f'Failed to load model: {str(e)}'}), 500
        
        # Preprocess the image
        mask = preprocess_image(image_data)
        if mask is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        # Run the model
        try:
            output = model_service.generate(mask)
            print(f"Generated output with shape: {output.shape}")
        except Exception as e:
            print(f"Error during generation: {e}")
            return jsonify({'error': f'Failed to generate image: {str(e)}'}), 500

        # Convert back to base64
        result_img = tensor_to_base64(output)
        if result_img is None:
            return jsonify({'error': 'Failed to convert result to image'}), 500
        
        return jsonify({
            'result': result_img,
            'model': model_type,
            'model_info': model_service.model_info
        })
    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({'error': str(e)}), 500

# Add a health check endpoint
@ModelController.route('/api/health', methods=['GET'])
def health_check():
    try:
        import flask
        flask_version = flask.__version__
    except (ImportError, AttributeError):
        flask_version = "unknown"
        
    return jsonify({
        'status': 'ok',
        'model_loaded': model_service.model is not None,
        'model_type': model_service.model_type,
        'flask_version': flask_version,
        'torch_version': torch.__version__,
        'model_info': model_service.model_info
    })

if __name__ == '__main__':
    ModelController.run(host='0.0.0.0', port=5000, debug=True)
