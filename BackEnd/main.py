"""
Main entry point for the ArchieProflio Synthetic CT Generator Backend
This file serves as the launcher for the ModelController Flask application
"""

import os
import sys
from controllers.ModelController import ModelController

def main():
    """Main function to start the Flask server"""
    print("Starting ArchieProflio Synthetic CT Generator Backend...")
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Set any environment variables or configuration here if needed
    # os.environ['VAE_MODEL_PATH'] = '/path/to/custom/model.pt'
    
    # Run the Flask application
    ModelController.run(
        host='0.0.0.0',  # Make the server publicly available
        port=5000,       # Run on port 5000
        debug=True       # Enable debug mode for development
    )

if __name__ == "__main__":
    # Add the current directory to path to ensure imports work
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        import torch
        main()
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1) 