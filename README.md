# ArchieProflio Synthetic CT Generator

This application generates synthetic CT scans from user-drawn masks using different AI models (VAE, GAN, and Diffusion).

## Project Structure

The project consists of two main parts:
- **Backend**: A Flask API that serves the AI models
- **Frontend**: A React application with a drawing interface for creating masks

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ArchieProflio.git
cd ArchieProflio
```

### 2. Install Dependencies
Run the installation script to install all required dependencies:
```bash
install.bat
```

This will install all required Python packages and Node.js modules.

### 3. Run the Application
Start the application using the provided script:
```bash
start_project.bat
```

This will start both the backend and frontend servers.

### 4. Access the Application
- Frontend: http://localhost:5173
- Backend API: http://localhost:5000

## Manual Setup

If you prefer to set up the application manually, follow these steps:

### Backend Setup
```bash
cd BackEnd
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
cd FrontEnd
npm install
npm run dev
```

## Available Models

The application supports the following models:

1. **VAE (Variational Autoencoder)**: A generative model that encodes masks into a latent space and decodes them into CT scans.
2. **GAN (Generative Adversarial Network)**: A model that uses a generator and discriminator to create realistic CT scans.
3. **Diffusion**: A model based on diffusion probabilistic models that gradually transforms noise into a CT scan.

## API Endpoints

- `GET /api/health`: Check the health status of the backend
- `POST /api/generate`: Generate a CT scan from a mask, specifying the model type

## Model Configuration

The model paths are configured in `BackEnd/service/ModelService.py`. You can customize the paths by:

1. Setting environment variables:
   - `VAE_MODEL_PATH`
   - `GAN_MODEL_PATH`
   - `DIFFUSION_MODEL_PATH`

2. Directly modifying the default paths in the code.

## Troubleshooting

If you encounter issues:

1. Check the console output for any error messages
2. Ensure all dependencies are installed correctly
3. Verify that the model paths in `ModelService.py` are correct
4. If using CUDA, ensure your GPU is compatible with PyTorch

## Development

### Project Structure
```
ArchieProflio/
├── BackEnd/
│   ├── controllers/       # API routes and request handling
│   ├── models/            # Model definitions and implementations
│   ├── service/           # Business logic and model service
│   ├── main.py            # Entry point for the backend
│   └── requirements.txt   # Python dependencies
├── FrontEnd/
│   ├── src/               # React source code
│   ├── public/            # Static assets
│   └── package.json       # Node.js dependencies
├── install.bat            # Installation script
└── start_project.bat      # Startup script
```

### Adding New Models

To add a new model:

1. Create a new model class in the `BackEnd/models/` directory
2. Implement the `generate` method that takes an input tensor and returns an output tensor
3. Update the `ModelService.py` file to support loading and using the new model
4. Add the new model type to the frontend UI 