# VAE Model Backend Integration

This backend integration connects the SyntheticCTDemo React frontend with a PyTorch VAE model for generating synthetic CT scans from drawn masks.

## Setup Instructions

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the Flask backend:
   ```
   python app.py
   ```

3. Ensure the React frontend is configured to connect to the backend:
   - The API URL is set in `src/pages/SyntheticCTDemo.tsx`
   - By default, it connects to `http://localhost:5000`

## API Endpoints

### Generate CT Scan

**Endpoint:** `POST /api/generate`

**Request Body:**
```json
{
  "image": "base64-encoded-image-data",
  "model": "vae"
}
```

**Response:**
```json
{
  "result": "base64-encoded-generated-image",
  "model": "vae"
}
```

## Model Information

The VAE model is loaded from `public/models/vae.pth`. It expects:
- Input: Grayscale image of size 256x256
- Output: Generated CT scan image

## Error Handling

If the model fails to generate an image, the frontend will display an error message and fall back to using placeholder images for demonstration purposes.

## Troubleshooting

1. If you encounter CORS issues, ensure the Flask server has CORS properly enabled
2. Check that the model file exists at the correct path
3. Verify the PyTorch version is compatible with the saved model 