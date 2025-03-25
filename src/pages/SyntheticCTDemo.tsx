import { useState } from "react";
import { Helmet } from "react-helmet";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import DrawingCanvas from "@/components/DrawingCanvas";
import ModelSelector, { ModelType } from "@/components/ModelSelector";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, BrainCircuit, Download } from "lucide-react";
import { toast } from "sonner";
import { useNavigate } from "react-router-dom";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

// API URL - adjust this based on your deployment configuration
const API_URL = "http://localhost:5000";

const SyntheticCTDemo = () => {
  const [mask, setMask] = useState<ImageData | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelType>("vae");
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const navigate = useNavigate();

  const handleMaskCreated = (maskData: ImageData) => {
    setMask(maskData);
    setGeneratedImage(null);
  };
  
  const handleModelSelected = (model: ModelType) => {
    setSelectedModel(model);
    setGeneratedImage(null);
  };

  const generateCTScan = async () => {
    if (!mask) {
      toast.error("Please draw a mask first");
      return;
    }
    
    setIsGenerating(true);
    
    try {
      // First check if the backend is available
      try {
        const healthResponse = await fetch(`${API_URL}/api/health`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        
        if (!healthResponse.ok) {
          throw new Error("Backend server is not responding");
        }
        
        const healthData = await healthResponse.json();
        if (!healthData.model_loaded) {
          throw new Error("Backend model is not loaded properly");
        }
      } catch (healthError) {
        console.error("Backend health check failed:", healthError);
        throw new Error("Cannot connect to the backend server. Make sure it's running.");
      }
      
      // Create a temporary canvas to get the base64 image
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = mask.width;
      tempCanvas.height = mask.height;
      const tempCtx = tempCanvas.getContext('2d');
      
      if (!tempCtx) {
        throw new Error("Failed to create temporary canvas context");
      }
      
      tempCtx.putImageData(mask, 0, 0);
      const base64Image = tempCanvas.toDataURL('image/png');
      
      // Call the backend API
      const response = await fetch(`${API_URL}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image,
          model: selectedModel
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to generate image");
      }
      
      const data = await response.json();
      setGeneratedImage(data.result);
      toast.success(`CT Scan generated using ${selectedModel.toUpperCase()} model`);
    } catch (error) {
      console.error("Error generating CT scan:", error);
      toast.error(`Failed to generate CT Scan: ${error instanceof Error ? error.message : 'Unknown error'}`);
      
      // Fallback to placeholders for demo purposes only
      const placeholderImages = {
        vae: "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?q=80&w=2670&auto=format&fit=crop",
        gan: "https://images.unsplash.com/photo-1530497610245-94d3c16cda28?q=80&w=2400&auto=format&fit=crop",
        diffusion: "https://images.unsplash.com/photo-1584555616150-2a3dd218c357?q=80&w=2564&auto=format&fit=crop"
      };
      
      setGeneratedImage(placeholderImages[selectedModel]);
      toast.info("Using placeholder image for demo purposes");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Helmet>
        <title>Synthetic CT Scan Generator | Archie Tan</title>
        <meta name="description" content="Create synthetic CT scans using AI models built by Archie Tan" />
      </Helmet>
      
      <Navbar />
      
      <main className="flex-grow pt-20 pb-10 px-4 sm:px-6">
        <div className="container mx-auto max-w-5xl">
          <div className="mb-8">
            <div className="flex flex-col md:flex-row md:items-center gap-4 mb-4">
              <div className="flex items-center gap-3">
                <h1 className="text-3xl sm:text-4xl font-bold relative inline-block">
                  <span className="relative z-10">Synthetic CT Scan Generator</span>
                  <span className="absolute left-0 bottom-0 w-full h-3 bg-primary/20 -z-10 transform -rotate-1"></span>
                </h1>
              </div>
            </div>
            
            <p className="text-lg text-muted-foreground mb-4">
              Draw a segmentation mask and generate a realistic synthetic CT scan using different AI models.
            </p>
            
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <BrainCircuit className="h-4 w-4 text-primary" />
              <span>Based on my research in medical imaging and AI-generated synthetic data</span>
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div className="bg-background border rounded-xl shadow-sm p-6">
                <h2 className="text-xl font-semibold mb-4">Draw Segmentation Mask</h2>
                <DrawingCanvas width={256} height={256} onMaskCreated={handleMaskCreated} />
              </div>
              
              <div className="bg-background border rounded-xl shadow-sm p-6">
                <h2 className="text-xl font-semibold mb-4">Model Settings</h2>
                <ModelSelector onModelSelect={handleModelSelected} />
                
                <div className="mt-6">
                  <Button 
                    onClick={generateCTScan} 
                    disabled={isGenerating || !mask}
                    className="w-full gradient-bg text-white border-none"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      'Generate CT Scan'
                    )}
                  </Button>
                </div>
              </div>
            </div>
            
            <div className="bg-background border rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-semibold mb-4">Generated CT Scan</h2>
              <div className="border-2 border-dashed border-muted-foreground/20 rounded-lg h-[400px] flex items-center justify-center overflow-hidden bg-black/5">
                {generatedImage ? (
                  <img 
                    src={generatedImage} 
                    alt="Generated CT Scan" 
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-center p-6">
                    <BrainCircuit className="h-10 w-10 mx-auto mb-3 text-muted-foreground/50" />
                    <p className="text-muted-foreground">
                      {mask ? "Draw a mask and click Generate to see the result" : "Draw a segmentation mask to begin"}
                    </p>
                  </div>
                )}
              </div>
              
              {generatedImage && (
                <div className="mt-4 bg-secondary/30 p-4 rounded-lg">
                  <h3 className="font-medium mb-2">Model: {selectedModel.toUpperCase()}</h3>
                  <p className="text-sm text-muted-foreground">
                    This synthetic CT scan was generated based on your drawn mask using the {selectedModel === "vae" ? "Variational Autoencoder" : 
                    selectedModel === "gan" ? "Generative Adversarial Network" : "Diffusion Model"}.
                  </p>
                  <div className="mt-3">
                    <Button 
                      variant="outline" 
                      size="sm"
                      disabled={isGenerating}
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = generatedImage;
                        link.download = `synthetic-ct-scan-${selectedModel}.jpg`;
                        link.click();
                        toast.success("CT Scan image downloaded");
                      }}
                    >
                      <Download size={16} className="mr-1" />
                      Download Result
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          <div className="mt-10 bg-background border rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4">About This Project</h2>
            <p className="text-muted-foreground mb-4">
              This demo is based on my research in building synthetic medical imaging generators. The real model uses advanced
              deep learning techniques to create realistic CT scans from segmentation masks, which can be used for training
              medical image analysis models without requiring real patient data.
            </p>
            <p className="text-muted-foreground mb-4">
              My full research implementation achieved 87.16% accuracy in pancreas segmentation tasks and includes
              custom data augmentation techniques and a complete machine learning pipeline for medical imaging applications.
            </p>
            <div className="flex flex-wrap gap-2">
              <Badge className="rounded-full">PyTorch</Badge>
              <Badge className="rounded-full">TorchVision</Badge>
              <Badge className="rounded-full">3D Imaging</Badge>
              <Badge className="rounded-full">Data Augmentation</Badge>
              <Badge className="rounded-full">Medical AI</Badge>
            </div>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default SyntheticCTDemo;
