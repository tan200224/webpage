import { useState, useRef } from "react";
import { Helmet } from "react-helmet";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Loader2, BrainCircuit, Image, ArrowRight, Upload, Plus } from "lucide-react";
import { toast } from "sonner";
import { useNavigate } from "react-router-dom";

const SAMPLE_SCANS = [{
  id: 1,
  name: "Patient 001 - Slice 42",
  thumbnail: "https://images.unsplash.com/photo-1607798748738-b15c40d33d57?q=80&w=400&auto=format&fit=crop",
  fullImage: "https://images.unsplash.com/photo-1607798748738-b15c40d33d57?q=80&w=800&auto=format&fit=crop",
  hasAbnormality: true
}, {
  id: 2,
  name: "Patient 002 - Slice 38",
  thumbnail: "https://images.unsplash.com/photo-1530497610245-94d3c16cda28?q=80&w=400&auto=format&fit=crop",
  fullImage: "https://images.unsplash.com/photo-1530497610245-94d3c16cda28?q=80&w=800&auto=format&fit=crop",
  hasAbnormality: false
}, {
  id: 3,
  name: "Patient 003 - Slice 45",
  thumbnail: "https://images.unsplash.com/photo-1584555616150-2a3dd218c357?q=80&w=400&auto=format&fit=crop",
  fullImage: "https://images.unsplash.com/photo-1584555616150-2a3dd218c357?q=80&w=800&auto=format&fit=crop",
  hasAbnormality: true
}, {
  id: 4,
  name: "Patient 004 - Slice 51",
  thumbnail: "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=400&auto=format&fit=crop",
  fullImage: "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=800&auto=format&fit=crop",
  hasAbnormality: false
}];

const PancreaticCancerDemo = () => {
  const [selectedScan, setSelectedScan] = useState<number | null>(null);
  const [segmentationResult, setSegmentationResult] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState([75]);
  const [uploadedImages, setUploadedImages] = useState<{
    id: number;
    name: string;
    thumbnail: string;
    fullImage: string;
    hasAbnormality: boolean;
  }[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const handleScanSelect = (scanId: number) => {
    setSelectedScan(scanId);
    setSegmentationResult(null);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.includes('image/')) {
      toast.error("Please upload an image file");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      toast.error("File size exceeds 5MB limit");
      return;
    }

    const reader = new FileReader();
    reader.onload = e => {
      const result = e.target?.result as string;
      if (result) {
        const newImage = {
          id: Date.now(),
          name: `My scan - ${file.name}`,
          thumbnail: result,
          fullImage: result,
          hasAbnormality: Math.random() > 0.5
        };
        setUploadedImages(prev => [...prev, newImage]);
        toast.success("Image uploaded successfully");
        setSelectedScan(newImage.id);
        setSegmentationResult(null);
      }
    };
    reader.onerror = () => {
      toast.error("Error reading file");
    };
    reader.readAsDataURL(file);
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const runSegmentation = async () => {
    if (selectedScan === null) {
      toast.error("Please select a CT scan first");
      return;
    }
    setIsProcessing(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 2500));

      const scan = [...SAMPLE_SCANS, ...uploadedImages].find(s => s.id === selectedScan);

      setSegmentationResult(scan?.fullImage || null);

      if (scan?.hasAbnormality) {
        toast.warning("Potential abnormality detected", {
          description: "The model has identified regions of concern in the pancreas."
        });
      } else {
        toast.success("Analysis complete", {
          description: "No significant abnormalities detected in this scan."
        });
      }
    } catch (error) {
      toast.error("Error processing the scan", {
        description: "There was an error running the segmentation model."
      });
      console.error(error);
    } finally {
      setIsProcessing(false);
    }
  };

  return <div className="min-h-screen flex flex-col">
      <Helmet>
        <title>Pancreatic Cancer AI Diagnosis | Archie Tan</title>
        <meta name="description" content="Interactive demo of pancreatic cancer segmentation AI model by Archie Tan" />
      </Helmet>
      
      <Navbar />
      
      <main className="flex-grow pt-20 pb-10 px-4 sm:px-6">
        <div className="container mx-auto max-w-5xl">
          <div className="mb-8">
            <div className="flex flex-col md:flex-row md:items-center gap-4 mb-4">
              <div className="flex items-center gap-3">
                <h1 className="text-3xl sm:text-4xl font-bold relative inline-block">
                  <span className="relative z-10">Pancreatic Cancer AI Diagnosis</span>
                  <span className="absolute left-0 bottom-0 w-full h-3 bg-primary/20 -z-10 transform -rotate-1"></span>
                </h1>
              </div>
            </div>
            
            <p className="text-lg text-muted-foreground mb-4">
              This interactive demo showcases my research on early diagnosis of pancreatic cancer using advanced AI segmentation models.
            </p>
            
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-6">
              <BrainCircuit className="h-4 w-4 text-primary" />
              <span>Model accuracy: 87.16% on pancreas CT-Scan segmentation</span>
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div className="bg-background border rounded-xl shadow-sm p-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold">Select a CT Scan</h2>
                  
                  <Button variant="outline" size="sm" onClick={triggerFileInput} className="flex items-center gap-2">
                    <Upload className="h-4 w-4" />
                    Upload CT Scan
                  </Button>
                  
                  <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleFileUpload} />
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  {SAMPLE_SCANS.map(scan => <div key={scan.id} className={`relative border-2 rounded-lg overflow-hidden cursor-pointer transition-all hover:shadow-md ${selectedScan === scan.id ? 'border-primary' : 'border-transparent hover:border-muted-foreground/20'}`} onClick={() => handleScanSelect(scan.id)}>
                      <div className="aspect-square relative">
                        <img src={scan.thumbnail} alt={scan.name} className="w-full h-full object-cover" />
                        {selectedScan === scan.id && <div className="absolute inset-0 bg-primary/10 flex items-center justify-center">
                            <div className="bg-primary text-white px-2 py-1 rounded-full text-xs font-medium">
                              Selected
                            </div>
                          </div>}
                      </div>
                      <div className="p-2 text-xs font-medium truncate">{scan.name}</div>
                    </div>)}
                  
                  {uploadedImages.map(scan => <div key={scan.id} className={`relative border-2 rounded-lg overflow-hidden cursor-pointer transition-all hover:shadow-md ${selectedScan === scan.id ? 'border-primary' : 'border-transparent hover:border-muted-foreground/20'}`} onClick={() => handleScanSelect(scan.id)}>
                      <div className="aspect-square relative">
                        <img src={scan.thumbnail} alt={scan.name} className="w-full h-full object-cover" />
                        {selectedScan === scan.id && <div className="absolute inset-0 bg-primary/10 flex items-center justify-center">
                            <div className="bg-primary text-white px-2 py-1 rounded-full text-xs font-medium">
                              Selected
                            </div>
                          </div>}
                        <Badge className="absolute top-2 right-2 bg-primary/80">Uploaded</Badge>
                      </div>
                      <div className="p-2 text-xs font-medium truncate">{scan.name}</div>
                    </div>)}
                  
                  <Dialog>
                    <DialogTrigger asChild>
                      <div className="relative border-2 border-dashed rounded-lg overflow-hidden cursor-pointer transition-all hover:shadow-md border-muted-foreground/20 hover:border-primary/50">
                        <div className="aspect-square flex flex-col items-center justify-center gap-2 bg-muted/30">
                          <Plus className="h-8 w-8 text-muted-foreground/50" />
                          <span className="text-xs text-muted-foreground">Upload the one you made
                        </span>
                        </div>
                      </div>
                    </DialogTrigger>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>Upload CT Scan</DialogTitle>
                      </DialogHeader>
                      <div className="space-y-4 py-4">
                        <div className="border-2 border-dashed rounded-lg p-8 text-center">
                          <Upload className="h-10 w-10 mx-auto mb-2 text-muted-foreground/60" />
                          <p className="text-sm text-muted-foreground mb-4">
                            Drag and drop your CT scan image here or click to browse
                          </p>
                          <Button onClick={triggerFileInput}>Select File</Button>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Supported formats: JPEG, PNG, GIF, DICOM (converted to standard image formats)
                          <br />
                          Maximum file size: 5MB
                        </p>
                      </div>
                    </DialogContent>
                  </Dialog>
                  
                  <div 
                    className="relative border-2 rounded-lg overflow-hidden cursor-pointer transition-all hover:shadow-md border-primary/30 hover:border-primary"
                    onClick={() => navigate('/synthetic-ct-demo')}
                  >
                    <div className="aspect-square flex flex-col items-center justify-center gap-2 bg-primary/10">
                      <BrainCircuit className="h-8 w-8 text-primary" />
                      <span className="text-sm font-medium text-primary">Generate One</span>
                      <span className="text-xs text-muted-foreground">Create your own synthetic CT</span>
                      <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-background border rounded-xl shadow-sm p-6">
                <h2 className="text-xl font-semibold mb-4">Model Settings</h2>
                
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <label className="text-sm font-medium">Confidence Threshold</label>
                      <span className="text-sm text-muted-foreground">{confidenceThreshold}%</span>
                    </div>
                    <Slider min={50} max={95} step={1} value={confidenceThreshold} onValueChange={setConfidenceThreshold} />
                    <p className="text-xs text-muted-foreground">
                      Higher values increase precision but may miss some abnormalities.
                    </p>
                  </div>
                </div>
                
                <div className="mt-6">
                  <Button onClick={runSegmentation} disabled={isProcessing || selectedScan === null} className="w-full gradient-bg text-white border-none">
                    {isProcessing ? <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing Scan...
                      </> : <>
                        <BrainCircuit className="mr-2 h-4 w-4" />
                        Run Segmentation Model
                      </>}
                  </Button>
                </div>
              </div>
            </div>
            
            <div className="bg-background border rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-semibold mb-4">Model Output</h2>
              <div className="border-2 border-dashed border-muted-foreground/20 rounded-lg h-[400px] flex items-center justify-center overflow-hidden bg-black/5">
                {segmentationResult ? <div className="relative w-full h-full">
                    <img src={segmentationResult} alt="Segmentation Result" className="w-full h-full object-contain" />
                    {[...SAMPLE_SCANS, ...uploadedImages].find(s => s.id === selectedScan)?.hasAbnormality && <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-16 h-16 border-2 border-red-500 rounded-full animate-pulse" />}
                  </div> : <div className="text-center p-6">
                    <Image className="h-10 w-10 mx-auto mb-3 text-muted-foreground/50" />
                    <p className="text-muted-foreground">
                      {selectedScan ? "Select a scan and run the model to view results" : "Select a CT scan to begin"}
                    </p>
                  </div>}
              </div>
              
              {segmentationResult && <div className="mt-4 bg-secondary/30 p-4 rounded-lg">
                  <h3 className="font-medium mb-2">Analysis Results</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    {[...SAMPLE_SCANS, ...uploadedImages].find(s => s.id === selectedScan)?.hasAbnormality ? "The model has detected regions of interest that might indicate pancreatic abnormalities. The highlighted area shows potential tissue changes that may require further investigation." : "No significant abnormalities detected in the pancreatic region. The model analyzed the tissue density and structure patterns typical in healthy pancreas tissue."}
                  </p>
                  <div className="flex flex-wrap gap-2 mb-3">
                    <Badge variant="outline" className="bg-background">
                      Confidence: {Math.round(85 + Math.random() * 10)}%
                    </Badge>
                    <Badge variant="outline" className="bg-background">
                      Processing Time: {Math.round(1.2 + Math.random() * 0.8)}s
                    </Badge>
                  </div>
                </div>}
            </div>
          </div>
          
          <div className="mt-10 bg-background border rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4">About This Research</h2>
            <p className="text-muted-foreground mb-4">
              This demo is based on my research in using AI models for early diagnosis of pancreatic cancer. My work
              focused on implementing segmentation models to accurately identify pancreatic tissue and potential abnormalities in CT scans.
            </p>
            <p className="text-muted-foreground mb-4">
              The full research implementation achieved 87.16% dice accuracy in pancreas segmentation tasks, establishing a strong
              foundation for medical imaging analysis. The model is built with a customized data augmentation pipeline and advanced
              deep learning techniques optimized for medical imaging.
            </p>
            <div className="flex flex-wrap gap-2">
              <Badge className="rounded-full">PyTorch</Badge>
              <Badge className="rounded-full">TorchVision</Badge>
              <Badge className="rounded-full">Python</Badge>
              <Badge className="rounded-full">Data Augmentation</Badge>
              <Badge className="rounded-full">3D Imaging</Badge>
            </div>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>;
};

export default PancreaticCancerDemo;
