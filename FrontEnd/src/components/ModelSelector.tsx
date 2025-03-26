
import { useState } from "react";
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";

export type ModelType = "vae" | "gan" | "diffusion";

interface ModelSelectorProps {
  onModelSelect: (model: ModelType) => void;
}

const models = [
  { 
    id: "vae", 
    name: "VAE", 
    fullName: "Variational Autoencoder",
    description: "Generates CT scans by learning a compressed latent representation of the data.",
    badge: "Fastest"
  },
  { 
    id: "gan", 
    name: "GAN", 
    fullName: "Generative Adversarial Network",
    description: "Creates realistic CT scans through adversarial training between generator and discriminator networks.",
    badge: "Balanced"
  },
  { 
    id: "diffusion", 
    name: "Diffusion", 
    fullName: "Diffusion Model",
    description: "Produces high-quality CT scans by gradually denoising random noise patterns.",
    badge: "Highest Quality"
  }
];

const ModelSelector = ({ onModelSelect }: ModelSelectorProps) => {
  const [selectedModel, setSelectedModel] = useState<ModelType>("vae");
  const selectedModelDetails = models.find(m => m.id === selectedModel);

  const handleModelChange = (value: string) => {
    const modelType = value as ModelType;
    setSelectedModel(modelType);
    onModelSelect(modelType);
  };

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="model-select">Select Generation Model</Label>
        <Select value={selectedModel} onValueChange={handleModelChange}>
          <SelectTrigger id="model-select" className="w-full">
            <SelectValue placeholder="Select a model" />
          </SelectTrigger>
          <SelectContent>
            {models.map((model) => (
              <SelectItem key={model.id} value={model.id}>
                <span className="flex items-center gap-2">
                  {model.name}
                  <Badge variant="secondary" className="ml-2">{model.badge}</Badge>
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      
      {selectedModelDetails && (
        <div className="bg-secondary/30 p-4 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium text-lg">{selectedModelDetails.fullName}</h3>
            <Badge variant="outline">{selectedModelDetails.badge}</Badge>
          </div>
          <p className="text-sm text-muted-foreground">{selectedModelDetails.description}</p>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;
