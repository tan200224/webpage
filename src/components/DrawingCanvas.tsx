
import { useRef, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { RotateCcw, Download } from "lucide-react";
import { toast } from "sonner";

interface DrawingCanvasProps {
  width: number;
  height: number;
  onMaskCreated: (maskData: ImageData) => void;
}

const DrawingCanvas = ({ width, height, onMaskCreated }: DrawingCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [ctx, setCtx] = useState<CanvasRenderingContext2D | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    // Set initial canvas state
    context.fillStyle = "black";
    context.fillRect(0, 0, width, height);
    context.lineWidth = 10;
    context.lineCap = "round";
    context.strokeStyle = "white";
    
    setCtx(context);
  }, [width, height]);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!ctx) return;
    
    setIsDrawing(true);
    ctx.beginPath();
    
    const { offsetX, offsetY } = getCoordinates(e);
    ctx.moveTo(offsetX, offsetY);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !ctx) return;
    
    const { offsetX, offsetY } = getCoordinates(e);
    ctx.lineTo(offsetX, offsetY);
    ctx.stroke();
  };

  const stopDrawing = () => {
    if (!ctx) return;
    
    setIsDrawing(false);
    ctx.closePath();
    
    // Pass the mask data to parent component
    const imageData = ctx.getImageData(0, 0, width, height);
    onMaskCreated(imageData);
  };

  const getCoordinates = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return { offsetX: 0, offsetY: 0 };
    
    let offsetX = 0;
    let offsetY = 0;
    
    if ('touches' in e) {
      // Touch event
      const rect = canvasRef.current.getBoundingClientRect();
      offsetX = e.touches[0].clientX - rect.left;
      offsetY = e.touches[0].clientY - rect.top;
    } else {
      // Mouse event
      offsetX = e.nativeEvent.offsetX;
      offsetY = e.nativeEvent.offsetY;
    }
    
    return { offsetX, offsetY };
  };

  const clearCanvas = () => {
    if (!ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, width, height);
    onMaskCreated(ctx.getImageData(0, 0, width, height));
    toast.info("Canvas cleared");
  };

  const downloadMask = () => {
    if (!canvasRef.current) return;
    
    const link = document.createElement('a');
    link.download = 'ct-scan-mask.png';
    link.href = canvasRef.current.toDataURL();
    link.click();
    toast.success("Mask downloaded");
  };

  return (
    <div className="flex flex-col items-center">
      <div className="border-2 border-primary rounded-lg overflow-hidden shadow-lg">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
          className="touch-none"
        />
      </div>
      <div className="flex gap-2 mt-4">
        <Button 
          variant="outline" 
          onClick={clearCanvas}
          className="flex items-center gap-2"
        >
          <RotateCcw size={16} />
          Clear
        </Button>
        <Button 
          variant="outline"
          onClick={downloadMask}
          className="flex items-center gap-2"
        >
          <Download size={16} />
          Save Mask
        </Button>
      </div>
      <p className="text-sm text-muted-foreground mt-2">
        Draw a white mask on the black background. The mask will be used to generate a CT scan.
      </p>
    </div>
  );
};

export default DrawingCanvas;
