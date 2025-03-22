
import { ArrowDown, Github, Linkedin, Mail, Code, Terminal, BrainCircuit } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useRef, useState } from "react";

const MatrixRain = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    const fontSize = 14;
    const columns = Math.floor(canvas.width / fontSize);
    
    const drops: number[] = [];
    for (let i = 0; i < columns; i++) {
      drops[i] = Math.random() * -100;
    }
    
    const matrix = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ123456789@#$%^&*()*&^%+-/~{[|`]}";
    
    const draw = () => {
      ctx.fillStyle = 'rgba(240, 247, 255, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      ctx.fillStyle = 'rgba(99, 102, 241, 0.8)';
      ctx.font = `${fontSize}px monospace`;
      
      for (let i = 0; i < drops.length; i++) {
        const text = matrix[Math.floor(Math.random() * matrix.length)];
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);
        
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.98) {
          drops[i] = 0;
        }
        
        drops[i]++;
      }
    };
    
    const interval = setInterval(draw, 35);
    
    return () => clearInterval(interval);
  }, []);
  
  return <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full opacity-10 pointer-events-none" />;
};

const Hero = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [typedText, setTypedText] = useState("");
  const fullText = "Building intelligent software solutions";
  const heroRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setIsVisible(true);
    
    let i = 0;
    const typeInterval = setInterval(() => {
      if (i < fullText.length) {
        setTypedText(fullText.substring(0, i + 1));
        i++;
      } else {
        clearInterval(typeInterval);
      }
    }, 100);
    
    return () => clearInterval(typeInterval);
  }, []);

  const scrollToNext = () => {
    const nextSection = document.querySelector('#experience');
    if (nextSection) {
      nextSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section 
      id="home" 
      ref={heroRef}
      className="relative min-h-screen flex flex-col justify-center items-center overflow-hidden"
    >
      <MatrixRain />
      
      <div className="section-container flex flex-col items-center z-10">
        <div className="max-w-4xl mx-auto text-center">
          <div className="overflow-hidden">
            <p 
              className={`font-mono text-primary text-sm sm:text-base mb-2 ${
                isVisible ? 'animate-text-reveal' : 'opacity-0'
              }`}
              style={{ animationDelay: '0.2s' }}
            >
              <Terminal className="inline-block w-4 h-4 mr-2" />
              Hello, I'm a
            </p>
          </div>
          
          <div className="overflow-hidden mb-4">
            <h1 
              className={`text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight ${
                isVisible ? 'animate-text-reveal' : 'opacity-0'
              }`}
              style={{ animationDelay: '0.4s' }}
            >
              <span className="ai-gradient-text">Software Engineer</span>
            </h1>
          </div>
          
          <div className="overflow-hidden">
            <p 
              className={`font-mono text-lg text-foreground ${
                isVisible ? '' : 'opacity-0'
              }`}
            >
              {typedText}<span className="animate-blink">|</span>
            </p>
          </div>
          
          <div className="overflow-hidden">
            <p 
              className={`text-muted-foreground mt-6 mb-8 max-w-2xl mx-auto px-4 sm:px-0 ${
                isVisible ? 'animate-text-reveal' : 'opacity-0'
              }`}
              style={{ animationDelay: '0.6s' }}
            >
              I build exceptional digital experiences with a focus on <span className="text-primary font-medium">AI-powered solutions</span>, 
              <span className="text-primary font-medium"> intelligent algorithms</span>, and 
              <span className="text-primary font-medium"> cutting-edge technologies</span>. 
              Transforming complex challenges into elegant software is my passion.
            </p>
          </div>
          
          <div className="flex justify-center space-x-4 mb-8">
            <div className={`ai-card p-3 floating ${isVisible ? 'animate-fade-up' : 'opacity-0'}`} style={{ animationDelay: '0.7s' }}>
              <Code className="h-5 w-5 text-primary" />
            </div>
            <div className={`ai-card p-3 floating ${isVisible ? 'animate-fade-up' : 'opacity-0'}`} style={{ animationDelay: '0.8s', animationDuration: '3.5s' }}>
              <BrainCircuit className="h-5 w-5 text-purple-500" />
            </div>
            <div className={`ai-card p-3 floating ${isVisible ? 'animate-fade-up' : 'opacity-0'}`} style={{ animationDelay: '0.9s', animationDuration: '4s' }}>
              <Terminal className="h-5 w-5 text-blue-500" />
            </div>
          </div>
          
          <div 
            className={`flex flex-col sm:flex-row items-center justify-center gap-4 ${
              isVisible ? 'animate-fade-up' : 'opacity-0'
            }`}
            style={{ animationDelay: '0.8s' }}
          >
            <Button 
              className="w-full sm:w-auto button-hover gradient-bg text-white border-none"
              onClick={() => {
                const projectsSection = document.querySelector('#projects');
                if (projectsSection) {
                  projectsSection.scrollIntoView({ behavior: 'smooth' });
                }
              }}
            >
              View My Work
            </Button>
            <Button 
              variant="outline" 
              className="w-full sm:w-auto button-hover"
              onClick={() => {
                const contactSection = document.querySelector('#contact');
                if (contactSection) {
                  contactSection.scrollIntoView({ behavior: 'smooth' });
                }
              }}
            >
              Contact Me
            </Button>
          </div>
          
          <div 
            className={`flex justify-center mt-8 space-x-4 ${
              isVisible ? 'animate-fade-up' : 'opacity-0'
            }`}
            style={{ animationDelay: '1s' }}
          >
            <a href="#" className="text-muted-foreground hover:text-primary transition-colors" aria-label="GitHub">
              <Github size={20} />
            </a>
            <a href="#" className="text-muted-foreground hover:text-primary transition-colors" aria-label="LinkedIn">
              <Linkedin size={20} />
            </a>
            <a href="#" className="text-muted-foreground hover:text-primary transition-colors" aria-label="Email">
              <Mail size={20} />
            </a>
          </div>
        </div>
        
        <button 
          onClick={scrollToNext}
          className={`absolute bottom-10 animate-bounce border border-border rounded-full p-2 ${
            isVisible ? 'opacity-80' : 'opacity-0'
          } transition-opacity hover:opacity-100 z-10`}
          aria-label="Scroll down"
        >
          <ArrowDown size={18} className="text-primary" />
        </button>
      </div>
      
      <div className="absolute top-0 left-0 right-0 bottom-0 -z-10">
        <div className="absolute top-0 left-0 h-96 w-96 bg-primary/5 rounded-full blur-3xl -translate-x-1/2 -translate-y-1/2"></div>
        <div className="absolute bottom-0 right-0 h-96 w-96 bg-primary/5 rounded-full blur-3xl translate-x-1/3 translate-y-1/3"></div>
      </div>
    </section>
  );
};

export default Hero;
