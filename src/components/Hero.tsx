import { ArrowDown, Github, Linkedin, Mail, Code, Terminal, BrainCircuit, Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useRef, useState } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

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

const GlowingCircle = ({
  size,
  delay,
  color
}: {
  size: string;
  delay: string;
  color: string;
}) => <div className="absolute rounded-full animate-pulse-glow" style={{
  width: size,
  height: size,
  backgroundColor: color,
  filter: `blur(${parseInt(size) / 4}px)`,
  opacity: 0.2,
  animationDelay: delay
}} />;

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
      nextSection.scrollIntoView({
        behavior: 'smooth'
      });
    }
  };

  return <section id="home" ref={heroRef} className="relative min-h-screen flex flex-col justify-center items-center overflow-hidden">
      <MatrixRain />
      
      <div className="section-container flex flex-col items-center z-10">
        <div className="max-w-5xl mx-auto w-full relative z-10">
          {/* Background glow effects */}
          <div className="absolute -z-10 inset-0 flex justify-center items-center opacity-70">
            <GlowingCircle size="350px" delay="0s" color="rgba(99, 102, 241, 0.5)" />
            <GlowingCircle size="250px" delay="2s" color="rgba(139, 92, 246, 0.5)" />
            <GlowingCircle size="300px" delay="4s" color="rgba(59, 130, 246, 0.5)" />
          </div>
          
          {/* Main content */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-12 items-center px-4">
            {/* Left side: Profile image with fancy border */}
            <div className={`flex justify-center md:justify-end ${isVisible ? 'animate-fade-up' : 'opacity-0'}`}
                style={{ animationDelay: '0.2s' }}>
              <div className="relative">
                {/* Circular gradient border */}
                <div className="absolute inset-0 rounded-full bg-gradient-to-br from-primary via-purple-500 to-blue-500 
                     blur-sm p-1 -m-1 animate-rotate-slow"></div>
                
                {/* Profile image */}
                <div className="relative rounded-full p-1 bg-background overflow-hidden">
                  <Avatar className="w-52 h-52 md:w-72 md:h-72 border-2 border-background">
                    <AvatarImage 
                      src="/lovable-uploads/4da5e4b9-fb36-4cda-b475-591f1702c4db.png" 
                      alt="Archie Tan"
                      className="object-cover" />
                    <AvatarFallback>AT</AvatarFallback>
                  </Avatar>
                  
                  {/* Tech icons floating around the profile */}
                  <div className="absolute -top-3 -right-3 ai-card p-3 floating animate-float-slow">
                    <Code className="h-6 w-6 text-primary" />
                  </div>
                  <div className="absolute -bottom-3 -right-3 ai-card p-3 floating" 
                       style={{ animationDuration: '4s', animationDelay: '1s' }}>
                    <BrainCircuit className="h-6 w-6 text-purple-500" />
                  </div>
                  <div className="absolute -bottom-3 -left-3 ai-card p-3 floating" 
                       style={{ animationDuration: '3.5s', animationDelay: '0.5s' }}>
                    <Cpu className="h-6 w-6 text-blue-500" />
                  </div>
                </div>
              </div>
            </div>
            
            {/* Right side: Text content */}
            <div className="text-center md:text-left">
              <div className="overflow-hidden mb-1">
                <p className={`font-mono text-primary text-sm sm:text-base ${isVisible ? 'animate-text-reveal' : 'opacity-0'}`} 
                   style={{ animationDelay: '0.3s' }}>
                  <Terminal className="inline-block w-4 h-4 mr-2" />
                  Hello, I'm
                </p>
              </div>
              
              <div className="overflow-hidden mb-2">
                <h1 className={`text-4xl sm:text-6xl md:text-7xl font-bold tracking-tight ${isVisible ? 'animate-text-reveal' : 'opacity-0'}`}
                    style={{ animationDelay: '0.5s' }}>
                  <span className="ai-gradient-text">Archie Tan</span>
                </h1>
              </div>
              
              <div className="overflow-hidden mb-4">
                <h2 className={`text-2xl sm:text-3xl md:text-4xl font-medium ${isVisible ? 'animate-text-reveal' : 'opacity-0'}`}
                    style={{ animationDelay: '0.6s' }}>
                  AI & Software Engineer
                </h2>
              </div>
              
              <div className="overflow-hidden mb-3">
                <p className={`font-mono text-lg text-foreground ${isVisible ? '' : 'opacity-0'}`}>
                  {typedText}<span className="animate-blink">|</span>
                </p>
              </div>
              
              <div className="overflow-hidden mb-6">
                <p className={`text-muted-foreground mt-4 max-w-lg ${isVisible ? 'animate-text-reveal' : 'opacity-0'}`}
                   style={{ animationDelay: '0.8s' }}>
                  I'm a <span className="text-primary font-medium">researcher and developer</span> with a focus on 
                  <span className="text-primary font-medium"> machine learning</span> and 
                  <span className="text-primary font-medium"> software engineering</span>. 
                  Specializing in pancreatic cancer diagnosis through AI models and synthetic data generation.
                </p>
              </div>
              
              <div className={`flex flex-col sm:flex-row items-center md:items-start justify-center md:justify-start gap-4 ${isVisible ? 'animate-fade-up' : 'opacity-0'}`}
                   style={{ animationDelay: '1s' }}>
                <Button className="w-full sm:w-auto button-hover gradient-bg text-white border-none" 
                        onClick={() => {
                          const projectsSection = document.querySelector('#projects');
                          if (projectsSection) {
                            projectsSection.scrollIntoView({
                              behavior: 'smooth'
                            });
                          }
                        }}>
                  View My Work
                </Button>
                <Button variant="outline" className="w-full sm:w-auto button-hover" 
                        onClick={() => {
                          const contactSection = document.querySelector('#contact');
                          if (contactSection) {
                            contactSection.scrollIntoView({
                              behavior: 'smooth'
                            });
                          }
                        }}>
                  Contact Me
                </Button>
              </div>
              
              <div className={`flex justify-center md:justify-start mt-6 space-x-4 ${isVisible ? 'animate-fade-up' : 'opacity-0'}`}
                   style={{ animationDelay: '1.2s' }}>
                <a href="https://github.com/tan200224" 
                   className="text-muted-foreground hover:text-primary transition-colors" 
                   aria-label="GitHub">
                  <Github size={20} />
                </a>
                <a href="https://linkedin.com/in/zhuohaotan/" 
                   className="text-muted-foreground hover:text-primary transition-colors" 
                   aria-label="LinkedIn">
                  <Linkedin size={20} />
                </a>
                <a href="mailto:tan200224@gmail.com" 
                   className="text-muted-foreground hover:text-primary transition-colors" 
                   aria-label="Email">
                  <Mail size={20} />
                </a>
              </div>
            </div>
          </div>
        </div>
        
        <button onClick={scrollToNext} 
                className={`absolute bottom-10 animate-bounce border border-border rounded-full p-2 ${isVisible ? 'opacity-80' : 'opacity-0'} transition-opacity hover:opacity-100 z-10`} 
                aria-label="Scroll down">
          <ArrowDown size={18} className="text-primary" />
        </button>
      </div>
      
      <div className="absolute top-0 left-0 right-0 bottom-0 -z-10">
        <div className="absolute top-0 left-0 h-96 w-96 bg-primary/5 rounded-full blur-3xl -translate-x-1/2 -translate-y-1/2"></div>
        <div className="absolute bottom-0 right-0 h-96 w-96 bg-primary/5 rounded-full blur-3xl translate-x-1/3 translate-y-1/3"></div>
      </div>
    </section>;
};

export default Hero;
