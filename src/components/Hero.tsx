
import { ArrowDown, Github, Linkedin, Mail } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useRef, useState } from "react";

const Hero = () => {
  const [isVisible, setIsVisible] = useState(false);
  const heroRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setIsVisible(true);
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
      className="relative min-h-screen flex flex-col justify-center items-center"
    >
      <div className="section-container flex flex-col items-center">
        <div className="max-w-4xl mx-auto text-center">
          <div className="overflow-hidden">
            <p 
              className={`font-mono text-primary text-sm sm:text-base mb-2 ${
                isVisible ? 'animate-text-reveal' : 'opacity-0'
              }`}
              style={{ animationDelay: '0.2s' }}
            >
              Hello, I'm a
            </p>
          </div>
          
          <div className="overflow-hidden">
            <h1 
              className={`text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight mb-4 ${
                isVisible ? 'animate-text-reveal' : 'opacity-0'
              }`}
              style={{ animationDelay: '0.4s' }}
            >
              Software Engineer
            </h1>
          </div>
          
          <div className="overflow-hidden">
            <p 
              className={`text-muted-foreground mb-8 max-w-2xl mx-auto px-4 sm:px-0 ${
                isVisible ? 'animate-text-reveal' : 'opacity-0'
              }`}
              style={{ animationDelay: '0.6s' }}
            >
              I build exceptional digital experiences that are fast, accessible, 
              and designed with best practices. With a focus on clean code and innovative 
              solutions, I help transform ideas into reality.
            </p>
          </div>
          
          <div 
            className={`flex flex-col sm:flex-row items-center justify-center gap-4 ${
              isVisible ? 'animate-fade-up' : 'opacity-0'
            }`}
            style={{ animationDelay: '0.8s' }}
          >
            <Button 
              className="w-full sm:w-auto button-hover"
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
            <a href="#" className="text-muted-foreground hover:text-foreground transition-colors" aria-label="GitHub">
              <Github size={20} />
            </a>
            <a href="#" className="text-muted-foreground hover:text-foreground transition-colors" aria-label="LinkedIn">
              <Linkedin size={20} />
            </a>
            <a href="#" className="text-muted-foreground hover:text-foreground transition-colors" aria-label="Email">
              <Mail size={20} />
            </a>
          </div>
        </div>
        
        <button 
          onClick={scrollToNext}
          className={`absolute bottom-10 animate-bounce border border-border rounded-full p-2 ${
            isVisible ? 'opacity-80' : 'opacity-0'
          } transition-opacity hover:opacity-100`}
          aria-label="Scroll down"
        >
          <ArrowDown size={18} />
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
