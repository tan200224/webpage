
import { ChevronUp, Github, Linkedin, Twitter, Instagram, Code, Terminal, BrainCircuit } from "lucide-react";

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };
  
  return (
    <footer className="bg-secondary/50 border-t border-border py-12 relative overflow-hidden">
      <div className="container mx-auto px-6 relative z-10">
        <div className="flex flex-col items-center">
          <button
            onClick={scrollToTop}
            className="p-4 bg-white rounded-full shadow-md mb-8 hover:shadow-lg transition-all group"
            aria-label="Scroll to top"
          >
            <ChevronUp className="h-5 w-5 text-primary group-hover:animate-bounce" />
          </button>
          
          <div className="text-center">
            <a href="#home" className="font-bold text-2xl tracking-tight inline-flex items-center gap-2">
              <Code className="h-6 w-6 text-primary" />
              <span className="ai-gradient-text">Dev</span>Portfolio
            </a>
            
            <div className="flex items-center justify-center gap-3 mt-3 mb-6">
              <Terminal className="h-4 w-4 text-primary" />
              <span className="font-mono text-sm text-muted-foreground">AI + Software Engineering</span>
              <BrainCircuit className="h-4 w-4 text-primary" />
            </div>
            
            <p className="text-sm text-muted-foreground mb-8 max-w-lg mx-auto">
              Building intelligent software solutions with a focus on performance, 
              usability, and cutting-edge AI technologies.
            </p>
            
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 max-w-md mx-auto mb-10">
              <a 
                href="#" 
                className="flex flex-col items-center gap-2 text-muted-foreground hover:text-primary transition-colors"
              >
                <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center shadow-sm">
                  <Github className="h-5 w-5" />
                </div>
                <span className="text-xs">GitHub</span>
              </a>
              <a 
                href="#" 
                className="flex flex-col items-center gap-2 text-muted-foreground hover:text-primary transition-colors"
              >
                <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center shadow-sm">
                  <Linkedin className="h-5 w-5" />
                </div>
                <span className="text-xs">LinkedIn</span>
              </a>
              <a 
                href="#" 
                className="flex flex-col items-center gap-2 text-muted-foreground hover:text-primary transition-colors"
              >
                <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center shadow-sm">
                  <Twitter className="h-5 w-5" />
                </div>
                <span className="text-xs">Twitter</span>
              </a>
              <a 
                href="#" 
                className="flex flex-col items-center gap-2 text-muted-foreground hover:text-primary transition-colors"
              >
                <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center shadow-sm">
                  <Instagram className="h-5 w-5" />
                </div>
                <span className="text-xs">Instagram</span>
              </a>
            </div>
            
            <div className="text-xs text-muted-foreground pt-6 border-t border-border">
              <p>© {currentYear} All Rights Reserved</p>
              <p className="mt-1 font-mono">
                <span className="text-primary">const</span> passion = <span className="text-purple-500">'coding'</span>;
              </p>
            </div>
          </div>
        </div>
      </div>
      
      <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-primary via-purple-500 to-blue-500"></div>
      
      <div className="absolute -top-24 -right-24 h-64 w-64 bg-primary/5 rounded-full blur-3xl"></div>
      <div className="absolute -bottom-32 -left-32 h-64 w-64 bg-purple-500/5 rounded-full blur-3xl"></div>
    </footer>
  );
};

export default Footer;
