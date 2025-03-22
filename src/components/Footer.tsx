
import { ChevronUp } from "lucide-react";

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };
  
  return (
    <footer className="bg-secondary/50 border-t border-border py-8">
      <div className="container mx-auto px-6">
        <div className="flex flex-col items-center">
          <button
            onClick={scrollToTop}
            className="p-2 bg-white rounded-full shadow-md mb-6 hover:shadow-lg transition-all"
            aria-label="Scroll to top"
          >
            <ChevronUp className="h-4 w-4" />
          </button>
          
          <div className="text-center">
            <a href="#home" className="font-bold text-xl tracking-tight">
              <span className="text-primary">Dev</span>Portfolio
            </a>
            
            <p className="text-sm text-muted-foreground mt-2 mb-6">
              Building exceptional software solutions with passion and precision.
            </p>
            
            <div className="flex justify-center space-x-6 mb-8">
              <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                GitHub
              </a>
              <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                LinkedIn
              </a>
              <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                Twitter
              </a>
              <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                Instagram
              </a>
            </div>
            
            <div className="text-xs text-muted-foreground">
              <p>© {currentYear} All Rights Reserved</p>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
