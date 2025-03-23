
import { useState, useEffect, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Menu, X } from "lucide-react";
import { Link, useLocation } from 'react-router-dom';

const MENU_ITEMS = [{
  name: 'Home',
  href: '#home'
}, {
  name: 'Experience',
  href: '#experience'
}, {
  name: 'Projects',
  href: '#projects'
}, {
  name: 'Skills',
  href: '#skills'
}, {
  name: 'Contact',
  href: '#contact'
}];

const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navRef = useRef<HTMLDivElement>(null);
  const location = useLocation();
  const isMainPage = location.pathname === '/';

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 10) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };
    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const handleLinkClick = (href: string) => {
    setMobileMenuOpen(false);
    
    // If we're on the main page, scroll to the section
    if (isMainPage) {
      const element = document.querySelector(href);
      if (element) {
        element.scrollIntoView({
          behavior: 'smooth'
        });
      }
    }
  };

  return (
    <nav 
      ref={navRef} 
      className={`fixed top-0 w-full z-50 px-6 md:px-12 py-4 transition-all duration-300 ${
        isScrolled ? 'bg-white/80 backdrop-blur-md shadow-sm' : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {isMainPage ? (
          <a 
            href="#home" 
            className="font-bold text-xl tracking-tight" 
            onClick={e => {
              e.preventDefault();
              handleLinkClick('#home');
            }}
          >
            <span className="text-primary">Archie</span>Portfolio
          </a>
        ) : (
          <Link to="/" className="font-bold text-xl tracking-tight">
            <span className="text-primary">Archie</span>Portfolio
          </Link>
        )}
        
        {/* Desktop Menu */}
        <div className="hidden md:flex items-center space-x-8">
          {MENU_ITEMS.map(item => (
            isMainPage ? (
              <a 
                key={item.name} 
                href={item.href} 
                className="text-sm font-medium link-underline" 
                onClick={e => {
                  e.preventDefault();
                  handleLinkClick(item.href);
                }}
              >
                {item.name}
              </a>
            ) : (
              <Link 
                key={item.name} 
                to={`/${item.href}`} 
                className="text-sm font-medium link-underline"
              >
                {item.name}
              </Link>
            )
          ))}
          {isMainPage ? (
            <Button 
              className="ml-4 px-5 button-hover" 
              onClick={() => handleLinkClick('#contact')}
            >
              Hire Me
            </Button>
          ) : (
            <Link to="/#contact">
              <Button className="ml-4 px-5 button-hover">
                Hire Me
              </Button>
            </Link>
          )}
        </div>
        
        {/* Mobile Menu Button */}
        <Button 
          variant="ghost" 
          size="icon" 
          className="md:hidden" 
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        >
          {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </Button>
      </div>
      
      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden absolute top-full left-0 right-0 bg-white shadow-lg p-6 flex flex-col space-y-4 animate-fade-in">
          {MENU_ITEMS.map(item => (
            isMainPage ? (
              <a 
                key={item.name} 
                href={item.href} 
                className="text-base font-medium py-2" 
                onClick={e => {
                  e.preventDefault();
                  handleLinkClick(item.href);
                }}
              >
                {item.name}
              </a>
            ) : (
              <Link 
                key={item.name} 
                to={`/${item.href}`} 
                className="text-base font-medium py-2"
              >
                {item.name}
              </Link>
            )
          ))}
          {isMainPage ? (
            <Button 
              className="w-full mt-4 button-hover" 
              onClick={() => handleLinkClick('#contact')}
            >
              Hire Me
            </Button>
          ) : (
            <Link to="/#contact">
              <Button className="w-full mt-4 button-hover">
                Hire Me
              </Button>
            </Link>
          )}
        </div>
      )}
    </nav>
  );
};

export default Navbar;
