
import { useState, useEffect, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Menu, X } from "lucide-react";
import { Link, useLocation, useNavigate } from 'react-router-dom';

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
  const navigate = useNavigate();
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
    
    if (isMainPage) {
      // If we're on the main page, scroll to the section
      const element = document.querySelector(href);
      if (element) {
        element.scrollIntoView({
          behavior: 'smooth'
        });
      }
    } else {
      // If we're not on the main page, navigate to the main page with the section hash
      navigate('/' + href);
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
        <a 
          onClick={(e) => {
            e.preventDefault();
            navigate('/');
            setTimeout(() => {
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }, 100);
          }}
          className="font-bold text-xl tracking-tight cursor-pointer" 
        >
          <span className="text-primary">Archie</span>Portfolio
        </a>
        
        {/* Desktop Menu */}
        <div className="hidden md:flex items-center space-x-8">
          {MENU_ITEMS.map(item => (
            <a 
              key={item.name} 
              onClick={(e) => {
                e.preventDefault();
                if (isMainPage) {
                  handleLinkClick(item.href);
                } else {
                  navigate('/' + item.href);
                }
              }}
              className="text-sm font-medium link-underline cursor-pointer" 
            >
              {item.name}
            </a>
          ))}
          <Button 
            className="ml-4 px-5 button-hover" 
            onClick={() => {
              if (isMainPage) {
                handleLinkClick('#contact');
              } else {
                navigate('/#contact');
              }
            }}
          >
            Hire Me
          </Button>
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
            <a 
              key={item.name} 
              onClick={(e) => {
                e.preventDefault();
                if (isMainPage) {
                  handleLinkClick(item.href);
                } else {
                  navigate('/' + item.href);
                }
              }}
              className="text-base font-medium py-2 cursor-pointer" 
            >
              {item.name}
            </a>
          ))}
          <Button 
            className="w-full mt-4 button-hover" 
            onClick={() => {
              if (isMainPage) {
                handleLinkClick('#contact');
              } else {
                navigate('/#contact');
              }
            }}
          >
            Hire Me
          </Button>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
