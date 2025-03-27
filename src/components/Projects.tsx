
import { useState, useRef, useEffect } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, Github, Code, BrainCircuit, Cpu, Bot, Film } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";

const PROJECTS_DATA = [
  {
    id: 1,
    title: "Pancreatic Cancer AI Diagnosis",
    description: "Cutting-edge research implementing segmentation models for early diagnosis of pancreatic cancer through CT scan analysis with 87.16% accuracy.",
    image: "https://images.unsplash.com/photo-1607798748738-b15c40d33d57?q=80&w=2670&auto=format&fit=crop",
    technologies: ["PyTorch", "TorchVision", "Python", "Data Augmentation", "3D Imaging"],
    liveUrl: "/pancreatic-cancer-demo",
    githubUrl: "https://github.com/tan200224/Research_Blog",
    icon: <BrainCircuit className="h-10 w-10 text-primary" />,
    hasDemo: true
  },
  {
    id: 2,
    title: "Synthetic CT Scan Generator",
    description: "Innovative Annotation-to-3D-CT-Scan generative model built from scratch with PyTorch to produce realistic synthetic medical image data.",
    image: "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?q=80&w=2670&auto=format&fit=crop",
    technologies: ["PyTorch", "Generative AI", "3D Modeling", "Medical Imaging", "Data Synthesis"],
    liveUrl: "/synthetic-ct-demo",
    githubUrl: "https://github.com/tan200224/Research_Blog",
    icon: <Cpu className="h-10 w-10 text-purple-500" />,
    hasDemo: true
  },
  {
    id: 3,
    title: "Course Specific Chatbot",
    description: "AI-powered chatbot assistant for academic courses utilizing OpenAI API for NLP, trained on course materials while preventing academic dishonesty.",
    image: "https://images.unsplash.com/photo-1655721530791-65e945810047?q=80&w=2670&auto=format&fit=crop",
    technologies: ["OpenAI API", "NLP", "Python", "Fine-tuning", "Academic Ethics"],
    liveUrl: "https://drive.google.com/file/d/1v0mN920ChuhSzbn-3CTaxQaV9ebfnEud/view?usp=drive_link",
    githubUrl: "#",
    icon: <Bot className="h-10 w-10 text-blue-500" />
  },
  {
    id: 4,
    title: "Movie Trailer Viewer",
    description: "Mobile app that accesses MovieDatabase API, enabling users to browse movies, view details, and watch trailers with optimized performance.",
    image: "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=2670&auto=format&fit=crop",
    technologies: ["Android", "Async HTTP", "Glide", "API Integration", "Java"],
    liveUrl: "https://drive.google.com/file/d/1kxaEK7cI3u-g3_Q93S-ETFQqUNQ05H2Z/view?usp=drive_link",
    githubUrl: "https://github.com/tan200224/Movie-Trailer-Viewer/",
    icon: <Film className="h-10 w-10 text-green-500" />
  }
];

const Projects = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);
  const sectionRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => {
      observer.disconnect();
    };
  }, []);
  
  const handleDemoClick = (url: string) => {
    navigate(url);
    window.scrollTo(0, 0);
  };

  return (
    <section id="projects" ref={sectionRef} className="py-20 relative overflow-hidden">
      <div className="absolute w-full h-full top-0 left-0 -z-5">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-primary/5 rounded-full blur-3xl"></div>
        <div className="absolute top-3/4 right-1/4 w-64 h-64 bg-purple-500/5 rounded-full blur-3xl"></div>
      </div>
      
      <div className="section-container relative z-10">
        <div
          className={`transform transition-all duration-700 ${
            isVisible
              ? "translate-y-0 opacity-100"
              : "translate-y-10 opacity-0"
          }`}
        >
          <h2 className="section-title">
            <span className="ai-gradient-text">Research & Projects</span>
          </h2>
          <p className="section-subtitle">
            Showcasing my work in machine learning research and software development
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {PROJECTS_DATA.map((project, index) => (
              <Card
                key={project.id}
                className={`overflow-hidden border-0 shadow-lg relative rounded-xl transition-all duration-500 group ${
                  isVisible
                    ? "animate-fade-up opacity-100"
                    : "opacity-0"
                }`}
                style={{ animationDelay: `${0.2 * index}s` }}
                onMouseEnter={() => setHoveredCard(project.id)}
                onMouseLeave={() => setHoveredCard(null)}
              >
                <div className="absolute top-4 right-4 z-20 bg-white rounded-full p-3 shadow-lg transform transition-transform duration-500 group-hover:rotate-12">
                  {project.icon}
                </div>
                
                <div className="h-56 overflow-hidden relative">
                  <div 
                    className="absolute inset-0 transition-transform duration-700 ease-in-out bg-cover bg-center"
                    style={{
                      backgroundImage: `url(${project.image})`,
                      transform: hoveredCard === project.id ? 'scale(1.05)' : 'scale(1)',
                    }}
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent" />
                  
                  <div className={`absolute inset-0 transition-opacity duration-300 ${hoveredCard === project.id ? 'opacity-30' : 'opacity-0'}`}>
                    <svg className="w-full h-full">
                      <defs>
                        <pattern id={`pattern-${project.id}`} x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
                          <circle cx="1" cy="1" r="1" fill="rgba(255,255,255,0.3)" />
                        </pattern>
                      </defs>
                      <rect x="0" y="0" width="100%" height="100%" fill={`url(#pattern-${project.id})`} />
                    </svg>
                  </div>
                </div>
                
                <CardContent className="relative z-10 p-6 bg-background">
                  <h3 className="font-bold text-xl mb-2">{project.title}</h3>
                  <p className="text-muted-foreground mb-4">{project.description}</p>
                  
                  <div className="flex flex-wrap gap-2 mb-6">
                    {project.technologies.map((tech, i) => (
                      <Badge key={i} variant="secondary" className="rounded-full">
                        {tech}
                      </Badge>
                    ))}
                  </div>
                  
                  <div className="flex flex-wrap gap-4 mt-auto">
                    <Button
                      size="sm"
                      variant="outline"
                      className="gap-2 button-hover"
                      asChild
                    >
                      <a href={project.githubUrl} target="_blank" rel="noopener noreferrer">
                        <Github className="h-4 w-4" />
                        <span>Code</span>
                      </a>
                    </Button>
                    
                    {project.hasDemo ? (
                      <Button
                        size="sm"
                        className="gap-2 button-hover gradient-bg text-white border-none"
                        onClick={() => handleDemoClick(project.liveUrl)}
                      >
                        <ExternalLink className="h-4 w-4" />
                        <span>Live Demo</span>
                      </Button>
                    ) : (
                      <Button
                        size="sm"
                        className="gap-2 button-hover gradient-bg text-white border-none"
                        asChild
                      >
                        <a href={project.liveUrl} target="_blank" rel="noopener noreferrer">
                          <ExternalLink className="h-4 w-4" />
                          <span>Live Demo</span>
                        </a>
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          
          <div className="text-center mt-12">
            <Button 
              variant="outline" 
              size="lg"
              className="button-hover"
              asChild
            >
              <a href="https://github.com/tan200224" target="_blank" rel="noopener noreferrer">
                <Github className="h-4 w-4 mr-2" />
                <span>View All Projects on GitHub</span>
              </a>
            </Button>
          </div>
        </div>
      </div>
      
      <div className="absolute inset-0 -z-10">
        <div className="absolute -bottom-24 -left-24 h-64 w-64 bg-primary/10 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 right-0 h-72 w-72 bg-purple-500/5 rounded-full blur-3xl"></div>
      </div>
    </section>
  );
};

export default Projects;
