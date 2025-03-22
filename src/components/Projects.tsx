
import { useState, useRef, useEffect } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, Github, Code, BrainCircuit, Cpu } from "lucide-react";

// Sample projects - replace with your actual projects
const PROJECTS_DATA = [
  {
    id: 1,
    title: "AI-Powered Code Generator",
    description: "A cutting-edge platform that leverages transformer models to generate, review, and optimize code across multiple programming languages.",
    image: "https://images.unsplash.com/photo-1607798748738-b15c40d33d57?q=80&w=2670&auto=format&fit=crop",
    technologies: ["React", "TypeScript", "Python", "TensorFlow", "transformers.js"],
    liveUrl: "#",
    githubUrl: "#",
    icon: <BrainCircuit className="h-10 w-10 text-primary" />
  },
  {
    id: 2,
    title: "Neural Network Visualizer",
    description: "An interactive tool for visualizing neural networks, allowing users to build, train and observe AI models in real-time with animated learning processes.",
    image: "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?q=80&w=2670&auto=format&fit=crop",
    technologies: ["Vue.js", "D3.js", "TensorFlow.js", "WebGL", "Express"],
    liveUrl: "#",
    githubUrl: "#",
    icon: <Cpu className="h-10 w-10 text-purple-500" />
  },
  {
    id: 3,
    title: "Real-Time ML Collaboration",
    description: "A collaborative workspace for data scientists with real-time model training, interactive notebooks, and integrated deployment pipelines.",
    image: "https://images.unsplash.com/photo-1655721530791-65e945810047?q=80&w=2670&auto=format&fit=crop",
    technologies: ["React", "Python", "WebSockets", "PyTorch", "Docker"],
    liveUrl: "#",
    githubUrl: "#",
    icon: <Code className="h-10 w-10 text-blue-500" />
  },
  {
    id: 4,
    title: "Computer Vision API",
    description: "A comprehensive API for image recognition, object detection, and scene understanding with a dashboard for monitoring and fine-tuning models.",
    image: "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=2670&auto=format&fit=crop",
    technologies: ["FastAPI", "OpenCV", "React", "TensorFlow", "AWS"],
    liveUrl: "#",
    githubUrl: "#",
    icon: <BrainCircuit className="h-10 w-10 text-green-500" />
  }
];

const Projects = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);
  const sectionRef = useRef<HTMLDivElement>(null);

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

  return (
    <section id="projects" ref={sectionRef} className="py-20 relative overflow-hidden">
      <div className="section-container relative z-10">
        <div
          className={`transform transition-all duration-700 ${
            isVisible
              ? "translate-y-0 opacity-100"
              : "translate-y-10 opacity-0"
          }`}
        >
          <h2 className="section-title">
            <span className="ai-gradient-text">Projects</span>
          </h2>
          <p className="section-subtitle">
            A selection of my best work that demonstrates my expertise in AI and software engineering.
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
                <div 
                  className="h-56 overflow-hidden relative"
                >
                  <div 
                    className="absolute inset-0 transition-transform duration-700 ease-in-out bg-cover bg-center"
                    style={{
                      backgroundImage: `url(${project.image})`,
                      transform: hoveredCard === project.id ? 'scale(1.05)' : 'scale(1)',
                    }}
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
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
              <a href="#" target="_blank" rel="noopener noreferrer">
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
