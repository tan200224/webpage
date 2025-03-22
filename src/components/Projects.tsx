
import { useState, useRef, useEffect } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, Github } from "lucide-react";

// Sample projects - replace with your actual projects
const PROJECTS_DATA = [
  {
    id: 1,
    title: "E-commerce Platform",
    description: "A modern e-commerce platform with real-time inventory, payment processing, and an interactive product discovery system.",
    image: "https://images.unsplash.com/photo-1661956602868-6ae368943878?q=80&w=2670&auto=format&fit=crop",
    technologies: ["React", "Node.js", "MongoDB", "Stripe", "Redis"],
    liveUrl: "#",
    githubUrl: "#",
  },
  {
    id: 2,
    title: "AI-Powered Analytics Dashboard",
    description: "A data visualization platform using machine learning to provide predictive insights and interactive reporting tools.",
    image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2670&auto=format&fit=crop",
    technologies: ["Python", "TensorFlow", "React", "D3.js", "FastAPI"],
    liveUrl: "#",
    githubUrl: "#",
  },
  {
    id: 3,
    title: "Real-Time Collaboration Tool",
    description: "A collaborative workspace for remote teams with document sharing, video conferencing, and project management features.",
    image: "https://images.unsplash.com/photo-1462899006636-339e08d1844e?q=80&w=2670&auto=format&fit=crop",
    technologies: ["Vue.js", "WebSockets", "Express", "PostgreSQL", "WebRTC"],
    liveUrl: "#",
    githubUrl: "#",
  },
  {
    id: 4,
    title: "Mobile Fitness App",
    description: "A cross-platform fitness application with custom workout plans, progress tracking, and social features.",
    image: "https://images.unsplash.com/photo-1576678927484-cc907957088c?q=80&w=2687&auto=format&fit=crop",
    technologies: ["React Native", "Firebase", "GraphQL", "TensorFlow Lite"],
    liveUrl: "#",
    githubUrl: "#",
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
    <section id="projects" ref={sectionRef} className="py-20">
      <div className="section-container">
        <div
          className={`transform transition-all duration-700 ${
            isVisible
              ? "translate-y-0 opacity-100"
              : "translate-y-10 opacity-0"
          }`}
        >
          <h2 className="section-title">Projects</h2>
          <p className="section-subtitle">
            A selection of my best work that demonstrates my skills and problem-solving abilities.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {PROJECTS_DATA.map((project, index) => (
              <Card
                key={project.id}
                className={`overflow-hidden border-0 shadow-lg relative rounded-xl transition-all duration-500 ${
                  isVisible
                    ? "animate-fade-up opacity-100"
                    : "opacity-0"
                }`}
                style={{ animationDelay: `${0.2 * index}s` }}
                onMouseEnter={() => setHoveredCard(project.id)}
                onMouseLeave={() => setHoveredCard(null)}
              >
                <div 
                  className="h-56 overflow-hidden"
                  style={{
                    backgroundImage: `url(${project.image})`,
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                    transition: 'transform 0.6s cubic-bezier(0.33, 1, 0.68, 1)',
                    transform: hoveredCard === project.id ? 'scale(1.05)' : 'scale(1)',
                  }}
                />
                
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
                      className="gap-2 button-hover"
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
    </section>
  );
};

export default Projects;
