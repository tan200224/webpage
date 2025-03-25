
import { useState, useRef, useEffect } from 'react';
import { 
  Database, 
  Server, 
  Code, 
  Workflow, 
  Laptop, 
  Layers,
  BrainCircuit,
  LineChart,
  BookOpen
} from "lucide-react";

// Archie's actual skills based on his experience
const SKILLS_DATA = [
  {
    category: "AI & Machine Learning",
    icon: <BrainCircuit className="h-6 w-6" />,
    skills: [
      { name: "PyTorch", level: 90 },
      { name: "TorchVision", level: 85 },
      { name: "ML Pipelines", level: 90 },
      { name: "Data Augmentation", level: 85 },
      { name: "3D Image Processing", level: 80 },
    ]
  },
  {
    category: "Programming",
    icon: <Code className="h-6 w-6" />,
    skills: [
      { name: "Python", level: 95 },
      { name: "Java/C++", level: 90 },
      { name: "SQL", level: 85 },
      { name: "HTML/CSS", level: 85 },
      { name: "JavaScript", level: 80 },
    ]
  },
  {
    category: "Data Science",
    icon: <LineChart className="h-6 w-6" />,
    skills: [
      { name: "Scikit-Learn", level: 90 },
      { name: "Data Processing", level: 95 },
      { name: "Data Visualization", level: 85 },
      { name: "Statistical Analysis", level: 90 },
      { name: "QGIS", level: 85 },
    ]
  },
  {
    category: "Software Development",
    icon: <Laptop className="h-6 w-6" />,
    skills: [
      { name: "Object-Oriented Programming", level: 95 },
      { name: "Unit Testing", level: 85 },
      { name: "Development Lifecycle", level: 90 },
      { name: "API Design", level: 85 },
      { name: "Full-Stack Development", level: 80 },
    ]
  },
  {
    category: "DevOps & Tools",
    icon: <Workflow className="h-6 w-6" />,
    skills: [
      { name: "AWS", level: 80 },
      { name: "Linux", level: 85 },
      { name: "Git", level: 90 },
      { name: "Docker", level: 80 },
      { name: "Bash", level: 85 },
    ]
  },
  {
    category: "Academic & Research",
    icon: <BookOpen className="h-6 w-6" />,
    skills: [
      { name: "Research Methodology", level: 95 },
      { name: "Academic Writing", level: 90 },
      { name: "Grant Proposal", level: 85 },
      { name: "Math & Statistics", level: 90 },
      { name: "Problem Solving", level: 95 },
    ]
  }
];

const SkillBar = ({ name, level }: { name: string; level: number }) => {
  const [width, setWidth] = useState(0);
  const barRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);

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

    if (barRef.current) {
      observer.observe(barRef.current);
    }

    return () => {
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    if (isVisible) {
      const timer = setTimeout(() => {
        setWidth(level);
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [isVisible, level]);

  return (
    <div ref={barRef} className="mb-3">
      <div className="flex justify-between mb-1">
        <span className="text-sm">{name}</span>
        <span className="text-xs text-muted-foreground">{level}%</span>
      </div>
      <div className="h-2.5 w-full bg-secondary rounded-full overflow-hidden shadow-inner">
        <div
          className="h-full bg-gradient-to-r from-primary via-purple-500 to-blue-500 transition-all duration-1000 ease-out rounded-full"
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
};

const Skills = () => {
  const [isVisible, setIsVisible] = useState(false);
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
    <section id="skills" ref={sectionRef} className="py-20 bg-secondary/50 relative overflow-hidden">
      <div className="absolute top-0 left-0 translate-x-1/4 -translate-y-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl"></div>
      <div className="absolute bottom-0 right-0 -translate-x-1/4 translate-y-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl"></div>
      
      <div className="section-container">
        <div
          className={`transform transition-all duration-700 ${
            isVisible
              ? "translate-y-0 opacity-100"
              : "translate-y-10 opacity-0"
          }`}
        >
          <h2 className="section-title">Skills & Expertise</h2>
          <p className="section-subtitle">
            My technical competencies across AI, machine learning, and software development.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {SKILLS_DATA.map((category, index) => (
              <div
                key={category.category}
                className={`bg-white backdrop-blur-sm bg-white/90 rounded-xl p-6 shadow-md border border-border ${
                  isVisible
                    ? "animate-fade-up opacity-100"
                    : "opacity-0"
                }`}
                style={{ animationDelay: `${0.15 * index}s` }}
              >
                <div className="flex items-center mb-5">
                  <div className="p-2.5 bg-gradient-to-br from-primary/20 to-purple-500/20 rounded-lg text-primary mr-3">
                    {category.icon}
                  </div>
                  <h3 className="font-bold text-lg">{category.category}</h3>
                </div>
                
                <div>
                  {category.skills.map((skill) => (
                    <SkillBar
                      key={skill.name}
                      name={skill.name}
                      level={skill.level}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-16 p-6 bg-white/70 backdrop-blur-sm rounded-xl shadow-md">
            <h3 className="text-xl font-bold mb-4 text-center">Certificates & Additional Qualifications</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
              {["Modern Computer Vision", "Software Engineering", "Cloud Foundation", "Android Development", "Learn Statistics with Python"].map((cert, i) => (
                <div 
                  key={i}
                  className={`p-3 text-center rounded-lg border border-primary/20 bg-white shadow-sm ${
                    isVisible ? "animate-fade-up" : "opacity-0"
                  }`}
                  style={{ animationDelay: `${0.5 + (0.1 * i)}s` }}
                >
                  <span className="text-sm font-medium">{cert}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Skills;
