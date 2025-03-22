
import { useState, useRef, useEffect } from 'react';
import { 
  Database, 
  Server, 
  Code, 
  Workflow, 
  Laptop, 
  Layers
} from "lucide-react";

// Sample skill data - replace with your actual skills
const SKILLS_DATA = [
  {
    category: "Front-End",
    icon: <Laptop className="h-6 w-6" />,
    skills: [
      { name: "React / React Native", level: 90 },
      { name: "JavaScript / TypeScript", level: 95 },
      { name: "HTML / CSS", level: 90 },
      { name: "Redux", level: 85 },
      { name: "Next.js", level: 80 },
    ]
  },
  {
    category: "Back-End",
    icon: <Server className="h-6 w-6" />,
    skills: [
      { name: "Node.js", level: 85 },
      { name: "Python", level: 80 },
      { name: "Express", level: 90 },
      { name: "Java", level: 75 },
      { name: "GraphQL", level: 80 },
    ]
  },
  {
    category: "Database",
    icon: <Database className="h-6 w-6" />,
    skills: [
      { name: "MongoDB", level: 85 },
      { name: "PostgreSQL", level: 80 },
      { name: "Firebase", level: 90 },
      { name: "Redis", level: 75 },
      { name: "MySQL", level: 80 },
    ]
  },
  {
    category: "DevOps",
    icon: <Workflow className="h-6 w-6" />,
    skills: [
      { name: "Docker", level: 85 },
      { name: "Kubernetes", level: 75 },
      { name: "CI/CD", level: 80 },
      { name: "AWS", level: 85 },
      { name: "GitHub Actions", level: 90 },
    ]
  },
  {
    category: "Tools",
    icon: <Code className="h-6 w-6" />,
    skills: [
      { name: "Git", level: 95 },
      { name: "Webpack", level: 85 },
      { name: "Jest / Testing Library", level: 80 },
      { name: "Figma", level: 75 },
      { name: "VS Code", level: 95 },
    ]
  },
  {
    category: "Architecture",
    icon: <Layers className="h-6 w-6" />,
    skills: [
      { name: "Microservices", level: 85 },
      { name: "RESTful APIs", level: 90 },
      { name: "Design Patterns", level: 85 },
      { name: "System Design", level: 80 },
      { name: "OOP / Functional", level: 90 },
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
      <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
        <div
          className="h-full bg-primary transition-all duration-1000 ease-out rounded-full"
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
    <section id="skills" ref={sectionRef} className="py-20 bg-secondary/50">
      <div className="section-container">
        <div
          className={`transform transition-all duration-700 ${
            isVisible
              ? "translate-y-0 opacity-100"
              : "translate-y-10 opacity-0"
          }`}
        >
          <h2 className="section-title">Skills</h2>
          <p className="section-subtitle">
            My technical expertise and proficiencies across different domains.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {SKILLS_DATA.map((category, index) => (
              <div
                key={category.category}
                className={`bg-white rounded-xl p-6 shadow-md border border-border ${
                  isVisible
                    ? "animate-fade-up opacity-100"
                    : "opacity-0"
                }`}
                style={{ animationDelay: `${0.15 * index}s` }}
              >
                <div className="flex items-center mb-5">
                  <div className="p-2 bg-primary/10 rounded-lg text-primary mr-3">
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
        </div>
      </div>
    </section>
  );
};

export default Skills;
