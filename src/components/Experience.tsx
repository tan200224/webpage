
import { useState, useRef, useEffect } from 'react';
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Briefcase, Calendar } from "lucide-react";

// Sample work experience - replace with your actual experience
const EXPERIENCE_DATA = [
  {
    id: 1,
    role: "Senior Software Engineer",
    company: "Tech Innovations Inc.",
    date: "2021 - Present",
    description: "Lead developer for enterprise SaaS platform. Architected microservices infrastructure and implemented CI/CD pipeline that increased deployment frequency by 70%.",
    achievements: [
      "Reduced API response time by 40% through query optimization",
      "Led team of 5 developers to deliver major platform overhaul",
      "Implemented automated testing that reduced bugs in production by 35%"
    ],
    skills: ["React", "Node.js", "TypeScript", "AWS", "Docker", "Kubernetes"]
  },
  {
    id: 2,
    role: "Full Stack Developer",
    company: "Digital Solutions",
    date: "2018 - 2021",
    description: "Built and maintained web applications for clients in finance, healthcare, and e-commerce sectors.",
    achievements: [
      "Developed custom CRM that increased sales team efficiency by 25%",
      "Created real-time dashboard used by 10,000+ users daily",
      "Redesigned authentication system improving security and UX"
    ],
    skills: ["JavaScript", "React", "Python", "Django", "PostgreSQL", "Redis"]
  },
  {
    id: 3,
    role: "Software Developer",
    company: "WebApps Ltd",
    date: "2016 - 2018",
    description: "Worked on consumer-facing mobile applications with focus on performance and offline capabilities.",
    achievements: [
      "Developed offline-first architecture for field service application",
      "Optimized app load time from 6s to 2s through code splitting",
      "Implemented analytics system that drove product roadmap decisions"
    ],
    skills: ["React Native", "GraphQL", "Firebase", "MobX", "Jest"]
  }
];

const Experience = () => {
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
    <section id="experience" ref={sectionRef} className="py-20 bg-secondary/50">
      <div className="section-container">
        <div
          className={`transform transition-all duration-700 ${
            isVisible
              ? "translate-y-0 opacity-100"
              : "translate-y-10 opacity-0"
          }`}
        >
          <h2 className="section-title">Experience</h2>
          <p className="section-subtitle">
            My professional journey and key roles that have shaped my expertise.
          </p>

          <div className="space-y-12">
            {EXPERIENCE_DATA.map((job, index) => (
              <Card
                key={job.id}
                className={`overflow-hidden border-0 shadow-md card-hover ${
                  isVisible
                    ? "animate-fade-up opacity-100"
                    : "opacity-0"
                }`}
                style={{ animationDelay: `${0.2 * index}s` }}
              >
                <CardHeader className="bg-background border-b p-6">
                  <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-4">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <Briefcase className="h-4 w-4 text-primary" />
                        <span className="font-bold text-lg">{job.role}</span>
                      </div>
                      <div className="text-muted-foreground">{job.company}</div>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">{job.date}</span>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="p-6">
                  <p className="mb-4">{job.description}</p>
                  <div className="mb-6">
                    <h4 className="text-sm font-semibold mb-2">Key Achievements:</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm">
                      {job.achievements.map((achievement, i) => (
                        <li key={i} className="text-muted-foreground">
                          {achievement}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {job.skills.map((skill, i) => (
                      <Badge key={i} variant="outline" className="rounded-full">
                        {skill}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Experience;
