
import { useState, useRef, useEffect } from 'react';
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Briefcase, Calendar, GraduationCap, Award, Lightbulb } from "lucide-react";

// Archie's actual work experience data
const EXPERIENCE_DATA = [
  {
    id: 1,
    role: "Machine Learning Researcher",
    company: "Elon University",
    date: "Apr. 2023 - Present",
    description: "Led research on early diagnosis of pancreatic cancer using AI models and synthetic data, secured $20,000+ grants.",
    achievements: [
      "Built and trained segmentation model to achieve 87.16% dice accuracy in pancreas CT-Scan segmentation",
      "Designed scalable ML pipeline with customized data augmentation, achieved 10%+ improvement",
      "Built Annotation-to-3D-CT-Scan generative model from scratch with PyTorch"
    ],
    skills: ["PyTorch", "Torchvision", "ML Pipelines", "Data Augmentation", "3D Image Processing"],
    icon: <Lightbulb className="h-5 w-5 text-purple-500" />
  },
  {
    id: 2,
    role: "Software Engineer Intern",
    company: "CiPASS",
    date: "Jun. 2022 - Aug. 2022",
    description: "Collaborated with NYC DOT to analyze transit needs, delivering data-driven strategic plans impacting over 180,000 residents.",
    achievements: [
      "Bent the development from building geographical images to an interactable real-time visualization application",
      "Led discovery of demographic and transit vulnerability assessment, identifying 17+ strategies",
      "Designed and developed scalable data processing and data visualization pipeline using Python and QGIS"
    ],
    skills: ["Python", "QGIS", "Data Visualization", "Data Processing", "Real-time Applications"],
    icon: <Briefcase className="h-5 w-5 text-blue-500" />
  },
  {
    id: 3,
    role: "Operations Committee Member & Software Engineer Intern",
    company: "CAN International",
    date: "Oct. 2021 - Mar. 2024",
    description: "Led AI-integrated chatbot project and redesigned webpage, improving user experience and responsiveness.",
    achievements: [
      "Led and initiated AI-integrated chatbot project, using fine-tuning and data retrieval with OpenAI API",
      "Identified critical webpage vulnerabilities and redesigned using HTML and CSS",
      "Initiated a partnership with City College, bent the growth curve of company funding by 6%, and size by 12%"
    ],
    skills: ["OpenAI API", "HTML", "CSS", "JavaScript", "Fine-tuning", "Data Retrieval"],
    icon: <Briefcase className="h-5 w-5 text-green-500" />
  },
  {
    id: 4,
    role: "B.S. in Computer Science",
    company: "Elon University",
    date: "Expected May 2025",
    description: "GPA: 3.93/4.0 | Minors in Math and Data Science",
    achievements: [
      "Lumen Prize ($20,000 award, 2023)",
      "Golden Door Scholarship (4-year full scholarship, 2022)",
      "Provost Scholar (Outstanding Researcher, 2025)",
      "President's List (5 Semesters) & Dean's List (1 Semester)",
      "Phi Beta Kappa (2025) & Phi Kappa Phi (2024)"
    ],
    skills: ["Computer Science", "Mathematics", "Data Science", "Research", "Academic Leadership"],
    icon: <GraduationCap className="h-5 w-5 text-primary" />
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
    <section id="experience" ref={sectionRef} className="py-20 bg-secondary/50 relative overflow-hidden">
      <div className="absolute top-0 right-0 -translate-x-1/4 -translate-y-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl"></div>
      <div className="absolute bottom-0 left-0 translate-x-1/4 translate-y-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl"></div>
      
      <div className="section-container">
        <div
          className={`transform transition-all duration-700 ${
            isVisible
              ? "translate-y-0 opacity-100"
              : "translate-y-10 opacity-0"
          }`}
        >
          <h2 className="section-title">Experience & Education</h2>
          <p className="section-subtitle">
            My professional journey combining machine learning research and software development.
          </p>

          <div className="space-y-12 relative">
            {/* Timeline connector */}
            <div className="absolute left-8 top-8 bottom-8 w-0.5 bg-gradient-to-b from-primary via-purple-500 to-blue-500 hidden md:block"></div>
            
            {EXPERIENCE_DATA.map((job, index) => (
              <Card
                key={job.id}
                className={`overflow-hidden border-0 shadow-md card-hover relative ${
                  isVisible
                    ? "animate-fade-up opacity-100"
                    : "opacity-0"
                }`}
                style={{ animationDelay: `${0.2 * index}s` }}
              >
                <div className="absolute -left-3 top-6 w-6 h-6 rounded-full bg-white border-2 border-primary z-10 hidden md:flex items-center justify-center">
                  {job.icon}
                </div>
                
                <CardHeader className="bg-background border-b p-6">
                  <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-4">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <div className="p-1.5 rounded-full bg-primary/10 md:hidden">
                          {job.icon}
                        </div>
                        <span className="font-bold text-lg">{job.role}</span>
                      </div>
                      <div className="text-muted-foreground">{job.company}</div>
                    </div>
                    <div className="flex items-center gap-1.5 bg-secondary/50 px-3 py-1 rounded-full">
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm font-mono text-muted-foreground">{job.date}</span>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="p-6">
                  <p className="mb-4">{job.description}</p>
                  <div className="mb-6">
                    <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                      <Award className="h-4 w-4 text-primary" />
                      <span>Key Achievements:</span>
                    </h4>
                    <ul className="space-y-2 text-sm">
                      {job.achievements.map((achievement, i) => (
                        <li key={i} className="text-muted-foreground flex items-start">
                          <span className="mr-2 text-primary">•</span>
                          <span>{achievement}</span>
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
