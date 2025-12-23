import { Target, Wrench, MapPin, Shield, Users, Zap } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const About = () => {
  const objectives = [
    {
      icon: Target,
      title: "Accurate Change Detection",
      description: "Precisely identify and classify changes in satellite imagery using advanced AI algorithms"
    },
    {
      icon: Shield,
      title: "Disaster Assessment",
      description: "Provide rapid and reliable damage assessment for emergency response and recovery planning"
    },
    {
      icon: MapPin,
      title: "Geographic Intelligence",
      description: "Generate actionable geographic intelligence for decision-makers and first responders"
    }
  ];

  const tools = [
    {
      category: "AI & Machine Learning",
      items: ["TensorFlow", "PyTorch", "OpenCV", "Scikit-learn", "NumPy"]
    },
    // {
    //   category: "Satellite Data Processing",
    //   items: ["GDAL", "Sentinel-2", "Landsat", "MODIS", "Planet Labs API"]
    // },
    {
      category: "Web Technologies",
      items: ["React", "TypeScript", "Tailwind CSS", "Node.js", "Python Flask"]
    },
    // {
    //   category: "Cloud & Infrastructure",
    //   items: ["AWS", "Google Earth Engine", "Docker", "Kubernetes", "PostgreSQL"]
    // }
  ];

  const applications = [
    {
      icon: Shield,
      title: "Disaster Response",
      description: "Rapid damage assessment for hurricanes, earthquakes, floods, and wildfires",
      examples: ["Emergency response planning", "Resource allocation", "Recovery prioritization"]
    },
    {
      icon: Users,
      title: "Urban Planning",
      description: "Monitor urban development and infrastructure changes over time",
      examples: ["Construction monitoring", "Land use changes", "Infrastructure development"]
    },
    // {
    //   icon: MapPin,
    //   title: "Environmental Monitoring",
    //   description: "Track environmental changes and natural resource management",
    //   examples: ["Deforestation tracking", "Coastal erosion", "Agricultural monitoring"]
    // },
    // {
    //   icon: Zap,
    //   title: "Insurance & Risk Assessment",
    //   description: "Support insurance claims processing and risk evaluation",
    //   examples: ["Damage verification", "Risk modeling", "Claims automation"]
    // }
  ];

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            About GeoSense
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Advanced AI-based satellite image change detection system designed for rapid disaster assessment, 
            urban planning, and environmental monitoring applications.
          </p>
        </div>

        {/* Project Overview */}
        <section className="mb-16">
          <Card className="card-professional">
            <CardHeader>
              <CardTitle className="text-2xl">Project Overview</CardTitle>
            </CardHeader>
            <CardContent className="prose prose-gray max-w-none">
              <p className="text-muted-foreground leading-relaxed">
                GeoSense is a cutting-edge artificial intelligence system that analyzes satellite imagery to detect 
                and classify changes over time. Our system combines advanced machine learning algorithms with 
                high-resolution satellite data to provide accurate, rapid assessment of damage from natural disasters, 
                urban development changes, and environmental modifications.
              </p>
              <p className="text-muted-foreground leading-relaxed mt-4">
                Built with modern web technologies and powered by state-of-the-art AI models, GeoSense offers 
                an intuitive interface for uploading satellite images and receiving detailed change detection 
                analysis with color-coded damage assessments and confidence metrics.
              </p>
            </CardContent>
          </Card>
        </section>

        {/* Objectives */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-foreground mb-8 text-center">Project Objectives</h2>
          <div className="grid md:grid-cols-3 gap-6">
            {objectives.map((objective, index) => (
              <Card key={index} className="card-professional">
                <CardHeader>
                  <div className="w-12 h-12 bg-gradient-to-r from-primary to-secondary rounded-xl flex items-center justify-center mb-4">
                    <objective.icon className="w-6 h-6 text-white" />
                  </div>
                  <CardTitle className="text-xl">{objective.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">{objective.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Tools & Technologies */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-foreground mb-8 text-center">Tools & Technologies</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {tools.map((toolCategory, index) => (
              <Card key={index} className="card-professional">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Wrench className="w-5 h-5 text-primary" />
                    <span>{toolCategory.category}</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {toolCategory.items.map((tool, toolIndex) => (
                      <span 
                        key={toolIndex}
                        className="inline-block bg-accent text-accent-foreground px-3 py-1 rounded-full text-sm font-medium"
                      >
                        {tool}
                      </span>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Applications */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-foreground mb-8 text-center">Key Applications</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {applications.map((application, index) => (
              <Card key={index} className="card-professional">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-r from-primary to-secondary rounded-lg flex items-center justify-center">
                      <application.icon className="w-5 h-5 text-white" />
                    </div>
                    <span>{application.title}</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-muted-foreground">{application.description}</p>
                  <div>
                    <h4 className="font-medium text-foreground mb-2">Use Cases:</h4>
                    <ul className="space-y-1">
                      {application.examples.map((example, exampleIndex) => (
                        <li key={exampleIndex} className="text-sm text-muted-foreground flex items-center">
                          <div className="w-1.5 h-1.5 bg-primary rounded-full mr-2 flex-shrink-0" />
                          {example}
                        </li>
                      ))}
                    </ul>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Technical Specifications */}
        <section>
          <h2 className="text-3xl font-bold text-foreground mb-8 text-center">Technical Specifications</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="card-professional">
              <CardHeader>
                <CardTitle>System Capabilities</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Processing Speed</span>
                  <span className="font-medium">2-5 seconds per analysis</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Supported Formats</span>
                  <span className="font-medium">PNG, JPEG, TIFF, GeoTIFF</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Maximum Resolution</span>
                  <span className="font-medium">1024 x 1024 pixels</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Accuracy</span>
                  <span className="font-medium text-success">85-90%</span>
                </div>
              </CardContent>
            </Card>
            
            <Card className="card-professional">
              <CardHeader>
                <CardTitle>Classification System</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center space-x-3">
                  <div className="w-4 h-4 status-none rounded"></div>
                  <span className="text-muted-foreground">No Damage</span>
                  <span className="font-medium ml-auto">Green</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-4 h-4 status-minor rounded"></div>
                  <span className="text-muted-foreground">Minor Damage</span>
                  <span className="font-medium ml-auto">Yellow</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-4 h-4 status-major rounded"></div>
                  <span className="text-muted-foreground">Major Damage</span>
                  <span className="font-medium ml-auto">Orange</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-4 h-4 status-destroyed rounded"></div>
                  <span className="text-muted-foreground">Destroyed</span>
                  <span className="font-medium ml-auto">Red</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>
      </div>
    </div>
  );
};

export default About;