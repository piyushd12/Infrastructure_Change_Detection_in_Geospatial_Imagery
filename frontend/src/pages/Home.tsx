import { useNavigate } from "react-router-dom";
import { ArrowRight, Satellite, Zap, Target, Globe } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import heroImage from "@/assets/hero-satellite.jpg";

const Home = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Satellite,
      title: "AI-Powered Analysis",
      description: "Advanced machine learning algorithms analyze satellite imagery for precise change detection"
    },
    {
      icon: Zap,
      title: "Real-time Processing",
      description: "Fast and efficient processing of high-resolution satellite images with immediate results"
    },
    {
      icon: Target,
      title: "Damage Assessment",
      description: "Accurate classification of damage levels from minor changes to complete destruction"
    },
    {
      icon: Globe,
      title: "Global Coverage",
      description: "Support for satellite imagery from any location worldwide with standardized analysis"
    }
  ];

  return (
    <div className="min-h-screen">
      {/* Luxury Hero Section */}
      <section className="hero-background min-h-screen flex items-center justify-center relative overflow-hidden">
        {/* Animated Background Elements */}
        <div className="absolute inset-0 opacity-30">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-r from-primary/20 to-secondary/20 rounded-full blur-3xl floating-element" />
          <div className="absolute top-3/4 right-1/4 w-80 h-80 bg-gradient-to-r from-secondary/20 to-primary/20 rounded-full blur-3xl floating-element" style={{animationDelay: '2s'}} />
          <div className="absolute top-1/2 right-1/2 w-64 h-64 bg-gradient-to-r from-primary/10 to-secondary/10 rounded-full blur-2xl floating-element" style={{animationDelay: '4s'}} />
        </div>
        
        {/* Geometric Accent Lines */}
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-0 left-1/3 w-px h-full bg-gradient-to-b from-transparent via-white to-transparent" />
          <div className="absolute top-0 right-1/3 w-px h-full bg-gradient-to-b from-transparent via-white to-transparent" />
          <div className="absolute top-1/3 left-0 w-full h-px bg-gradient-to-r from-transparent via-white to-transparent" />
        </div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 z-10">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div className="space-y-10">
              <div className="space-y-6">
                <div className="inline-flex items-center px-4 py-2 rounded-full glass-card text-white/90 text-sm font-medium mb-4">
                  <Satellite className="w-4 h-4 mr-2" />
                  Infrastructure Change Detection
                </div>
                <h1 className="text-6xl lg:text-7xl font-bold text-white leading-tight tracking-tight">
                  <span className="bg-gradient-to-r from-white via-white to-white/80 bg-clip-text text-transparent">
                    GeoSense
                  </span>
                </h1>
                <p className="text-2xl lg:text-3xl text-white/90 font-light leading-relaxed">
                  AI-Powered Geospatial Intelligence
                </p>
                <p className="text-xl text-white/70 max-w-2xl leading-relaxed font-light">
                  Revolutionary artificial intelligence system for analyzing satellite imagery and detecting infrastructure changes over time. Advanced algorithms provide precise damage assessment for disaster response, urban planning, and environmental monitoring.
                </p>
              </div>
              
              <div className="flex flex-col sm:flex-row gap-6">
                <Button 
                  className="btn-hero text-xl px-10 py-5"
                  onClick={() => navigate('/upload')}
                >
                  Launch Analysis
                  <ArrowRight className="ml-3 w-6 h-6" />
                </Button>
                <Button 
                  variant="outline" 
                  size="lg"
                  onClick={() => navigate('/about')}
                  className="glass-card border-white/30 text-white hover:bg-white/10 text-lg px-8 py-5 font-medium"
                >
                  Explore Technology
                </Button>
              </div>
              
              {/* Stats Section */}
              <div className="grid grid-cols-3 gap-8 pt-8">
                <div className="text-center">
                  <div className="text-3xl font-bold text-white">99.7%</div>
                  <div className="text-sm text-white/60 font-medium">Accuracy</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-white">&lt; 5min</div>
                  <div className="text-sm text-white/60 font-medium">Processing</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-white">Global</div>
                  <div className="text-sm text-white/60 font-medium">Coverage</div>
                </div>
              </div>
            </div>
            
            <div className="relative lg:ml-8">
              {/* Main Image Container */}
              <div className="relative">
                <div className="glass-card p-2 rounded-3xl shadow-floating">
                  <div className="rounded-2xl overflow-hidden">
                    <img 
                      src={heroImage} 
                      alt="Advanced satellite image change detection and infrastructure analysis" 
                      className="w-full h-auto transform hover:scale-105 transition-transform duration-700"
                    />
                  </div>
                </div>
                
                {/* Floating UI Elements */}
                <div className="absolute -top-6 -left-6 glass-card p-4 rounded-2xl floating-element">
                  <div className="flex items-center space-x-3">
                    <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                    <span className="text-white text-sm font-medium">Live Analysis</span>
                  </div>
                </div>
                
                <div className="absolute -bottom-8 -right-8 glass-card p-6 rounded-2xl floating-element" style={{animationDelay: '1s'}}>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">AI</div>
                    <div className="text-xs text-white/60">Powered</div>
                  </div>
                </div>
                
                {/* Ambient Light Effects */}
                <div className="absolute -inset-4 bg-gradient-to-r from-primary/20 to-secondary/20 rounded-3xl blur-2xl -z-10 opacity-60" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Premium Features Section */}
      <section className="py-32 bg-gradient-to-b from-background to-muted/30 relative overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-5">
          <div className="absolute inset-0" style={{
            backgroundImage: `radial-gradient(circle at 2px 2px, hsl(var(--primary)) 1px, transparent 0)`,
            backgroundSize: '48px 48px'
          }} />
        </div>
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
          <div className="text-center mb-20">
            <div className="inline-flex items-center px-6 py-3 rounded-full bg-gradient-to-r from-primary/10 to-secondary/10 border border-primary/20 text-primary font-medium mb-8">
              <Zap className="w-5 h-5 mr-2" />
              Advanced Technology Suite
            </div>
            <h2 className="text-5xl lg:text-6xl font-bold text-foreground mb-6 leading-tight">
              Next-Generation
              <br />
              <span className="bg-gradient-to-r from-primary via-secondary to-primary bg-clip-text text-transparent">
                Geospatial Intelligence
              </span>
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              Our revolutionary AI system transforms satellite imagery analysis with unprecedented accuracy and speed, delivering actionable insights for critical infrastructure monitoring and disaster response.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <Card key={index} className="group glass-card p-8 h-full hover:shadow-floating transition-all duration-500 hover:-translate-y-2 border-white/10">
                <CardContent className="p-0 flex flex-col items-center text-center space-y-6">
                  <div className="relative">
                    <div className="w-20 h-20 bg-gradient-to-br from-primary via-secondary to-primary rounded-3xl flex items-center justify-center shadow-glow group-hover:scale-110 transition-transform duration-500">
                      <feature.icon className="w-10 h-10 text-white" />
                    </div>
                    <div className="absolute -inset-2 bg-gradient-to-r from-primary/20 to-secondary/20 rounded-3xl blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                  </div>
                  <h3 className="text-2xl font-bold text-foreground group-hover:text-primary transition-colors duration-300">
                    {feature.title}
                  </h3>
                  <p className="text-muted-foreground leading-relaxed text-base">
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Premium CTA Section */}
      <section className="py-32 hero-background relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-transparent" />
        
        {/* Animated Elements */}
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-1/4 left-1/6 w-72 h-72 bg-gradient-to-r from-white/10 to-transparent rounded-full blur-3xl floating-element" />
          <div className="absolute bottom-1/4 right-1/6 w-96 h-96 bg-gradient-to-r from-white/5 to-transparent rounded-full blur-3xl floating-element" style={{animationDelay: '3s'}} />
        </div>
        
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
          <div className="glass-card p-16 rounded-3xl">
            <h2 className="text-4xl lg:text-5xl font-bold text-white mb-8 leading-tight">
              Transform Your
              <br />
              <span className="bg-gradient-to-r from-white via-white to-white/70 bg-clip-text text-transparent">
                Infrastructure Analysis
              </span>
            </h2>
            <p className="text-2xl text-white/80 mb-12 max-w-3xl mx-auto leading-relaxed font-light">
              Experience the future of satellite imagery analysis. Upload your geospatial data and receive comprehensive change detection insights powered by cutting-edge AI technology.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
              <Button 
                size="lg"
                className="btn-hero text-xl px-12 py-6"
                onClick={() => navigate('/upload')}
              >
                Begin Analysis
                <ArrowRight className="ml-3 w-6 h-6" />
              </Button>
              <div className="flex items-center space-x-4 text-white/60">
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse" />
                  <span className="text-sm">No signup required</span>
                </div>
                <div className="w-px h-4 bg-white/20" />
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-blue-400 rounded-full mr-2 animate-pulse" />
                  <span className="text-sm">Instant results</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;