import { useState } from "react";
import { Download, Info, BarChart3, Map } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const Results = () => {
  const [selectedView, setSelectedView] = useState<'overlay' | 'comparison'>('overlay');

  const analysisData = {
    totalArea: "2.5 km²",
    noChange: "65%",
    minorDamage: "20%",
    majorDamage: "10%",
    destroyed: "5%",
    processingTime: "2.3 seconds",
    confidence: "94%"
  };

  const legendItems = [
    { color: "status-none", label: "No Damage", percentage: "65%", description: "Areas with no detectable change" },
    { color: "status-minor", label: "Minor Damage", percentage: "20%", description: "Light structural damage or debris" },
    { color: "status-major", label: "Major Damage", percentage: "10%", description: "Significant structural damage" },
    { color: "status-destroyed", label: "Destroyed", percentage: "5%", description: "Complete destruction or collapse" }
  ];

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Change Detection Results
          </h1>
          <p className="text-lg text-muted-foreground">
            AI-powered analysis of your satellite imagery with detailed damage assessment
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Results Display */}
          <div className="lg:col-span-2 space-y-6">
            {/* View Toggle */}
            <div className="flex space-x-2">
              <Button
                variant={selectedView === 'overlay' ? 'default' : 'outline'}
                onClick={() => setSelectedView('overlay')}
                className="flex items-center space-x-2"
              >
                <Map className="w-4 h-4" />
                <span>Change Overlay</span>
              </Button>
              <Button
                variant={selectedView === 'comparison' ? 'default' : 'outline'}
                onClick={() => setSelectedView('comparison')}
                className="flex items-center space-x-2"
              >
                <BarChart3 className="w-4 h-4" />
                <span>Side by Side</span>
              </Button>
            </div>

            {/* Result Image Display */}
            <Card className="card-professional">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Analysis Results</span>
                  <div className="flex space-x-2">
                    <Badge variant="secondary" className="bg-success/10 text-success">
                      Confidence: {analysisData.confidence}
                    </Badge>
                    <Button size="sm" variant="outline">
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </Button>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {selectedView === 'overlay' ? (
                  <div className="relative">
                    <div className="w-full h-96 bg-gradient-to-br from-muted to-accent rounded-xl flex items-center justify-center">
                      <div className="text-center">
                        <Map className="w-16 h-16 text-primary mx-auto mb-4" />
                        <h3 className="text-xl font-semibold text-foreground mb-2">
                          Change Detection Overlay
                        </h3>
                        <p className="text-muted-foreground">
                          Color-coded analysis showing damage levels across the region
                        </p>
                      </div>
                    </div>
                    <div className="absolute top-4 right-4 bg-card/90 backdrop-blur-sm rounded-lg p-3">
                      <div className="text-xs text-muted-foreground mb-1">Processing Time</div>
                      <div className="font-semibold">{analysisData.processingTime}</div>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-medium text-foreground mb-2">Pre-Disaster</h4>
                      <div className="w-full h-48 bg-muted rounded-lg flex items-center justify-center">
                        <div className="text-center">
                          <Map className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                          <p className="text-sm text-muted-foreground">Before Image</p>
                        </div>
                      </div>
                    </div>
                    <div>
                      <h4 className="font-medium text-foreground mb-2">Post-Disaster</h4>
                      <div className="w-full h-48 bg-muted rounded-lg flex items-center justify-center">
                        <div className="text-center">
                          <Map className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                          <p className="text-sm text-muted-foreground">After Image</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Sidebar with Legend and Statistics */}
          <div className="space-y-6">
            {/* Color Legend */}
            <Card className="card-professional">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Info className="w-5 h-5 text-primary" />
                  <span>Damage Legend</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {legendItems.map((item, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className={`w-6 h-6 rounded-lg ${item.color} flex-shrink-0 mt-0.5`} />
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-foreground">{item.label}</span>
                        <Badge variant="outline" className="text-xs">
                          {item.percentage}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground">{item.description}</p>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Analysis Statistics */}
            <Card className="card-professional">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5 text-primary" />
                  <span>Analysis Summary</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Total Area</p>
                    <p className="text-lg font-semibold text-foreground">{analysisData.totalArea}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Confidence</p>
                    <p className="text-lg font-semibold text-success">{analysisData.confidence}</p>
                  </div>
                </div>
                
                <div className="space-y-3 pt-4 border-t border-border">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">No Change</span>
                    <span className="font-medium text-success">{analysisData.noChange}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Minor Damage</span>
                    <span className="font-medium text-warning">{analysisData.minorDamage}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Major Damage</span>
                    <span className="font-medium text-orange">{analysisData.majorDamage}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Destroyed</span>
                    <span className="font-medium text-destructive">{analysisData.destroyed}</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Actions */}
            <div className="space-y-3">
              <Button className="w-full" variant="default">
                <Download className="w-4 h-4 mr-2" />
                Export Report
              </Button>
              <Button className="w-full" variant="outline">
                <BarChart3 className="w-4 h-4 mr-2" />
                Detailed Analysis
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;