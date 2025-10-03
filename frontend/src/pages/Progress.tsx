import { CheckCircle, Clock, AlertCircle, Calendar } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress as ProgressBar } from "@/components/ui/progress";

const Progress = () => {
  const analysisHistory = [
    {
      id: 1,
      name: "Hurricane Damage Assessment - Florida",
      status: "completed",
      progress: 100,
      uploadTime: "2024-01-15 14:30",
      completionTime: "2024-01-15 14:33",
      confidence: "96%",
      totalArea: "15.2 km²",
      damages: { none: "45%", minor: "30%", major: "15%", destroyed: "10%" }
    },
    {
      id: 2,
      name: "Wildfire Impact Analysis - California",
      status: "completed",
      progress: 100,
      uploadTime: "2024-01-12 09:15",
      completionTime: "2024-01-12 09:18",
      confidence: "94%",
      totalArea: "8.7 km²",
      damages: { none: "25%", minor: "20%", major: "35%", destroyed: "20%" }
    },
    {
      id: 3,
      name: "Earthquake Assessment - Turkey",
      status: "processing",
      progress: 75,
      uploadTime: "2024-01-16 11:45",
      completionTime: null,
      confidence: null,
      totalArea: "12.3 km²",
      damages: null
    },
    {
      id: 4,
      name: "Flood Damage Survey - Bangladesh",
      status: "pending",
      progress: 0,
      uploadTime: "2024-01-16 16:20",
      completionTime: null,
      confidence: null,
      totalArea: "22.1 km²",
      damages: null
    }
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-success" />;
      case 'processing':
        return <Clock className="w-5 h-5 text-warning animate-pulse" />;
      case 'pending':
        return <AlertCircle className="w-5 h-5 text-muted-foreground" />;
      default:
        return <Clock className="w-5 h-5 text-muted-foreground" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge className="status-none">Completed</Badge>;
      case 'processing':
        return <Badge className="status-minor">Processing</Badge>;
      case 'pending':
        return <Badge variant="outline">Pending</Badge>;
      default:
        return <Badge variant="outline">Unknown</Badge>;
    }
  };

  const currentStats = {
    totalAnalyses: analysisHistory.length,
    completed: analysisHistory.filter(a => a.status === 'completed').length,
    processing: analysisHistory.filter(a => a.status === 'processing').length,
    pending: analysisHistory.filter(a => a.status === 'pending').length
  };

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Analysis Progress
          </h1>
          <p className="text-lg text-muted-foreground">
            Track your satellite image analysis progress and view processing history
          </p>
        </div>

        {/* Summary Stats */}
        <div className="grid md:grid-cols-4 gap-4 mb-8">
          <Card className="card-professional">
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-primary mb-2">{currentStats.totalAnalyses}</div>
              <div className="text-sm text-muted-foreground">Total Analyses</div>
            </CardContent>
          </Card>
          <Card className="card-professional">
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-success mb-2">{currentStats.completed}</div>
              <div className="text-sm text-muted-foreground">Completed</div>
            </CardContent>
          </Card>
          <Card className="card-professional">
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-warning mb-2">{currentStats.processing}</div>
              <div className="text-sm text-muted-foreground">Processing</div>
            </CardContent>
          </Card>
          <Card className="card-professional">
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-muted-foreground mb-2">{currentStats.pending}</div>
              <div className="text-sm text-muted-foreground">Pending</div>
            </CardContent>
          </Card>
        </div>

        {/* Analysis History */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold text-foreground mb-6">Analysis History</h2>
          
          {analysisHistory.map((analysis) => (
            <Card key={analysis.id} className="card-professional">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(analysis.status)}
                    <div>
                      <CardTitle className="text-lg">{analysis.name}</CardTitle>
                      <div className="flex items-center space-x-2 text-sm text-muted-foreground mt-1">
                        <Calendar className="w-4 h-4" />
                        <span>Uploaded: {analysis.uploadTime}</span>
                        {analysis.completionTime && (
                          <span>• Completed: {analysis.completionTime}</span>
                        )}
                      </div>
                    </div>
                  </div>
                  {getStatusBadge(analysis.status)}
                </div>
              </CardHeader>
              <CardContent>
                {/* Progress Bar */}
                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-muted-foreground">Progress</span>
                    <span className="font-medium">{analysis.progress}%</span>
                  </div>
                  <ProgressBar value={analysis.progress} className="h-2" />
                </div>

                {/* Analysis Details */}
                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground">Total Area</p>
                    <p className="font-semibold">{analysis.totalArea}</p>
                  </div>
                  {analysis.confidence && (
                    <div>
                      <p className="text-xs text-muted-foreground">Confidence</p>
                      <p className="font-semibold text-success">{analysis.confidence}</p>
                    </div>
                  )}
                  {analysis.damages && (
                    <>
                      <div>
                        <p className="text-xs text-muted-foreground">No Damage</p>
                        <p className="font-semibold text-success">{analysis.damages.none}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Destroyed</p>
                        <p className="font-semibold text-destructive">{analysis.damages.destroyed}</p>
                      </div>
                    </>
                  )}
                </div>

                {/* Damage Breakdown for Completed Analyses */}
                {analysis.damages && (
                  <div className="mt-4 pt-4 border-t border-border">
                    <h4 className="text-sm font-medium text-foreground mb-3">Damage Distribution</h4>
                    <div className="grid grid-cols-4 gap-2">
                      <div className="text-center">
                        <div className="w-full h-2 status-none rounded mb-1"></div>
                        <p className="text-xs text-muted-foreground">No Damage</p>
                        <p className="text-sm font-medium">{analysis.damages.none}</p>
                      </div>
                      <div className="text-center">
                        <div className="w-full h-2 status-minor rounded mb-1"></div>
                        <p className="text-xs text-muted-foreground">Minor</p>
                        <p className="text-sm font-medium">{analysis.damages.minor}</p>
                      </div>
                      <div className="text-center">
                        <div className="w-full h-2 status-major rounded mb-1"></div>
                        <p className="text-xs text-muted-foreground">Major</p>
                        <p className="text-sm font-medium">{analysis.damages.major}</p>
                      </div>
                      <div className="text-center">
                        <div className="w-full h-2 status-destroyed rounded mb-1"></div>
                        <p className="text-xs text-muted-foreground">Destroyed</p>
                        <p className="text-sm font-medium">{analysis.damages.destroyed}</p>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Progress;