import { useEffect, useState } from "react";
import { Download, AlertTriangle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useAnalysis } from "@/context/AnalysisContext";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";

const Results = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { toast } = useToast();
  const {
    heatmapBase64,
    overlayBase64,
    preImagePreview,
    postImagePreview,
    statistics,
    analysisId,
    setHeatmapBase64,
    setOverlayBase64,
    setPreImagePreview,
    setPostImagePreview,
    setStatistics,
    setAnalysisId,
  } = useAnalysis();

  const [showOverlay, setShowOverlay] = useState(true); // Start with overlay shown
  const [loading, setLoading] = useState(true);
  const urlAnalysisId = searchParams.get("id");

  // Load analysis from backend if ID is in URL
  // === FIXED useEffect - no more flickering ===
  useEffect(() => {
    const loadAnalysis = async (id: string) => {
      setLoading(true);
      try {
        const res = await fetch(`http://localhost:5000/api/history/${id}`);
        if (!res.ok) throw new Error("Failed to load analysis");
        const data = await res.json();
        if (!data.success) throw new Error("Analysis not found");

        const result = data.result;
        const baseUrl = "http://localhost:5000/api/results";

        // Only set state if it's different (prevents re-trigger)
        if (analysisId !== id) {
          setAnalysisId(id);
        }
        if (preImagePreview !== `${baseUrl}/${id}/pre_disaster.jpg`) {
          setPreImagePreview(`${baseUrl}/${id}/pre_disaster.jpg`);
        }
        if (postImagePreview !== `${baseUrl}/${id}/post_disaster.jpg`) {
          setPostImagePreview(`${baseUrl}/${id}/post_disaster.jpg`);
        }
        if (overlayBase64 !== `${baseUrl}/${id}/overlay.png`) {
          setOverlayBase64(`${baseUrl}/${id}/overlay.png`);
        }
        if (heatmapBase64 !== `${baseUrl}/${id}/heatmap.png`) {
          setHeatmapBase64(`${baseUrl}/${id}/heatmap.png`);
        }
        if (JSON.stringify(statistics) !== JSON.stringify(result.statistics)) {
          setStatistics(result.statistics);
        }

        toast({
          title: "Analysis loaded",
          description: `From ${new Date(result.timestamp).toLocaleDateString()}`,
        });
      } catch (err: any) {
        toast({
          title: "Failed to load",
          description: err.message,
          variant: "destructive",
        });
        navigate("/upload");
      } finally {
        setLoading(false);
      }
    };

    // Case 1: URL has ?id= → load historical analysis
    if (urlAnalysisId) {
      loadAnalysis(urlAnalysisId);
      return;
    }

    // Case 2: Fresh analysis just completed (context already has data)
    if (overlayBase64 && statistics && analysisId) {
      setLoading(false);
      return;
    }

    // Case 3: Nothing available → go to upload
    navigate("/upload");
  }, [urlAnalysisId]); // ← Only depend on urlAnalysisId, not context state!
  // ================================================

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-lg">Loading analysis results...</p>
        </div>
      </div>
    );
  }

  if (!statistics || !overlayBase64) {
    navigate("/upload");
    return null;
  }

  const percentages = statistics.damage_percentages || {};
  const totalDamaged = percentages["Minor Damage"] + percentages["Major Damage"] + percentages["Destroyed"];

  const handleDownloadOverlay = () => {
    const a = document.createElement("a");
    a.href = overlayBase64;
    a.download = `damage_overlay_${analysisId || "latest"}.png`;
    a.click();
  };

  const handleDownloadJSON = async () => {
    if (!analysisId && !urlAnalysisId) return;
    const id = analysisId || urlAnalysisId;
    try {
      const res = await fetch(`http://localhost:5000/api/history/${id}`);
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data.result, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `damage_analysis_${id}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch { }
  };

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-7xl mx-auto px-6">
        <h1 className="text-4xl font-bold text-center mb-8">Damage Assessment Results</h1>

        <div className="grid lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex justify-between items-center">
                  <span>Damage Visualization</span>
                  <div className="flex items-center gap-4">
                    <span className="text-sm text-muted-foreground">
                      {showOverlay ? "Overlay View" : "Original View"}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowOverlay(!showOverlay)}
                      className="relative overflow-hidden"
                    >
                      <div
                        className={`absolute inset-0 w-1/2 bg-primary transition-transform duration-300 ${showOverlay ? "translate-x-full" : "translate-x-0"
                          }`}
                      />
                      <span className="relative z-10 px-6 py-2">Original</span>
                      <span className="relative z-10 px-6 py-2">Overlay</span>
                    </Button>
                    <Badge variant={totalDamaged > 20 ? "destructive" : "secondary"}>
                      {totalDamaged.toFixed(1)}% Damaged
                    </Badge>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="relative rounded-lg overflow-hidden bg-black">
                  <img
                    key={showOverlay ? "overlay" : "original"}
                    src={showOverlay ? overlayBase64 : postImagePreview}
                    alt={showOverlay ? "Damage overlay" : "Original post-disaster"}
                    className="w-full h-auto object-contain transition-opacity duration-500"
                    style={{ opacity: loading ? 0.5 : 1 }}
                  />
                  {showOverlay && (
                    <div className="absolute bottom-4 left-4 bg-black/70 text-white px-4 py-2 rounded-lg text-sm backdrop-blur">
                      White: No Damage │ Green: Minor │ Yellow: Major │ Magenta: Destroyed
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {preImagePreview && (
                <Card>
                  <CardHeader><CardTitle className="text-sm">Pre-Disaster Image</CardTitle></CardHeader>
                  <CardContent className="p-0">
                    <img src={preImagePreview} className="rounded-b-lg w-full" alt="Pre-disaster" />
                  </CardContent>
                </Card>
              )}
              {postImagePreview && (
                <Card>
                  <CardHeader><CardTitle className="text-sm">Post-Disaster Image</CardTitle></CardHeader>
                  <CardContent className="p-0">
                    <img src={postImagePreview} className="rounded-b-lg w-full" alt="Post-disaster" />
                  </CardContent>
                </Card>
              )}
            </div>
          </div>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Damage Statistics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                {Object.entries(percentages).map(([label, value]) => (
                  <div key={label}>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium">{label}</span>
                      <span className="text-sm">{value}%</span>
                    </div>
                    <Progress value={value as number} />
                  </div>
                ))}
                <div className="pt-4 border-t flex items-center gap-2 text-destructive">
                  <AlertTriangle className="w-5 h-5" />
                  <span className="font-semibold">{totalDamaged.toFixed(1)}% of buildings damaged</span>
                </div>
              </CardContent>
            </Card>

            <div className="space-y-3">
              <Button className="w-full" onClick={handleDownloadOverlay}>
                <Download className="mr-2 h-4 w-4" />
                Download Overlay Image
              </Button>
              <Button className="w-full" variant="outline" onClick={handleDownloadJSON}>
                <Download className="mr-2 h-4 w-4" />
                Download JSON Report
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;