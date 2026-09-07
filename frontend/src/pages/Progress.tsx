import { CheckCircle, Calendar, Image as ImageIcon, Trash2, Clock } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { useNavigate } from "react-router-dom";
import { useAnalysis } from "@/context/AnalysisContext";

const Progress = () => {
  const { toast } = useToast();
  const navigate = useNavigate();
  const { sessionHistory, removeFromHistory } = useAnalysis();

  const handleDelete = async (id: string) => {
    if (!confirm("Remove this analysis from the current session? (Backend files will also be deleted.)")) return;
    try {
      const res = await fetch(`/api/history/${id}`, { method: "DELETE" });
      const data = await res.json();
      if (data.success) {
        removeFromHistory(id);
        toast({ title: "Deleted", description: "Analysis removed." });
      } else {
        throw new Error(data.error || "Delete failed");
      }
    } catch (err: any) {
      toast({ title: "Delete failed", description: err.message, variant: "destructive" });
    }
  };

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-7xl mx-auto px-6">

        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-3">Analysis History</h1>
          <p className="text-muted-foreground mb-2">
            Analyses completed during this session. History clears when you refresh the page.
          </p>
          {/* Session badge */}
          <div className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full mt-1"
            style={{
              background: "rgba(99,102,241,0.1)",
              border: "1px solid rgba(99,102,241,0.3)",
              color: "#818cf8",
            }}
          >
            <Clock className="w-3.5 h-3.5" />
            Session-only &mdash; refreshing the page will clear this list
          </div>
        </div>

        {sessionHistory.length === 0 ? (
          <Card className="p-12 text-center">
            <ImageIcon className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
            <p className="text-lg font-medium mb-1">No analyses this session</p>
            <p className="text-sm text-muted-foreground mb-6">
              Upload pre &amp; post-disaster images to run an analysis. Results will appear here until you refresh the page.
            </p>
            <Button onClick={() => navigate("/upload")}>Go to Upload</Button>
          </Card>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {sessionHistory.map((analysis) => {
              const p = analysis.statistics?.damage_percentages || {};
              const totalDamaged = (p["Minor Damage"] ?? 0) + (p["Major Damage"] ?? 0) + (p["Destroyed"] ?? 0);
              const score = analysis.statistics?.weighted_damage_score ?? 0;
              const scoreColor = score >= 60 ? "#ef4444" : score >= 30 ? "#f97316" : "#22c55e";

              return (
                <Card
                  key={analysis.id}
                  className="overflow-hidden hover:shadow-lg transition group relative"
                >
                  {/* Delete button */}
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDelete(analysis.id); }}
                    className="absolute top-2 right-2 z-10 bg-red-600 text-white rounded-full p-2 opacity-0 group-hover:opacity-100 transition hover:bg-red-700 shadow-lg"
                    title="Remove analysis"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>

                  {/* Thumbnail */}
                  <div
                    className="aspect-video relative bg-black cursor-pointer"
                    onClick={() => navigate(`/results?id=${analysis.id}`)}
                  >
                    <img
                      src={analysis.thumbnail}
                      alt="Analysis thumbnail"
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        // If thumbnail fails, fall back to overlay
                        (e.target as HTMLImageElement).src = analysis.overlayBase64 || "";
                      }}
                    />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition" />
                    <div className="absolute top-2 left-2">
                      <Badge className="bg-green-100 text-green-800 flex items-center gap-1">
                        <CheckCircle className="w-3 h-3" /> Completed
                      </Badge>
                    </div>
                    <div className="absolute bottom-2 left-2 text-white bg-black/60 px-3 py-1 rounded text-sm">
                      {new Date(analysis.timestamp).toLocaleDateString()}
                    </div>
                  </div>

                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">
                      Analysis — {new Date(analysis.timestamp).toLocaleDateString()}
                    </CardTitle>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Calendar className="w-4 h-4" />
                      {new Date(analysis.timestamp).toLocaleTimeString()}
                    </div>
                  </CardHeader>

                  <CardContent>
                    <div className="space-y-3">
                      {/* Damage % */}
                      <div className="flex justify-between text-sm items-center">
                        <span className="text-muted-foreground">Damaged pixels</span>
                        <span className="font-semibold" style={{ color: scoreColor }}>
                          {totalDamaged.toFixed(1)}%
                        </span>
                      </div>

                      {/* Score bar */}
                      <div>
                        <div className="flex justify-between text-xs text-muted-foreground mb-1">
                          <span>Damage score</span>
                          <span style={{ color: scoreColor }}>{score}/100</span>
                        </div>
                        <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{ width: `${score}%`, backgroundColor: scoreColor }}
                          />
                        </div>
                      </div>

                      {/* Minor / Destroyed quick stats */}
                      <div className="grid grid-cols-2 gap-2 text-xs pt-1">
                        <div className="text-center">
                          <div className="w-full h-1 bg-blue-400 rounded mb-1" />
                          <p>Minor: {p["Minor Damage"] ?? 0}%</p>
                        </div>
                        <div className="text-center">
                          <div className="w-full h-1 bg-red-600 rounded mb-1" />
                          <p>Destroyed: {p["Destroyed"] ?? 0}%</p>
                        </div>
                      </div>

                      {/* View button */}
                      <Button
                        size="sm"
                        variant="outline"
                        className="w-full mt-1"
                        onClick={() => navigate(`/results?id=${analysis.id}`)}
                      >
                        View Full Results
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default Progress;