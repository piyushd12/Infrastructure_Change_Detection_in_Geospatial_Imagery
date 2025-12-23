import { useEffect, useState } from "react";
import { CheckCircle, Calendar, RefreshCw, Image as ImageIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { useNavigate } from "react-router-dom";
import { Trash2 } from "lucide-react";

interface Analysis {
  id: string;
  timestamp: string;
  name: string;
  thumbnail: string;
  files: {
    pre_disaster: string;
    post_disaster: string;
    overlay: string;
    heatmap: string;
  };
  statistics: {
    damage_percentages: {
      "No Damage": number;
      "Minor Damage": number;
      "Major Damage": number;
      "Destroyed": number;
    };
  };
}

const Progress = () => {
  const { toast } = useToast();
  const navigate = useNavigate();
  const [history, setHistory] = useState<Analysis[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/api/history");
      if (!response.ok) throw new Error("Failed to fetch");
      const data = await response.json();
      if (data.success) {
        const formatted = data.results.map((item: any) => ({
          id: item.id,
          timestamp: item.timestamp,
          name: `Analysis ${new Date(item.timestamp).toLocaleDateString()}`,
          thumbnail: `http://localhost:5000${item.thumbnail}`,
          files: item.files || {},
          statistics: item.statistics || {},
        }));
        setHistory(formatted);
      }
    } catch (err) {
      toast({
        title: "Failed to load history",
        description: "Check if backend is running",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Are you sure you want to delete this analysis? This action cannot be undone.")) {
      return;
    }

    try {
      const res = await fetch(`http://localhost:5000/api/history/${id}`, {
        method: "DELETE",
      });
      const data = await res.json();

      if (data.success) {
        toast({
          title: "Deleted",
          description: "Analysis removed successfully",
        });
        fetchHistory(); // Refresh the list
      } else {
        throw new Error(data.error || "Delete failed");
      }
    } catch (err: any) {
      toast({
        title: "Delete failed",
        description: err.message || "Something went wrong",
        variant: "destructive",
      });
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const percentages = (stats: any) => stats.damage_percentages || {};

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4">Analysis History</h1>
          <p className="text-lg text-muted-foreground mb-4">
            View all your past damage assessments with full images and results
          </p>
          <Button onClick={fetchHistory} variant="outline">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>

        {loading ? (
          <div className="text-center py-12">
            <RefreshCw className="w-12 h-12 animate-spin text-primary mx-auto" />
          </div>
        ) : history.length === 0 ? (
          <Card className="p-12 text-center">
            <ImageIcon className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
            <p className="text-lg text-muted-foreground">No analyses yet. Start one from Upload!</p>
          </Card>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {history.map((analysis) => {
              const p = percentages(analysis.statistics);
              const totalDamaged = p["Minor Damage"] + p["Major Damage"] + p["Destroyed"];

              return (
                <Card
                  key={analysis.id}
                  className="overflow-hidden hover:shadow-lg transition group relative"
                >
                  {/* Delete Button - appears on hover */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation(); // Prevent navigating to results
                      handleDelete(analysis.id);
                    }}
                    className="absolute top-2 right-2 z-10 bg-red-600 text-white rounded-full p-2 opacity-0 group-hover:opacity-100 transition hover:bg-red-700 shadow-lg"
                    title="Delete analysis"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>

                  {/* Clickable area to view results */}
                  <div
                    className="aspect-video relative bg-black cursor-pointer"
                    onClick={() => navigate(`/results?id=${analysis.id}`)}
                  >
                    <img
                      src={analysis.thumbnail}
                      alt="Analysis thumbnail"
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition" />
                    <div className="absolute top-2 left-2">
                      <Badge className="bg-green-100 text-green-800">Completed</Badge>
                    </div>
                    <div className="absolute bottom-2 left-2 text-white bg-black/60 px-3 py-1 rounded text-sm">
                      {new Date(analysis.timestamp).toLocaleDateString()}
                    </div>
                  </div>

                  <CardHeader>
                    <CardTitle className="text-lg">{analysis.name}</CardTitle>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Calendar className="w-4 h-4" />
                      {new Date(analysis.timestamp).toLocaleTimeString()}
                    </div>
                  </CardHeader>

                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Damaged Buildings</span>
                        <span className={`font-semibold ${totalDamaged > 20 ? 'text-destructive' : 'text-orange-600'}`}>
                          {totalDamaged.toFixed(1)}%
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="text-center">
                          <div className="w-full h-1 bg-green-500 rounded mb-1" />
                          <p>Minor: {p["Minor Damage"] || 0}%</p>
                        </div>
                        <div className="text-center">
                          <div className="w-full h-1 bg-red-600 rounded mb-1" />
                          <p>Destroyed: {p["Destroyed"] || 0}%</p>
                        </div>
                      </div>
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