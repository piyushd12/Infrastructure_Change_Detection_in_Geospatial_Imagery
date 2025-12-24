import { useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Upload as UploadIcon, Image, Loader2, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { useAnalysis } from "@/context/AnalysisContext";

const Upload = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const {
    setPreImagePreview,
    setPostImagePreview,
    setHeatmapBase64,
    setOverlayBase64,
    setLocalizationBase64,
    setStatistics,
    setAnalysisId,
    setIsAnalyzing,
    isAnalyzing,
    setError,
  } = useAnalysis();

  const preFileRef = useRef<File | null>(null);
  const postFileRef = useRef<File | null>(null);
  const preInputRef = useRef<HTMLInputElement>(null);
  const postInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (file: File, type: 'pre' | 'post') => {
    if (!file.type.startsWith('image/')) {
      toast({
        title: "Invalid file",
        description: "Please upload a valid image (JPG, PNG, etc.)",
        variant: "destructive",
      });
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      if (type === 'pre') {
        setPreImagePreview(result);
        preFileRef.current = file;
      } else {
        setPostImagePreview(result);
        postFileRef.current = file;
      }
    };
    reader.readAsDataURL(file);

    toast({
      title: "Image uploaded",
      description: `${type === 'pre' ? 'Pre' : 'Post'}-disaster image ready`,
    });
  };

  const handleAnalyze = async () => {
    if (!preFileRef.current || !postFileRef.current) {
      toast({
        title: "Missing images",
        description: "Please upload both pre and post-disaster images",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append("pre_image", preFileRef.current);
    formData.append("post_image", postFileRef.current);
    formData.append("threshold", "0.3"); // optional, adjust as needed

    try {
      toast({
        title: "Analysis started",
        description: "Running AI model on your images...",
      });

      const response = await fetch("http://localhost:5000/api/detect-damage", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Analysis failed");
      }

      // Store results
      setHeatmapBase64(data.visualizations.heatmap);
      setOverlayBase64(data.visualizations.overlay);
      setLocalizationBase64(data.visualizations.localization);
      setStatistics(data.statistics);
      setAnalysisId(data.analysis_id);

      toast({
        title: "Analysis Complete!",
        description: `Detected damage in ${data.statistics.damage_percentages["Minor Damage"] + data.statistics.damage_percentages["Major Damage"] + data.statistics.damage_percentages["Destroyed"]}% of buildings`,
      });

      navigate("/results");
    } catch (err: any) {
      console.error("Analysis error:", err);
      setError(err.message);
      toast({
        title: "Analysis Failed",
        description: err.message || "Could not connect to backend",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const UploadBox = ({ type, inputRef, onFileSelect }: {
    type: 'pre' | 'post';
    inputRef: React.RefObject<HTMLInputElement>;
    onFileSelect: (file: File) => void;
  }) => {
    const preview = type === 'pre' ?
      (preFileRef.current ? URL.createObjectURL(preFileRef.current) : null) :
      (postFileRef.current ? URL.createObjectURL(postFileRef.current) : null);

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Image className="w-5 h-5" />
            {type === 'pre' ? 'Pre-Disaster' : 'Post-Disaster'} Image
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div
            className="border-2 border-dashed rounded-xl p-8 text-center cursor-pointer hover:border-primary transition"
            onClick={() => inputRef.current?.click()}
          >
            {preview ? (
              <div className="space-y-4">
                <img src={preview} alt="preview" className="mx-auto max-h-64 rounded-lg" />
                <CheckCircle className="w-8 h-8 text-green-500 mx-auto" />
              </div>
            ) : (
              <>
                <UploadIcon className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="font-medium">Click to upload {type === 'pre' ? 'pre' : 'post'}-disaster image</p>
                <p className="text-sm text-muted-foreground">PNG, JPG, JPEG supported</p>
              </>
            )}
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && onFileSelect(e.target.files[0])}
            />
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="min-h-screen bg-background py-12">
      <div className="max-w-5xl mx-auto px-6">
        <h1 className="text-4xl font-bold text-center mb-4">Building Damage Assessment</h1>
        <p className="text-center text-muted-foreground mb-12">Upload satellite imagery to detect disaster damage using xView2 AI</p>

        <div className="grid md:grid-cols-2 gap-8 mb-12">
          <UploadBox type="pre" inputRef={preInputRef} onFileSelect={(f) => handleFileUpload(f, 'pre')} />
          <UploadBox type="post" inputRef={postInputRef} onFileSelect={(f) => handleFileUpload(f, 'post')} />
        </div>

        <div className="text-center">
          <Button
            size="lg"
            onClick={handleAnalyze}
            disabled={!preFileRef.current || !postFileRef.current || isAnalyzing}
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <UploadIcon className="mr-2 h-5 w-5" />
                Analyze Damage
              </>
            )}
          </Button>
        </div>

        {isAnalyzing && (
          <Card className="mt-8 max-w-2xl mx-auto">
            <CardContent className="pt-8 text-center">
              <Loader2 className="w-16 h-16 animate-spin text-primary mx-auto mb-4" />
              <p className="text-lg">Processing images with deep learning model...</p>
              <p className="text-sm text-muted-foreground mt-2">This may take 15–60 seconds</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default Upload;