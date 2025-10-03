import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Upload as UploadIcon, Image, Loader2, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";

const Upload = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [preDisasterImage, setPreDisasterImage] = useState<File | null>(null);
  const [postDisasterImage, setPostDisasterImage] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [preDisasterPreview, setPreDisasterPreview] = useState<string | null>(null);
  const [postDisasterPreview, setPostDisasterPreview] = useState<string | null>(null);

  const preDisasterRef = useRef<HTMLInputElement>(null);
  const postDisasterRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (file: File, type: 'pre' | 'post') => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        if (type === 'pre') {
          setPreDisasterImage(file);
          setPreDisasterPreview(result);
        } else {
          setPostDisasterImage(file);
          setPostDisasterPreview(result);
        }
      };
      reader.readAsDataURL(file);
      
      toast({
        title: "Image uploaded successfully",
        description: `${type === 'pre' ? 'Pre-disaster' : 'Post-disaster'} image has been uploaded.`,
      });
    } else {
      toast({
        title: "Invalid file type",
        description: "Please upload an image file (PNG, JPG, JPEG, etc.)",
        variant: "destructive",
      });
    }
  };

  const handleAnalyze = async () => {
    if (!preDisasterImage || !postDisasterImage) {
      toast({
        title: "Missing images",
        description: "Please upload both pre-disaster and post-disaster images.",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    
    // Simulate API call and processing
    toast({
      title: "Analysis started",
      description: "Processing your satellite images for change detection...",
    });

    // Simulate processing time
    setTimeout(() => {
      setIsAnalyzing(false);
      toast({
        title: "Analysis complete",
        description: "Your change detection analysis is ready!",
      });
      navigate('/results');
    }, 3000);
  };

  const UploadBox = ({ 
    type, 
    image, 
    preview, 
    inputRef, 
    onFileSelect 
  }: {
    type: 'pre' | 'post';
    image: File | null;
    preview: string | null;
    inputRef: React.RefObject<HTMLInputElement>;
    onFileSelect: (file: File) => void;
  }) => (
    <Card className="card-professional">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Image className="w-5 h-5 text-primary" />
          <span>{type === 'pre' ? 'Pre-Disaster Image' : 'Post-Disaster Image'}</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          className="card-upload cursor-pointer flex flex-col items-center justify-center min-h-[200px] space-y-4 relative"
          onClick={() => inputRef.current?.click()}
        >
          {preview ? (
            <div className="relative w-full h-48 rounded-lg overflow-hidden">
              <img 
                src={preview} 
                alt={`${type === 'pre' ? 'Pre-disaster' : 'Post-disaster'} preview`}
                className="w-full h-full object-cover"
              />
              <div className="absolute top-2 right-2">
                <CheckCircle className="w-6 h-6 text-success bg-white rounded-full" />
              </div>
            </div>
          ) : (
            <>
              <div className="w-16 h-16 bg-primary/10 rounded-2xl flex items-center justify-center">
                <UploadIcon className="w-8 h-8 text-primary" />
              </div>
              <div className="text-center">
                <p className="text-lg font-medium text-foreground mb-1">
                  Upload {type === 'pre' ? 'Pre-Disaster' : 'Post-Disaster'} Image
                </p>
                <p className="text-sm text-muted-foreground">
                  Click to browse or drag and drop
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Supports PNG, JPG, JPEG (Max 10MB)
                </p>
              </div>
            </>
          )}
          
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onFileSelect(file);
            }}
          />
        </div>
        
        {image && (
          <div className="mt-4 p-3 bg-accent rounded-lg">
            <p className="text-sm font-medium text-foreground">{image.name}</p>
            <p className="text-xs text-muted-foreground">
              {(image.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Upload Satellite Images
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload your pre-disaster and post-disaster satellite images to begin the AI-powered change detection analysis.
          </p>
        </div>

        <div className="max-w-4xl mx-auto space-y-8">
          {/* Upload Boxes */}
          <div className="grid md:grid-cols-2 gap-8">
            <UploadBox
              type="pre"
              image={preDisasterImage}
              preview={preDisasterPreview}
              inputRef={preDisasterRef}
              onFileSelect={(file) => handleFileUpload(file, 'pre')}
            />
            <UploadBox
              type="post"
              image={postDisasterImage}
              preview={postDisasterPreview}
              inputRef={postDisasterRef}
              onFileSelect={(file) => handleFileUpload(file, 'post')}
            />
          </div>

          {/* Analysis Button */}
          <div className="text-center">
            <Button
              className="btn-analyze"
              onClick={handleAnalyze}
              disabled={!preDisasterImage || !postDisasterImage || isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="mr-3 w-5 h-5 animate-spin" />
                  Analyzing Changes...
                </>
              ) : (
                <>
                  <UploadIcon className="mr-3 w-5 h-5" />
                  Analyze Changes
                </>
              )}
            </Button>
          </div>

          {/* Analysis Progress */}
          {isAnalyzing && (
            <Card className="card-professional">
              <CardContent className="p-6">
                <div className="flex items-center space-x-4">
                  <div className="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center pulse-analyze">
                    <Loader2 className="w-6 h-6 text-primary animate-spin" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground">Processing Images</h3>
                    <p className="text-sm text-muted-foreground">
                      Our AI is analyzing your satellite images for change detection...
                    </p>
                  </div>
                </div>
                <div className="mt-4 w-full bg-muted rounded-full h-2">
                  <div className="bg-gradient-to-r from-primary to-secondary h-2 rounded-full animate-pulse" style={{ width: '60%' }} />
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default Upload;