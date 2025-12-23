import { createContext, useContext, useState, ReactNode } from "react";

interface AnalysisState {
    preImagePreview: string | null;
    postImagePreview: string | null;
    heatmapBase64: string | null;
    overlayBase64: string | null;
    localizationBase64: string | null;
    statistics: any;
    analysisId: string | null;
    isAnalyzing: boolean;
    error: string | null;
    setPreImagePreview: (url: string | null) => void;
    setPostImagePreview: (url: string | null) => void;
    setHeatmapBase64: (b64: string | null) => void;
    setOverlayBase64: (b64: string | null) => void;
    setLocalizationBase64: (b64: string | null) => void;
    setStatistics: (stats: any) => void;
    setAnalysisId: (id: string | null) => void;
    setIsAnalyzing: (value: boolean) => void;
    setError: (msg: string | null) => void;
}

const AnalysisContext = createContext<AnalysisState | undefined>(undefined);

export const AnalysisProvider = ({ children }: { children: ReactNode }) => {
    const [preImagePreview, setPreImagePreview] = useState<string | null>(null);
    const [postImagePreview, setPostImagePreview] = useState<string | null>(null);
    const [heatmapBase64, setHeatmapBase64] = useState<string | null>(null);
    const [overlayBase64, setOverlayBase64] = useState<string | null>(null);
    const [localizationBase64, setLocalizationBase64] = useState<string | null>(null);
    const [statistics, setStatistics] = useState<any>(null);
    const [analysisId, setAnalysisId] = useState<string | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [error, setError] = useState<string | null>(null);

    return (
        <AnalysisContext.Provider
            value={{
                preImagePreview,
                postImagePreview,
                heatmapBase64,
                overlayBase64,
                localizationBase64,
                statistics,
                analysisId,
                isAnalyzing,
                error,
                setPreImagePreview,
                setPostImagePreview,
                setHeatmapBase64,
                setOverlayBase64,
                setLocalizationBase64,
                setStatistics,
                setAnalysisId,
                setIsAnalyzing,
                setError,
            }}
        >
            {children}
        </AnalysisContext.Provider>
    );
};

export const useAnalysis = () => {
    const context = useContext(AnalysisContext);
    if (!context) throw new Error("useAnalysis must be used within AnalysisProvider");
    return context;
};