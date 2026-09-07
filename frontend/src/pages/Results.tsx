// import { useEffect, useState } from "react";
// import { Download, AlertTriangle, Loader2, Building2, Target, MapPin, BarChart3 } from "lucide-react";
// import { Button } from "@/components/ui/button";
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
// import { Badge } from "@/components/ui/badge";
// import { Progress } from "@/components/ui/progress";
// import { useAnalysis } from "@/context/AnalysisContext";
// import { useNavigate, useSearchParams } from "react-router-dom";
// import { useToast } from "@/hooks/use-toast";

// const DAMAGE_COLORS: Record<string, string> = {
//   "No Damage":    "#ffffff",
//   "Minor Damage": "#69bfff",
//   "Major Damage": "#ff8c00",
//   "Destroyed":    "#00008b",
// };

// const DAMAGE_BG: Record<string, string> = {
//   "No Damage":    "bg-white border border-gray-200 text-gray-800",
//   "Minor Damage": "bg-blue-100 text-blue-800",
//   "Major Damage": "bg-orange-100 text-orange-800",
//   "Destroyed":    "bg-blue-950 text-white",
// };

// const Results = () => {
//   const navigate      = useNavigate();
//   const [searchParams] = useSearchParams();
//   const { toast }     = useToast();
//   const {
//     heatmapBase64, overlayBase64,
//     preImagePreview, postImagePreview,
//     statistics, analysisId,
//     setHeatmapBase64, setOverlayBase64,
//     setPreImagePreview, setPostImagePreview,
//     setStatistics, setAnalysisId,
//   } = useAnalysis();

//   const [showOverlay, setShowOverlay] = useState(true);
//   const [loading, setLoading]         = useState(true);
//   const urlAnalysisId = searchParams.get("id");

//   useEffect(() => {
//     const loadAnalysis = async (id: string) => {
//       setLoading(true);
//       try {
//         const res = await fetch(`/api/history/${id}`);
//         if (!res.ok) throw new Error("Failed to load analysis");
//         const data = await res.json();
//         if (!data.success) throw new Error("Analysis not found");

//         const result  = data.result;
//         const baseUrl = "/api/results";

//         if (analysisId !== id)                                            setAnalysisId(id);
//         if (preImagePreview !== `${baseUrl}/${id}/pre_disaster.jpg`)      setPreImagePreview(`${baseUrl}/${id}/pre_disaster.jpg`);
//         if (postImagePreview !== `${baseUrl}/${id}/post_disaster.jpg`)    setPostImagePreview(`${baseUrl}/${id}/post_disaster.jpg`);
//         if (overlayBase64   !== `${baseUrl}/${id}/overlay.png`)           setOverlayBase64(`${baseUrl}/${id}/overlay.png`);
//         if (heatmapBase64   !== `${baseUrl}/${id}/heatmap.png`)           setHeatmapBase64(`${baseUrl}/${id}/heatmap.png`);
//         if (JSON.stringify(statistics) !== JSON.stringify(result.statistics)) setStatistics(result.statistics);

//         toast({ title: "Analysis loaded", description: `From ${new Date(result.timestamp).toLocaleDateString()}` });
//       } catch (err: any) {
//         toast({ title: "Failed to load", description: err.message, variant: "destructive" });
//         navigate("/upload");
//       } finally {
//         setLoading(false);
//       }
//     };

//     if (urlAnalysisId)                          { loadAnalysis(urlAnalysisId); return; }
//     if (overlayBase64 && statistics && analysisId) { setLoading(false); return; }
//     navigate("/upload");
//   }, [urlAnalysisId]);

//   if (loading) {
//     return (
//       <div className="min-h-screen flex items-center justify-center">
//         <div className="text-center">
//           <Loader2 className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
//           <p className="text-lg">Loading analysis results...</p>
//         </div>
//       </div>
//     );
//   }

//   if (!statistics || !overlayBase64) { navigate("/upload"); return null; }

//   const percentages      = statistics.damage_percentages     || {};
//   const buildingCounts   = statistics.building_counts        || {};
//   const buildingCountPct = statistics.building_count_percentages || {};
//   const totalBuildings   = statistics.total_buildings        ?? 0;
//   const weightedScore    = statistics.weighted_damage_score  ?? 0;
//   const damageDensity    = statistics.damage_density         ?? 0;
//   const hotspot          = statistics.hotspot                ?? null;
//   const totalDamaged     = (percentages["Minor Damage"] ?? 0)
//                          + (percentages["Major Damage"] ?? 0)
//                          + (percentages["Destroyed"]    ?? 0);
//   const totalBuildingsDamaged = (buildingCounts["Minor Damage"] ?? 0)
//                               + (buildingCounts["Major Damage"] ?? 0)
//                               + (buildingCounts["Destroyed"]    ?? 0);

//   // Weighted score colour
//   const scoreColor = weightedScore >= 60 ? "text-red-500"
//                    : weightedScore >= 30 ? "text-orange-500"
//                    : "text-green-500";

//   const handleDownloadOverlay = () => {
//     const a = document.createElement("a");
//     a.href = overlayBase64;
//     a.download = `damage_overlay_${analysisId || "latest"}.png`;
//     a.click();
//   };

//   const handleDownloadJSON = async () => {
//     const id = analysisId || urlAnalysisId;
//     if (!id) return;
//     try {
//       const res  = await fetch(`/api/history/${id}`);
//       const data = await res.json();
//       const blob = new Blob([JSON.stringify(data.result, null, 2)], { type: "application/json" });
//       const url  = URL.createObjectURL(blob);
//       const a    = document.createElement("a");
//       a.href = url; a.download = `damage_analysis_${id}.json`; a.click();
//       URL.revokeObjectURL(url);
//     } catch {}
//   };

//   return (
//     <div className="min-h-screen bg-background py-8">
//       <div className="max-w-7xl mx-auto px-6">
//         <h1 className="text-4xl font-bold text-center mb-8">Damage Assessment Results</h1>

//         <div className="grid lg:grid-cols-3 gap-8">

//           {/* ── Left: visualisation + pre/post images ── */}
//           <div className="lg:col-span-2 space-y-6">

//             {/* Damage visualisation */}
//             <Card>
//               <CardHeader>
//                 <CardTitle className="flex justify-between items-center flex-wrap gap-2">
//                   <span>Damage Visualization</span>
//                   <div className="flex items-center gap-3 flex-wrap">
//                     <div className="flex rounded-md overflow-hidden border text-sm">
//                       <button
//                         onClick={() => setShowOverlay(false)}
//                         className={`px-3 py-1.5 transition-colors ${!showOverlay ? "bg-primary text-primary-foreground" : "bg-background text-muted-foreground hover:bg-muted"}`}
//                       >Original</button>
//                       <button
//                         onClick={() => setShowOverlay(true)}
//                         className={`px-3 py-1.5 transition-colors ${showOverlay  ? "bg-primary text-primary-foreground" : "bg-background text-muted-foreground hover:bg-muted"}`}
//                       >Overlay</button>
//                     </div>
//                     <Badge variant={totalDamaged > 20 ? "destructive" : "secondary"}>
//                       {totalDamaged.toFixed(1)}% Damaged
//                     </Badge>
//                   </div>
//                 </CardTitle>
//               </CardHeader>
//               <CardContent>
//                 <div className="relative rounded-lg overflow-hidden bg-black">
//                   <img
//                     key={showOverlay ? "overlay" : "original"}
//                     src={showOverlay ? overlayBase64 : postImagePreview}
//                     alt={showOverlay ? "Damage overlay" : "Original post-disaster"}
//                     className="w-full h-auto object-contain"
//                   />
//                   {/* Hotspot pin */}
//                   {showOverlay && hotspot && (
//                     <div
//                       className="absolute w-4 h-4 -translate-x-1/2 -translate-y-1/2 pointer-events-none"
//                       style={{ left: `${hotspot.x_norm * 100}%`, top: `${hotspot.y_norm * 100}%` }}
//                     >
//                       <div className="w-4 h-4 rounded-full bg-yellow-400 border-2 border-white shadow-lg animate-pulse" />
//                     </div>
//                   )}
//                   {showOverlay && (
//                     <div className="absolute bottom-3 left-3 bg-black/75 text-white px-3 py-2 rounded-lg text-xs backdrop-blur flex flex-wrap gap-x-3 gap-y-1">
//                       <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm bg-white border border-gray-400" />No damage</span>
//                       <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm" style={{backgroundColor:"#69bfff"}} />Minor</span>
//                       <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm" style={{backgroundColor:"#ff8c00"}} />Major</span>
//                       <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm" style={{backgroundColor:"#00008b"}} />Destroyed</span>
//                       <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm bg-black border border-gray-600" />Background</span>
//                       {hotspot && <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-full bg-yellow-400" />Damage hotspot</span>}
//                     </div>
//                   )}
//                 </div>
//               </CardContent>
//             </Card>

//             {/* Pre / post images */}
//             <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
//               {preImagePreview && (
//                 <Card>
//                   <CardHeader><CardTitle className="text-sm">Pre-Disaster Image</CardTitle></CardHeader>
//                   <CardContent className="p-0">
//                     <img src={preImagePreview} className="rounded-b-lg w-full" alt="Pre-disaster" />
//                   </CardContent>
//                 </Card>
//               )}
//               {postImagePreview && (
//                 <Card>
//                   <CardHeader><CardTitle className="text-sm">Post-Disaster Image</CardTitle></CardHeader>
//                   <CardContent className="p-0">
//                     <img src={postImagePreview} className="rounded-b-lg w-full" alt="Post-disaster" />
//                   </CardContent>
//                 </Card>
//               )}
//             </div>
//           </div>

//           {/* ── Right: statistics panel ── */}
//           <div className="space-y-5">

//             {/* KPI row */}
//             <div className="grid grid-cols-2 gap-3">
//               <Card className="p-4 flex flex-col gap-1">
//                 <div className="flex items-center gap-2 text-muted-foreground text-xs"><Building2 className="w-3.5 h-3.5" />Total buildings</div>
//                 <p className="text-3xl font-semibold">{totalBuildings}</p>
//                 <p className="text-xs text-muted-foreground">{totalBuildingsDamaged} damaged</p>
//               </Card>
//               <Card className="p-4 flex flex-col gap-1">
//                 <div className="flex items-center gap-2 text-muted-foreground text-xs"><BarChart3 className="w-3.5 h-3.5" />Damage score</div>
//                 <p className={`text-3xl font-semibold ${scoreColor}`}>{weightedScore}<span className="text-base font-normal text-muted-foreground">/100</span></p>
//                 <p className="text-xs text-muted-foreground">Weighted severity</p>
//               </Card>
//               <Card className="p-4 flex flex-col gap-1">
//                 <div className="flex items-center gap-2 text-muted-foreground text-xs"><Target className="w-3.5 h-3.5" />Damage density</div>
//                 <p className="text-3xl font-semibold">{damageDensity}<span className="text-base font-normal text-muted-foreground">%</span></p>
//                 <p className="text-xs text-muted-foreground">Of total image area</p>
//               </Card>
//               <Card className="p-4 flex flex-col gap-1">
//                 <div className="flex items-center gap-2 text-muted-foreground text-xs"><MapPin className="w-3.5 h-3.5" />Hotspot</div>
//                 {hotspot ? (
//                   <>
//                     <p className="text-sm font-semibold leading-tight">
//                       x: {hotspot.x_norm * 100 | 0}%<br />y: {hotspot.y_norm * 100 | 0}%
//                     </p>
//                     <p className="text-xs text-muted-foreground">Based on {hotspot.based_on.toLowerCase()}</p>
//                   </>
//                 ) : (
//                   <p className="text-sm text-muted-foreground">None detected</p>
//                 )}
//               </Card>
//             </div>

//             {/* Pixel % breakdown */}
//             <Card>
//               <CardHeader className="pb-2">
//                 <CardTitle className="text-sm">Pixel-level breakdown</CardTitle>
//               </CardHeader>
//               <CardContent className="space-y-4">
//                 {Object.entries(percentages).map(([label, value]) => (
//                   <div key={label}>
//                     <div className="flex justify-between mb-1 text-sm">
//                       <span className="flex items-center gap-2">
//                         <span
//                           className="inline-block w-3 h-3 rounded-sm border border-black/10"
//                           style={{ backgroundColor: DAMAGE_COLORS[label] ?? "#ccc" }}
//                         />
//                         {label}
//                       </span>
//                       <span className="text-muted-foreground">{value as number}%</span>
//                     </div>
//                     <Progress value={value as number} />
//                   </div>
//                 ))}
//                 <div className="pt-3 border-t flex items-center gap-2 text-destructive text-sm">
//                   <AlertTriangle className="w-4 h-4 flex-shrink-0" />
//                   <span className="font-medium">{totalDamaged.toFixed(1)}% of building pixels damaged</span>
//                 </div>
//               </CardContent>
//             </Card>

//             {/* Building count breakdown */}
//             {totalBuildings > 0 && (
//               <Card>
//                 <CardHeader className="pb-2">
//                   <CardTitle className="text-sm">Building count breakdown</CardTitle>
//                 </CardHeader>
//                 <CardContent className="space-y-3">
//                   {Object.entries(buildingCounts).map(([label, count]) => (
//                     <div key={label} className="flex items-center justify-between text-sm">
//                       <span className="flex items-center gap-2">
//                         <span
//                           className="inline-block w-3 h-3 rounded-sm border border-black/10"
//                           style={{ backgroundColor: DAMAGE_COLORS[label] ?? "#ccc" }}
//                         />
//                         {label}
//                       </span>
//                       <div className="flex items-center gap-2">
//                         <span className="text-muted-foreground text-xs">{buildingCountPct[label] ?? 0}%</span>
//                         <span className={`font-semibold px-2 py-0.5 rounded text-xs ${DAMAGE_BG[label] ?? ""}`}>
//                           {count as number} bldgs
//                         </span>
//                       </div>
//                     </div>
//                   ))}
//                   <div className="pt-2 border-t text-xs text-muted-foreground">
//                     {totalBuildingsDamaged} of {totalBuildings} buildings affected
//                   </div>
//                 </CardContent>
//               </Card>
//             )}

//             {/* Downloads */}
//             <div className="space-y-2">
//               <Button className="w-full" onClick={handleDownloadOverlay}>
//                 <Download className="mr-2 h-4 w-4" />Download Overlay Image
//               </Button>
//               <Button className="w-full" variant="outline" onClick={handleDownloadJSON}>
//                 <Download className="mr-2 h-4 w-4" />Download JSON Report
//               </Button>
//             </div>

//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Results;




import { useEffect, useState, useRef } from "react";
import { Download, AlertTriangle, Loader2, Building2, Target, MapPin, BarChart3, Link, Crop, X, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useAnalysis } from "@/context/AnalysisContext";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";

const DAMAGE_COLORS: Record<string, string> = {
  "No Damage":    "#ffffff",
  "Minor Damage": "#69bfff",
  "Major Damage": "#ff8c00",
  "Destroyed":    "#00008b",
};

const DAMAGE_BG: Record<string, string> = {
  "No Damage":    "bg-white border border-gray-200 text-gray-800",
  "Minor Damage": "bg-blue-100 text-blue-800",
  "Major Damage": "bg-orange-100 text-orange-800",
  "Destroyed":    "bg-blue-950 text-white",
};

interface BBox { x: number; y: number; w: number; h: number; }

const Results = () => {
  const navigate       = useNavigate();
  const [searchParams] = useSearchParams();
  const { toast }      = useToast();
  const {
    heatmapBase64, overlayBase64,
    preImagePreview, postImagePreview,
    statistics, analysisId,
    sessionHistory,
    setHeatmapBase64, setOverlayBase64,
    setPreImagePreview, setPostImagePreview,
    setStatistics, setAnalysisId,
  } = useAnalysis();

  const [showOverlay, setShowOverlay]         = useState(true);
  const [loading, setLoading]                 = useState(true);
  const [generatingPdf, setGeneratingPdf]     = useState(false);
  const [cropMode, setCropMode]               = useState(false);
  const [drawing, setDrawing]                 = useState(false);
  const [bbox, setBbox]                       = useState<BBox | null>(null);
  const [dragStart, setDragStart]             = useState<{x:number;y:number} | null>(null);
  const [regionStats, setRegionStats]         = useState<any>(null);
  const [analyzingRegion, setAnalyzingRegion] = useState(false);
  const imgContainerRef                       = useRef<HTMLDivElement>(null);
  const urlAnalysisId = searchParams.get("id");

  // ── Load analysis ──────────────────────────────────────────────────────────
  useEffect(() => {
    const loadFromBackend = async (id: string) => {
      setLoading(true);
      try {
        const res = await fetch(`/api/history/${id}`);
        if (!res.ok) throw new Error("Failed to load analysis");
        const data = await res.json();
        if (!data.success) throw new Error("Analysis not found");
        const result  = data.result;
        const baseUrl = "/api/results";
        if (analysisId !== id)                                           setAnalysisId(id);
        if (preImagePreview !== `${baseUrl}/${id}/pre_disaster.jpg`)     setPreImagePreview(`${baseUrl}/${id}/pre_disaster.jpg`);
        if (postImagePreview !== `${baseUrl}/${id}/post_disaster.jpg`)   setPostImagePreview(`${baseUrl}/${id}/post_disaster.jpg`);
        if (overlayBase64   !== `${baseUrl}/${id}/overlay.png`)          setOverlayBase64(`${baseUrl}/${id}/overlay.png`);
        if (heatmapBase64   !== `${baseUrl}/${id}/heatmap.png`)          setHeatmapBase64(`${baseUrl}/${id}/heatmap.png`);
        if (JSON.stringify(statistics) !== JSON.stringify(result.statistics)) setStatistics(result.statistics);
        toast({ title: "Analysis loaded", description: `From ${new Date(result.timestamp).toLocaleDateString()}` });
      } catch (err: any) {
        toast({ title: "Failed to load", description: err.message, variant: "destructive" });
        navigate("/upload");
      } finally { setLoading(false); }
    };

    if (urlAnalysisId) {
      // Check session history first (avoids backend round-trip when clicking from Progress)
      const sessionEntry = sessionHistory.find(e => e.id === urlAnalysisId);
      if (sessionEntry) {
        if (analysisId !== sessionEntry.id)                       setAnalysisId(sessionEntry.id);
        if (preImagePreview !== sessionEntry.preImagePreview)     setPreImagePreview(sessionEntry.preImagePreview);
        if (postImagePreview !== sessionEntry.postImagePreview)   setPostImagePreview(sessionEntry.postImagePreview);
        if (overlayBase64   !== sessionEntry.overlayBase64)       setOverlayBase64(sessionEntry.overlayBase64);
        if (heatmapBase64   !== sessionEntry.heatmapBase64)       setHeatmapBase64(sessionEntry.heatmapBase64);
        if (JSON.stringify(statistics) !== JSON.stringify(sessionEntry.statistics)) setStatistics(sessionEntry.statistics);
        setLoading(false);
        return;
      }
      // Not in session — fall back to backend (shareable link / post-refresh)
      loadFromBackend(urlAnalysisId);
      return;
    }

    if (overlayBase64 && statistics && analysisId) { setLoading(false); return; }
    navigate("/upload");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlAnalysisId]);

  // ── Shareable link ─────────────────────────────────────────────────────────
  const handleCopyLink = () => {
    const id = analysisId || urlAnalysisId;
    if (!id) return;
    navigator.clipboard.writeText(`${window.location.origin}/results?id=${id}`).then(() =>
      toast({ title: "Link copied!", description: "Share this link to open this analysis directly." })
    );
  };

  // ── PDF export ─────────────────────────────────────────────────────────────
  // const handleDownloadPDF = async () => {
  //   if (!statistics || !overlayBase64) return;
  //   setGeneratingPdf(true);
  //   try {
  //     // load jsPDF from CDN
  //     await new Promise<void>((resolve, reject) => {
  //       if ((window as any).jspdf) { resolve(); return; }
  //       const s  = document.createElement("script");
  //       s.src    = "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";
  //       s.onload = () => resolve();
  //       s.onerror = reject;
  //       document.head.appendChild(s);
  //     });
  //     const { jsPDF } = (window as any).jspdf;
  //     const doc       = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  //     const pageW     = 210;
  //     const margin    = 15;
  //     const contentW  = pageW - margin * 2;
  //     let   y         = margin;

  //     // header bar
  //     doc.setFillColor(20, 20, 20);
  //     doc.rect(0, 0, pageW, 30, "F");
  //     doc.setTextColor(255, 255, 255);
  //     doc.setFontSize(18); doc.setFont("helvetica", "bold");
  //     doc.text("Building Damage Assessment Report", margin, 18);
  //     doc.setFontSize(9);  doc.setFont("helvetica", "normal");
  //     doc.text(`GeoSense AI  |  Generated: ${new Date().toLocaleString()}`, margin, 25);
  //     y = 40;

  //     // analysis id
  //     doc.setTextColor(30, 30, 30); doc.setFontSize(9); doc.setFont("helvetica", "normal");
  //     const id = analysisId || urlAnalysisId || "N/A";
  //     doc.text(`Analysis ID: ${id}`, margin, y); y += 7;

  //     // summary paragraph
  //     const pct      = statistics.damage_percentages || {};
  //     const bc       = statistics.building_counts    || {};
  //     const totalDmg = ((pct["Minor Damage"] ?? 0) + (pct["Major Damage"] ?? 0) + (pct["Destroyed"] ?? 0)).toFixed(1);
  //     const score    = statistics.weighted_damage_score ?? 0;
  //     const density  = statistics.damage_density ?? 0;
  //     const totalB   = statistics.total_buildings ?? 0;
  //     const destroyed = bc["Destroyed"] ?? 0;
  //     const severity  = score >= 60 ? "SEVERE" : score >= 30 ? "MODERATE" : "LOW";

  //     doc.setFontSize(11); doc.setFont("helvetica", "bold");
  //     doc.text("Executive Summary", margin, y); y += 6;
  //     doc.setFont("helvetica", "normal"); doc.setFontSize(9.5);
  //     const summary =
  //       `The AI damage assessment identified ${totalB} buildings in the analysed area. ` +
  //       `${totalDmg}% of building pixels show signs of damage, with a weighted damage score of ${score}/100 ` +
  //       `indicating ${severity} overall impact. Damage density across the image is ${density}%, with ` +
  //       `${destroyed} buildings classified as destroyed. Immediate emergency response is ` +
  //       `${score >= 60 ? "strongly recommended" : score >= 30 ? "recommended" : "not urgently required"}.`;
  //     const lines = doc.splitTextToSize(summary, contentW);
  //     doc.text(lines, margin, y); y += lines.length * 5 + 5;

  //     // KPI boxes
  //     doc.setFontSize(11); doc.setFont("helvetica", "bold");
  //     doc.text("Key Metrics", margin, y); y += 6;
  //     const kpis = [
  //       { label: "Total Buildings", value: String(totalB) },
  //       { label: "Damage Score",    value: `${score} / 100` },
  //       { label: "Damage Density",  value: `${density}%` },
  //       { label: "Pixels Damaged",  value: `${totalDmg}%` },
  //     ];
  //     const kpiW = contentW / 4;
  //     kpis.forEach(({ label, value }, i) => {
  //       const kx = margin + i * kpiW;
  //       doc.setFillColor(245, 245, 245);
  //       doc.roundedRect(kx, y, kpiW - 3, 18, 2, 2, "F");
  //       doc.setTextColor(100, 100, 100); doc.setFontSize(7.5); doc.setFont("helvetica", "normal");
  //       doc.text(label, kx + 3, y + 6);
  //       doc.setTextColor(20, 20, 20);   doc.setFontSize(11);  doc.setFont("helvetica", "bold");
  //       doc.text(value,  kx + 3, y + 14);
  //     });
  //     y += 24;

  //     // damage table
  //     doc.setFontSize(11); doc.setFont("helvetica", "bold"); doc.setTextColor(30, 30, 30);
  //     doc.text("Damage Breakdown", margin, y); y += 6;
  //     const colorMap: Record<string, [number,number,number]> = {
  //       "No Damage":    [240,240,240], "Minor Damage": [105,191,255],
  //       "Major Damage": [255,140,  0], "Destroyed":    [  0,  0,139],
  //     };
  //     const headers   = ["Damage Class", "Pixel Count", "Pixel %", "Buildings", "Building %"];
  //     const colWidths = [45, 30, 25, 30, 30];
  //     let cx = margin;
  //     doc.setFillColor(30,30,30); doc.rect(margin, y, contentW, 7, "F");
  //     doc.setTextColor(255,255,255); doc.setFontSize(8); doc.setFont("helvetica","bold");
  //     headers.forEach((h, i) => { doc.text(h, cx + 2, y + 5); cx += colWidths[i]; });
  //     y += 7;
  //     Object.entries(pct).forEach(([label, pixPct], idx) => {
  //       doc.setFillColor(...(idx % 2 === 0 ? [255,255,255] : [248,248,248]) as [number,number,number]);
  //       doc.rect(margin, y, contentW, 7, "F");
  //       const [cr,cg,cb] = colorMap[label] ?? [200,200,200];
  //       doc.setFillColor(cr,cg,cb); doc.rect(margin+1, y+1.5, 5, 4, "F");
  //       cx = margin;
  //       doc.setTextColor(30,30,30); doc.setFontSize(8); doc.setFont("helvetica","normal");
  //       doc.text(label,                                            cx+8, y+5); cx+=colWidths[0];
  //       doc.text(String(statistics.damage_breakdown?.[label]??0), cx+2, y+5); cx+=colWidths[1];
  //       doc.text(`${pixPct}%`,                                    cx+2, y+5); cx+=colWidths[2];
  //       doc.text(String(bc[label]??0),                            cx+2, y+5); cx+=colWidths[3];
  //       doc.text(`${statistics.building_count_percentages?.[label]??0}%`, cx+2, y+5);
  //       y += 7;
  //     });
  //     y += 6;

  //     // overlay image
  //     doc.setFontSize(11); doc.setFont("helvetica","bold"); doc.setTextColor(30,30,30);
  //     doc.text("Damage Overlay Map", margin, y); y += 4;
  //     let imgData = overlayBase64;
  //     if (overlayBase64.startsWith("http")) {
  //       const resp = await fetch(overlayBase64);
  //       const blob = await resp.blob();
  //       imgData = await new Promise<string>((res) => {
  //         const r = new FileReader(); r.onload = () => res(r.result as string); r.readAsDataURL(blob);
  //       });
  //     }
  //     const imgH = 85;
  //     doc.addImage(imgData, "PNG", margin, y, contentW, imgH); y += imgH + 4;

  //     // legend
  //     const legendItems: { label: string; color: [number,number,number] }[] = [
  //       { label: "No Damage",    color: [240,240,240] },
  //       { label: "Minor Damage", color: [105,191,255] },
  //       { label: "Major Damage", color: [255,140,  0] },
  //       { label: "Destroyed",    color: [  0,  0,139] },
  //     ];
  //     let lx = margin;
  //     doc.setFontSize(7.5); doc.setFont("helvetica","normal");
  //     legendItems.forEach(({ label, color }) => {
  //       doc.setFillColor(...color); doc.rect(lx, y, 4, 4, "F");
  //       doc.setTextColor(50,50,50); doc.text(label, lx+5, y+3.5); lx += 42;
  //     });
  //     y += 8;

  //     // hotspot
  //     if (statistics.hotspot) {
  //       const hs = statistics.hotspot;
  //       doc.setFontSize(9); doc.setFont("helvetica","normal"); doc.setTextColor(80,80,80);
  //       doc.text(`Hotspot: x=${(hs.x_norm*100).toFixed(1)}%, y=${(hs.y_norm*100).toFixed(1)}% (based on ${hs.based_on})`, margin, y);
  //       y += 6;
  //     }

  //     // footer
  //     doc.setFontSize(7.5); doc.setTextColor(150,150,150);
  //     doc.text("Generated by GeoSense AI  |  For emergency response use only", margin, 290);

  //     doc.save(`damage_report_${id}.pdf`);
  //     toast({ title: "PDF exported!", description: `damage_report_${id}.pdf` });
  //   } catch (err: any) {z
  //     console.error(err);
  //     toast({ title: "PDF generation failed", description: err.message, variant: "destructive" });
  //   } finally { setGeneratingPdf(false); }
  // };

  // Replace the entire handleDownloadPDF function in Results.tsx with this.
// Uses a separate page for the overlay image so it never gets compressed/squished.

const handleDownloadPDF = async () => {
  if (!statistics || !overlayBase64) return;
  setGeneratingPdf(true);
  try {
    // Load jsPDF from CDN on first use
    await new Promise<void>((resolve, reject) => {
      if ((window as any).jspdf) { resolve(); return; }
      const s   = document.createElement('script');
      s.src     = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
      s.onload  = () => resolve();
      s.onerror = reject;
      document.head.appendChild(s);
    });
    const { jsPDF } = (window as any).jspdf;

    const doc      = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const pageW    = 210;
    const pageH    = 297;
    const margin   = 16;
    const cW       = pageW - margin * 2;   // content width = 178 mm
    let   y        = margin;

    // ── Helpers ────────────────────────────────────────────────────────────
    const heading = (text: string, size = 12) => {
      doc.setFontSize(size);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(20, 20, 20);
      doc.text(text, margin, y);
      y += size * 0.45 + 3;
    };

    const body = (text: string, size = 9.5, color = [50, 50, 50] as [number,number,number]) => {
      doc.setFontSize(size);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(...color);
      const lines = doc.splitTextToSize(text, cW);
      doc.text(lines, margin, y);
      y += lines.length * (size * 0.42) + 2;
    };

    const rule = (gap = 4) => {
      doc.setDrawColor(220, 220, 220);
      doc.line(margin, y, pageW - margin, y);
      y += gap;
    };

    // ════════════════════════════════════════════════════════════════════════
    // PAGE 1 — Text report
    // ════════════════════════════════════════════════════════════════════════

    // ── Header band ────────────────────────────────────────────────────────
    doc.setFillColor(15, 15, 15);
    doc.rect(0, 0, pageW, 32, 'F');

    // accent stripe
    doc.setFillColor(220, 50, 50);
    doc.rect(0, 32, pageW, 2, 'F');

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(20); doc.setFont('helvetica', 'bold');
    doc.text('Building Damage Assessment Report', margin, 14);

    doc.setFontSize(9); doc.setFont('helvetica', 'normal');
    doc.setTextColor(180, 180, 180);
    doc.text('GeoSense AI', margin, 22);
    doc.text(`Generated: ${new Date().toLocaleString()}`, margin, 28);

    y = 42;

    // ── Meta row ───────────────────────────────────────────────────────────
    const id = analysisId || urlAnalysisId || 'N/A';
    doc.setFontSize(8.5); doc.setFont('helvetica', 'normal');
    doc.setTextColor(120, 120, 120);
    doc.text(`Analysis ID: ${id}`, margin, y);
    y += 8;

    rule(5);

    // ── Executive summary ──────────────────────────────────────────────────
    const pct       = statistics.damage_percentages        || {};
    const bc        = statistics.building_counts           || {};
    const bldPct    = statistics.building_count_percentages || {};
    const bdBreak   = statistics.damage_breakdown          || {};
    const totalDmg  = ((pct['Minor Damage'] ?? 0) + (pct['Major Damage'] ?? 0) + (pct['Destroyed'] ?? 0)).toFixed(1);
    const score     = statistics.weighted_damage_score ?? 0;
    const density   = statistics.damage_density        ?? 0;
    const totalB    = statistics.total_buildings       ?? 0;
    const destroyed = bc['Destroyed']                  ?? 0;
    const severity  = score >= 60 ? 'SEVERE' : score >= 30 ? 'MODERATE' : 'LOW';
    const urgency   = score >= 60 ? 'strongly recommended' : score >= 30 ? 'recommended' : 'not urgently required';

    heading('Executive Summary', 11);
    body(
      `The AI damage assessment identified ${totalB} buildings in the analysed area. ` +
      `${totalDmg}% of building pixels show signs of damage, with a weighted damage score of ` +
      `${score}/100 indicating ${severity} overall impact. Damage density across the image ` +
      `is ${density}%, with ${destroyed} buildings classified as destroyed. ` +
      `Immediate emergency response is ${urgency}.`
    );
    y += 4;
    rule(6);

    // ── KPI boxes ──────────────────────────────────────────────────────────
    heading('Key Metrics', 11);
    const kpis = [
      { label: 'Total Buildings', value: String(totalB),        sub: `${(bc['Minor Damage']??0)+(bc['Major Damage']??0)+destroyed} damaged` },
      { label: 'Damage Score',    value: `${score}/100`,        sub: severity },
      { label: 'Damage Density',  value: `${density}%`,         sub: 'of image area' },
      { label: 'Pixels Damaged',  value: `${totalDmg}%`,        sub: 'of building pixels' },
    ];
    const kW = cW / 4 - 2;
    kpis.forEach(({ label, value, sub }, i) => {
      const kx = margin + i * (kW + 2.67);
      // box
      doc.setFillColor(245, 246, 248);
      doc.setDrawColor(220, 220, 220);
      doc.roundedRect(kx, y, kW, 22, 2, 2, 'FD');
      // label
      doc.setFontSize(7.5); doc.setFont('helvetica', 'normal'); doc.setTextColor(130, 130, 130);
      doc.text(label, kx + 4, y + 6);
      // value
      const vColor: [number,number,number] = label === 'Damage Score'
        ? (score >= 60 ? [200,40,40] : score >= 30 ? [210,110,20] : [30,140,60])
        : [20, 20, 20];
      doc.setFontSize(13); doc.setFont('helvetica', 'bold'); doc.setTextColor(...vColor);
      doc.text(value, kx + 4, y + 14);
      // sub-label
      doc.setFontSize(7); doc.setFont('helvetica', 'normal'); doc.setTextColor(150, 150, 150);
      doc.text(sub, kx + 4, y + 20);
    });
    y += 28;
    rule(6);

    // ── Damage breakdown table ──────────────────────────────────────────────
    heading('Damage Breakdown', 11);

    // Table header
    const cols  = [50, 30, 24, 32, 28, 14];   // widths mm
    const heads = ['Damage Class', 'Pixels', 'Pixel %', 'Buildings', 'Bldg %', ''];
    doc.setFillColor(25, 25, 25);
    doc.rect(margin, y, cW, 8, 'F');
    let cx = margin;
    doc.setFontSize(8); doc.setFont('helvetica', 'bold'); doc.setTextColor(255, 255, 255);
    heads.forEach((h, i) => { doc.text(h, cx + 3, y + 5.5); cx += cols[i]; });
    y += 8;

    // colour map (BGR → RGB for PDF)
    const CLASS_RGB: Record<string, [number,number,number]> = {
      'No Damage':    [230, 230, 230],
      'Minor Damage': [105, 191, 255],
      'Major Damage': [255, 140,   0],
      'Destroyed':    [  0,   0, 139],
    };
    const ORDER = ['Destroyed', 'Major Damage', 'Minor Damage', 'No Damage'];

    ORDER.forEach((label, idx) => {
      const pixCount = bdBreak[label]  ?? 0;
      const pixPct   = pct[label]      ?? 0;
      const bldCount = bc[label]       ?? 0;
      const bldP     = bldPct[label]   ?? 0;

      // row background
      doc.setFillColor(...(idx % 2 === 0 ? [255,255,255] : [248,249,251]) as [number,number,number]);
      doc.rect(margin, y, cW, 8, 'F');

      // colour swatch circle
      const [r,g,b] = CLASS_RGB[label] ?? [200,200,200];
      doc.setFillColor(r, g, b);
      doc.circle(margin + 7, y + 4, 2.5, 'F');

      cx = margin;
      doc.setFontSize(8.5); doc.setFont('helvetica', 'normal'); doc.setTextColor(30, 30, 30);
      doc.text(label,           cx + 13,  y + 5.5); cx += cols[0];
      doc.text(pixCount.toLocaleString(), cx + 3, y + 5.5); cx += cols[1];
      doc.setFont('helvetica', 'bold');
      doc.text(`${pixPct}%`,   cx + 3,   y + 5.5); cx += cols[2];
      doc.setFont('helvetica', 'normal');
      doc.text(String(bldCount), cx + 3, y + 5.5); cx += cols[3];
      doc.setFont('helvetica', 'bold');
      doc.text(`${bldP}%`,     cx + 3,   y + 5.5);
      y += 8;
    });

    // table bottom border
    doc.setDrawColor(220, 220, 220);
    doc.line(margin, y, pageW - margin, y);
    y += 8;
    rule(6);

    // ── Hotspot ────────────────────────────────────────────────────────────
    if (statistics.hotspot) {
      const hs = statistics.hotspot;
      heading('Damage Hotspot', 11);
      body(
        `The primary damage hotspot is located at x=${(hs.x_norm*100).toFixed(1)}%, ` +
        `y=${(hs.y_norm*100).toFixed(1)}% of the image area (based on ${hs.based_on.toLowerCase()} pixels). ` +
        `Emergency response teams should prioritise this area first.`
      );
      y += 4;
      rule(4);
    }

    // ── Footer page 1 ──────────────────────────────────────────────────────
    doc.setFillColor(245, 245, 245);
    doc.rect(0, pageH - 12, pageW, 12, 'F');
    doc.setFontSize(7.5); doc.setFont('helvetica', 'normal'); doc.setTextColor(150, 150, 150);
    doc.text('GeoSense AI  |  For emergency response use only  |  Page 1 of 2', margin, pageH - 4);

    // ════════════════════════════════════════════════════════════════════════
    // PAGE 2 — Damage overlay map (full page, no compression)
    // ════════════════════════════════════════════════════════════════════════
    doc.addPage();
    y = margin;

    // Header band
    doc.setFillColor(15, 15, 15);
    doc.rect(0, 0, pageW, 20, 'F');
    doc.setFillColor(220, 50, 50);
    doc.rect(0, 20, pageW, 2, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(14); doc.setFont('helvetica', 'bold');
    doc.text('Damage Overlay Map', margin, 13);
    doc.setFontSize(8); doc.setFont('helvetica', 'normal'); doc.setTextColor(180, 180, 180);
    doc.text(`Analysis ID: ${id}`, pageW - margin - 80, 13);
    y = 30;

    // Fetch overlay as base64 if it's a URL
    let imgData = overlayBase64;
    if (overlayBase64.startsWith('http')) {
      const resp = await fetch(overlayBase64);
      const blob = await resp.blob();
      imgData = await new Promise<string>((res) => {
        const reader = new FileReader();
        reader.onload = () => res(reader.result as string);
        reader.readAsDataURL(blob);
      });
    }

    // Fit image to full content width, maintaining aspect ratio
    // We read actual image dimensions via an Image element
    const imgDims = await new Promise<{w:number;h:number}>((res) => {
      const img   = new Image();
      img.onload  = () => res({ w: img.naturalWidth, h: img.naturalHeight });
      img.src     = imgData;
    });
    const maxImgH = pageH - y - 28;   // leave room for legend + footer
    const aspect  = imgDims.h / imgDims.w;
    const imgW    = cW;
    const imgH    = Math.min(maxImgH, imgW * aspect);

    // Black background rect behind image (matches app look)
    doc.setFillColor(0, 0, 0);
    doc.rect(margin, y, imgW, imgH, 'F');
    doc.addImage(imgData, 'PNG', margin, y, imgW, imgH, undefined, 'FAST');
    y += imgH + 6;

    // ── Legend row ─────────────────────────────────────────────────────────
    const legendItems: { label: string; rgb: [number,number,number] }[] = [
      { label: 'Background',   rgb: [20,  20,  20 ] },
      { label: 'No Damage',    rgb: [230, 230, 230] },
      { label: 'Minor Damage', rgb: [105, 191, 255] },
      { label: 'Major Damage', rgb: [255, 140,   0] },
      { label: 'Destroyed',    rgb: [  0,   0, 139] },
    ];
    let lx = margin;
    legendItems.forEach(({ label, rgb }) => {
      const [r,g,b] = rgb;
      doc.setFillColor(r,g,b);
      doc.setDrawColor(180,180,180);
      doc.rect(lx, y, 5, 4, 'FD');
      doc.setFontSize(8); doc.setFont('helvetica','normal'); doc.setTextColor(50,50,50);
      doc.text(label, lx + 6.5, y + 3.3);
      lx += label.length * 2.2 + 10;
    });
    y += 10;

    // ── Severity colour scale bar ──────────────────────────────────────────
    doc.setFontSize(8); doc.setFont('helvetica','normal'); doc.setTextColor(100,100,100);
    doc.text(`Damage Score: ${score}/100  (${severity})`, margin, y);

    const barX = margin + 55; const barW = 80; const barH = 4;
    // gradient simulation with rectangles
    const steps = 20;
    for (let s = 0; s < steps; s++) {
      const t  = s / steps;
      const r  = Math.round(30  + t * 225);
      const g  = Math.round(140 - t * 140);
      const b2 = 30;
      doc.setFillColor(r, g, b2);
      doc.rect(barX + s * (barW / steps), y - 3.5, barW / steps + 0.5, barH, 'F');
    }
    // score pointer
    const pointerX = barX + (score / 100) * barW;
    doc.setFillColor(255, 255, 255);
    doc.triangle(pointerX - 2, y + 1, pointerX + 2, y + 1, pointerX, y - 3, 'F');
    doc.setFontSize(7); doc.setTextColor(80,80,80);
    doc.text('Low', barX, y + 5.5);
    doc.text('Severe', barX + barW - 8, y + 5.5);
    y += 12;

    // ── Footer page 2 ──────────────────────────────────────────────────────
    doc.setFillColor(245, 245, 245);
    doc.rect(0, pageH - 12, pageW, 12, 'F');
    doc.setFontSize(7.5); doc.setFont('helvetica','normal'); doc.setTextColor(150,150,150);
    doc.text('GeoSense AI  |  For emergency response use only  |  Page 2 of 2', margin, pageH - 4);

    // ── Save ───────────────────────────────────────────────────────────────
    doc.save(`damage_report_${id}.pdf`);
    toast({ title: 'PDF exported!', description: `damage_report_${id}.pdf  (2 pages)` });

  } catch (err: any) {
    console.error(err);
    toast({ title: 'PDF generation failed', description: err.message, variant: 'destructive' });
  } finally {
    setGeneratingPdf(false);
  }
};

  // ── Region crop ────────────────────────────────────────────────────────────
  const getRelPos = (e: React.MouseEvent) => {
    const el = imgContainerRef.current;
    if (!el) return { x: 0, y: 0 };
    const rect = el.getBoundingClientRect();
    return { x: (e.clientX - rect.left) / rect.width, y: (e.clientY - rect.top) / rect.height };
  };

  const onMouseDown = (e: React.MouseEvent) => {
    if (!cropMode) return; e.preventDefault();
    const pos = getRelPos(e); setDragStart(pos); setBbox(null); setRegionStats(null); setDrawing(true);
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (!cropMode || !drawing || !dragStart) return;
    const pos = getRelPos(e);
    setBbox({ x: Math.min(dragStart.x, pos.x), y: Math.min(dragStart.y, pos.y),
              w: Math.abs(pos.x - dragStart.x), h: Math.abs(pos.y - dragStart.y) });
  };

  const onMouseUp = async (e: React.MouseEvent) => {
    if (!cropMode || !drawing) return;
    setDrawing(false);
    if (!bbox || bbox.w < 0.02 || bbox.h < 0.02) { setBbox(null); return; }
    const id = analysisId || urlAnalysisId;
    if (!id) return;
    setAnalyzingRegion(true);
    try {
      const res  = await fetch("/api/analyze-region", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ analysis_id: id, bbox: { x_norm: bbox.x, y_norm: bbox.y, w_norm: bbox.w, h_norm: bbox.h } }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || "Region analysis failed");
      setRegionStats(data.region_statistics);
      toast({ title: "Region analysed", description: "Statistics updated for selected area." });
    } catch (err: any) {
      toast({ title: "Region analysis failed", description: err.message, variant: "destructive" });
    } finally { setAnalyzingRegion(false); }
  };

  const clearCrop = () => { setBbox(null); setRegionStats(null); setCropMode(false); };

  // ── Guards ─────────────────────────────────────────────────────────────────
  if (loading) return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <Loader2 className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
        <p className="text-lg">Loading analysis results...</p>
      </div>
    </div>
  );
  if (!statistics || !overlayBase64) { navigate("/upload"); return null; }

  const activeStats      = regionStats || statistics;
  const percentages      = activeStats.damage_percentages        || {};
  const buildingCounts   = activeStats.building_counts           || {};
  const buildingCountPct = activeStats.building_count_percentages || {};
  const totalBuildings   = activeStats.total_buildings           ?? 0;
  const weightedScore    = activeStats.weighted_damage_score     ?? 0;
  const damageDensity    = activeStats.damage_density            ?? 0;
  const hotspot          = statistics.hotspot                    ?? null;
  const totalDamaged     = (percentages["Minor Damage"] ?? 0) + (percentages["Major Damage"] ?? 0) + (percentages["Destroyed"] ?? 0);
  const totalBldgDamaged = (buildingCounts["Minor Damage"] ?? 0) + (buildingCounts["Major Damage"] ?? 0) + (buildingCounts["Destroyed"] ?? 0);
  const scoreColor = weightedScore >= 60 ? "text-red-500" : weightedScore >= 30 ? "text-orange-500" : "text-green-500";

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-7xl mx-auto px-6">
        <h1 className="text-4xl font-bold text-center mb-8">Damage Assessment Results</h1>

        <div className="grid lg:grid-cols-3 gap-8">

          {/* ── Left: visualisation ── */}
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex justify-between items-center flex-wrap gap-2">
                  <span>Damage Visualization</span>
                  <div className="flex items-center gap-2 flex-wrap">
                    <Button variant={cropMode ? "default" : "outline"} size="sm"
                      onClick={() => { setCropMode(!cropMode); if (cropMode) clearCrop(); }}>
                      <Crop className="w-4 h-4 mr-1" />{cropMode ? "Cancel" : "Select region"}
                    </Button>
                    <div className="flex rounded-md overflow-hidden border text-sm">
                      <button onClick={() => setShowOverlay(false)}
                        className={`px-3 py-1.5 transition-colors ${!showOverlay ? "bg-primary text-primary-foreground" : "bg-background text-muted-foreground hover:bg-muted"}`}>
                        Original</button>
                      <button onClick={() => setShowOverlay(true)}
                        className={`px-3 py-1.5 transition-colors ${showOverlay ? "bg-primary text-primary-foreground" : "bg-background text-muted-foreground hover:bg-muted"}`}>
                        Overlay</button>
                    </div>
                    <Badge variant={totalDamaged > 20 ? "destructive" : "secondary"}>
                      {totalDamaged.toFixed(1)}% Damaged
                    </Badge>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {cropMode && (
                  <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1">
                    <Crop className="w-3 h-3" /> Click and drag on the image to select a region for focused statistics
                  </p>
                )}
                <div ref={imgContainerRef}
                  className={`relative rounded-lg overflow-hidden bg-black select-none ${cropMode ? "cursor-crosshair" : ""}`}
                  onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp}>
                  <img
                    key={showOverlay ? "overlay" : "original"}
                    src={showOverlay ? overlayBase64 : postImagePreview}
                    alt={showOverlay ? "Damage overlay" : "Original post-disaster"}
                    className="w-full h-auto object-contain pointer-events-none" draggable={false}
                  />
                  {/* Bounding box */}
                  {bbox && (
                    <div className="absolute border-2 border-yellow-400 bg-yellow-400/10 pointer-events-none"
                      style={{ left:`${bbox.x*100}%`, top:`${bbox.y*100}%`, width:`${bbox.w*100}%`, height:`${bbox.h*100}%` }} />
                  )}
                  {/* Hotspot pin */}
                  {showOverlay && hotspot && !bbox && (
                    <div className="absolute w-4 h-4 -translate-x-1/2 -translate-y-1/2 pointer-events-none"
                      style={{ left:`${hotspot.x_norm*100}%`, top:`${hotspot.y_norm*100}%` }}>
                      <div className="w-4 h-4 rounded-full bg-yellow-400 border-2 border-white shadow-lg animate-pulse" />
                    </div>
                  )}
                  {/* Analysing spinner */}
                  {analyzingRegion && (
                    <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                      <div className="text-white text-center">
                        <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2" />
                        <p className="text-sm">Analysing region...</p>
                      </div>
                    </div>
                  )}
                  {/* Legend */}
                  {showOverlay && (
                    <div className="absolute bottom-3 left-3 bg-black/75 text-white px-3 py-2 rounded-lg text-xs backdrop-blur flex flex-wrap gap-x-3 gap-y-1">
                      <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm bg-white border border-gray-400"/>No damage</span>
                      <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm" style={{backgroundColor:"#69bfff"}}/>Minor</span>
                      <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm" style={{backgroundColor:"#ff8c00"}}/>Major</span>
                      <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm" style={{backgroundColor:"#00008b"}}/>Destroyed</span>
                      <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm bg-black border border-gray-600"/>Background</span>
                      {hotspot && <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-full bg-yellow-400"/>Hotspot</span>}
                    </div>
                  )}
                </div>
                {/* Region stats banner */}
                {regionStats && (
                  <div className="mt-3 p-3 bg-yellow-50 dark:bg-yellow-950 border border-yellow-300 rounded-lg flex items-start justify-between gap-2">
                    <div className="text-sm">
                      <p className="font-medium text-yellow-800 dark:text-yellow-200 mb-1">Showing statistics for selected region</p>
                      <p className="text-yellow-700 dark:text-yellow-300 text-xs">
                        {regionStats.total_buildings ?? 0} buildings · damage score {regionStats.weighted_damage_score ?? 0}/100
                      </p>
                    </div>
                    <button onClick={clearCrop} className="text-yellow-600 hover:text-yellow-800 flex-shrink-0">
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Pre / post */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {preImagePreview && (
                <Card>
                  <CardHeader><CardTitle className="text-sm">Pre-Disaster Image</CardTitle></CardHeader>
                  <CardContent className="p-0"><img src={preImagePreview} className="rounded-b-lg w-full" alt="Pre-disaster"/></CardContent>
                </Card>
              )}
              {postImagePreview && (
                <Card>
                  <CardHeader><CardTitle className="text-sm">Post-Disaster Image</CardTitle></CardHeader>
                  <CardContent className="p-0"><img src={postImagePreview} className="rounded-b-lg w-full" alt="Post-disaster"/></CardContent>
                </Card>
              )}
            </div>
          </div>

          {/* ── Right: statistics ── */}
          <div className="space-y-5">
            {regionStats && (
              <div className="flex items-center gap-2 text-xs text-yellow-700 dark:text-yellow-300 bg-yellow-50 dark:bg-yellow-950 border border-yellow-300 px-3 py-2 rounded-lg">
                <Crop className="w-3.5 h-3.5 flex-shrink-0"/>Showing statistics for selected region
              </div>
            )}

            {/* KPIs */}
            <div className="grid grid-cols-2 gap-3">
              <Card className="p-4 flex flex-col gap-1">
                <div className="flex items-center gap-2 text-muted-foreground text-xs"><Building2 className="w-3.5 h-3.5"/>Total buildings</div>
                <p className="text-3xl font-semibold">{totalBuildings}</p>
                <p className="text-xs text-muted-foreground">{totalBldgDamaged} damaged</p>
              </Card>
              <Card className="p-4 flex flex-col gap-1">
                <div className="flex items-center gap-2 text-muted-foreground text-xs"><BarChart3 className="w-3.5 h-3.5"/>Damage score</div>
                <p className={`text-3xl font-semibold ${scoreColor}`}>{weightedScore}<span className="text-base font-normal text-muted-foreground">/100</span></p>
                <p className="text-xs text-muted-foreground">Weighted severity</p>
              </Card>
              <Card className="p-4 flex flex-col gap-1">
                <div className="flex items-center gap-2 text-muted-foreground text-xs"><Target className="w-3.5 h-3.5"/>Damage density</div>
                <p className="text-3xl font-semibold">{damageDensity}<span className="text-base font-normal text-muted-foreground">%</span></p>
                <p className="text-xs text-muted-foreground">Of total image area</p>
              </Card>
              <Card className="p-4 flex flex-col gap-1">
                <div className="flex items-center gap-2 text-muted-foreground text-xs"><MapPin className="w-3.5 h-3.5"/>Hotspot</div>
                {hotspot ? (
                  <><p className="text-sm font-semibold leading-tight">x: {(hotspot.x_norm*100)|0}%<br/>y: {(hotspot.y_norm*100)|0}%</p>
                  <p className="text-xs text-muted-foreground">Based on {hotspot.based_on.toLowerCase()}</p></>
                ) : <p className="text-sm text-muted-foreground">None detected</p>}
              </Card>
            </div>

            {/* Pixel breakdown */}
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Pixel-level breakdown</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                {Object.entries(percentages).map(([label, value]) => (
                  <div key={label}>
                    <div className="flex justify-between mb-1 text-sm">
                      <span className="flex items-center gap-2">
                        <span className="inline-block w-3 h-3 rounded-sm border border-black/10" style={{backgroundColor: DAMAGE_COLORS[label]??""}}/>
                        {label}
                      </span>
                      <span className="text-muted-foreground">{value as number}%</span>
                    </div>
                    <Progress value={value as number}/>
                  </div>
                ))}
                <div className="pt-3 border-t flex items-center gap-2 text-destructive text-sm">
                  <AlertTriangle className="w-4 h-4 flex-shrink-0"/>
                  <span className="font-medium">{totalDamaged.toFixed(1)}% of building pixels damaged</span>
                </div>
              </CardContent>
            </Card>

            {/* Building count */}
            {totalBuildings > 0 && (
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Building count breakdown</CardTitle></CardHeader>
                <CardContent className="space-y-3">
                  {Object.entries(buildingCounts).map(([label, count]) => (
                    <div key={label} className="flex items-center justify-between text-sm">
                      <span className="flex items-center gap-2">
                        <span className="inline-block w-3 h-3 rounded-sm border border-black/10" style={{backgroundColor: DAMAGE_COLORS[label]??""}}/>
                        {label}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground text-xs">{buildingCountPct[label]??0}%</span>
                        <span className={`font-semibold px-2 py-0.5 rounded text-xs ${DAMAGE_BG[label]??""}`}>{count as number} bldgs</span>
                      </div>
                    </div>
                  ))}
                  <div className="pt-2 border-t text-xs text-muted-foreground">{totalBldgDamaged} of {totalBuildings} buildings affected</div>
                </CardContent>
              </Card>
            )}

            {/* Actions */}
            <div className="space-y-2">
              <Button className="w-full" onClick={handleDownloadPDF} disabled={generatingPdf}>
                {generatingPdf
                  ? <><Loader2 className="mr-2 h-4 w-4 animate-spin"/>Generating PDF...</>
                  : <><FileText className="mr-2 h-4 w-4"/>Export PDF Report</>}
              </Button>
              <Button className="w-full" variant="outline" onClick={() => {
                const a = document.createElement("a");
                a.href = overlayBase64; a.download = `damage_overlay_${analysisId||"latest"}.png`; a.click();
              }}><Download className="mr-2 h-4 w-4"/>Download Overlay Image</Button>
              <Button className="w-full" variant="outline" onClick={async () => {
                const id = analysisId || urlAnalysisId; if (!id) return;
                try {
                  const res = await fetch(`/api/history/${id}`);
                  const data = await res.json();
                  const blob = new Blob([JSON.stringify(data.result, null, 2)], {type:"application/json"});
                  const url  = URL.createObjectURL(blob);
                  const a    = document.createElement("a");
                  a.href = url; a.download = `damage_analysis_${id}.json`; a.click();
                  URL.revokeObjectURL(url);
                } catch {}
              }}><Download className="mr-2 h-4 w-4"/>Download JSON Report</Button>
              <Button className="w-full" variant="outline" onClick={handleCopyLink}>
                <Link className="mr-2 h-4 w-4"/>Copy Shareable Link
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;