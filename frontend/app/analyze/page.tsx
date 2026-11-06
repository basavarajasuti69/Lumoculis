"use client";

import { useCallback, useRef, useState } from "react";
import { Image as ImageIcon, Loader2, ImageMinus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ProbabilityBar } from "@/components/ProbabilityBar";
import { QualityPanel } from "@/components/QualityPanel";
import { GradCamPanel } from "@/components/GradCamPanel";
import { ResearchDisclaimer } from "@/components/ResearchDisclaimer";
import { predictImage, resolveGradCamUrl, downloadGradCam, type PredictionResponse } from "@/lib/api";

const ACCEPTED_EXTENSIONS = ["image/png", "image/jpeg", "image/jpg"];
const MAX_FILE_SIZE = 10 * 1024 * 1024;

export default function Analyze() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [runQualityGate, setRunQualityGate] = useState(false);
  const [qualityRequired, setQualityRequired] = useState(false);
  const [gradcam, setGradcam] = useState(false);
  const [device, setDevice] = useState<"cpu" | "mps" | "auto">("auto");

  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    abortRef.current = null;
  }, [previewUrl]);

  const handleDrop = (acceptedFile: File) => {
    if (!acceptedFile) return;
    const f = acceptedFile;
    if (!ACCEPTED_EXTENSIONS.includes(f.type)) {
      setError("Unsupported file type. Use PNG or JPEG.");
      return;
    }
    if (f.size > MAX_FILE_SIZE) {
      setError(`File too large. Maximum ${(MAX_FILE_SIZE / 1024 / 1024).toFixed(0)} MB.`);
      return;
    }
    setError(null);
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    if (selected) handleDrop(selected);
  };

  const handleRemove = () => {
    reset();
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!file || loading) return;
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await predictImage(file, {
        runQualityGate,
        qualityRequired,
        gradcam,
        device,
      });
      setResult(response);
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") return;
      if (err instanceof Error && (err as { errorCode?: string | null }).errorCode === "quality_gate_failed") {
        const apiErr = err as Error & { quality?: unknown; detail?: unknown };
        const unwrapped =
          apiErr.detail && typeof apiErr.detail === "object" && "detail" in apiErr.detail
            ? (apiErr.detail as { detail: unknown }).detail
            : apiErr.detail;
        const quality =
          unwrapped && typeof unwrapped === "object" && "quality" in unwrapped
            ? (unwrapped as { quality: unknown }).quality
            : null;
        if (quality && typeof quality === "object" && "status" in quality) {
          const q = quality as { status: string; reasons?: string[] };
          const reasons = q.reasons && q.reasons.length > 0 ? ` (${q.reasons.join(", ")})` : "";
          setError(`Image did not pass quality assessment (${q.status})${reasons}. Inference was blocked because quality pass was required.`);
        } else {
          setError("Image did not pass quality assessment. Inference was blocked because quality pass was required.");
        }
      } else {
        setError(
          err instanceof Error
            ? err.message
            : "An unexpected error occurred during analysis."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const downloadJson = useCallback(() => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `lumoculis-prediction-${result.filename ?? "report"}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }, [result]);

  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)] bg-background">
      <div className="max-w-[1280px] mx-auto px-4 py-8 lg:px-8 lg:py-12">
        <div className="flex flex-col lg:flex-row lg:items-start gap-8">
          <div className="flex-1 lg:w-1/2">
            <UploadZone
              file={file}
              previewUrl={previewUrl}
              onFileChange={handleFileChange}
              onRemove={handleRemove}
              disabled={loading}
              onDropFile={handleDrop}
            />
            <Controls
              runQualityGate={runQualityGate}
              setRunQualityGate={setRunQualityGate}
              qualityRequired={qualityRequired}
              setQualityRequired={setQualityRequired}
              gradcam={gradcam}
              setGradcam={setGradcam}
              device={device}
              setDevice={setDevice}
              disabled={loading || !file}
            />
            <Button
              onClick={handleSubmit}
              disabled={loading || !file}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Analyzing…
                </>
              ) : (
                <>Run Analysis</>
              )}
            </Button>
            {error && (
              <div className="mt-4 rounded-md border border-destructive/30 bg-destructive/10 p-3 text-destructive text-sm">
                {error}
              </div>
            )}
          </div>

          <div className="flex-1 lg:w-1/2">
            {result ? (
              <Results result={result} onDownloadJson={downloadJson} previewUrl={previewUrl} />
            ) : (
              <Card className="h-full border-border bg-card">
                <CardContent className="p-8 text-center">
                  <ImageMinus className="mx-auto h-10 w-10 text-secondary mb-3" aria-hidden="true" />
                  <h3 className="text-sm font-bold text-foreground">No results yet</h3>
                  <p className="text-sm text-secondary">Upload a retinal image and run analysis.</p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>

        <ResearchDisclaimer className="mx-auto mt-16 max-w-[960px] px-4 lg:px-8" />
      </div>
    </div>
  );
}

function UploadZone({
  file,
  previewUrl,
  onFileChange,
  onRemove,
  disabled,
  onDropFile,
}: {
  file: File | null;
  previewUrl: string | null;
  onFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onRemove: () => void;
  disabled: boolean;
  onDropFile: (file: File) => void;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  return (
    <Card className="border-border bg-card relative overflow-hidden">
      <div
        className={`relative z-10 flex h-[360px] flex-col items-center justify-center rounded-lg border-2 border-dashed bg-card p-8 text-center transition-colors cursor-pointer disabled:opacity-50 disabled:pointer-events-none ${
          isDragging
            ? "border-accent bg-accent/10"
            : "border-border hover:bg-accent/10 hover:border-accent/50"
        }`}
        onClick={() => { if (!disabled) fileInputRef.current?.click(); }}
        onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); if (!disabled) setIsDragging(true); }}
        onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); }}
        onDrop={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDragging(false);
          if (disabled) return;
          const dropped = e.dataTransfer.files?.[0];
          if (dropped) onDropFile(dropped);
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); if (!disabled) fileInputRef.current?.click(); } }}
        aria-label="Upload retinal image"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png, image/jpeg, image/jpg"
          onChange={onFileChange}
          className="sr-only"
          disabled={disabled}
        />
        {previewUrl ? (
          <>
            <img src={previewUrl} alt="Uploaded image preview" className="max-h-[320px] max-w-full rounded-lg object-cover shadow-lg" />
            <div className="mt-6 flex flex-col gap-1">
              <p className="text-sm font-medium text-foreground truncate" title={file?.name}>
                {file?.name}
              </p>
              <p className="text-xs text-secondary">
                {((file?.size ?? 0) / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <div className="mt-6 flex gap-3">
              <Button variant="outline" size="sm" onClick={(e) => { e.stopPropagation(); onRemove(); }} disabled={disabled}>
                <ImageMinus className="mr-1 h-4 w-4" /> Replace
              </Button>
              <label className="cursor-pointer" onClick={(e) => e.stopPropagation()}>
                <Button variant="outline" size="sm" disabled={disabled}>
                  Choose file
                  <input type="file" accept="image/png, image/jpeg, image/jpg" onChange={onFileChange} className="sr-only" disabled={disabled} />
                </Button>
              </label>
            </div>
          </>
        ) : (
          <>
            <ImageIcon className="mx-auto h-10 w-10 text-secondary mb-4" aria-hidden="true" />
            <h3 className="text-sm font-bold text-foreground">Upload retinal image</h3>
            <p className="text-xs text-secondary">
              Drag and drop a PNG or JPEG image, or click to browse. Maximum 10 MB.
            </p>
            <p className="mt-2 text-xs text-muted">
              PNG, JPEG • Max 10 MB
            </p>
          </>
        )}
      </div>
    </Card>
  );
}

function Controls({
  runQualityGate,
  setRunQualityGate,
  qualityRequired,
  setQualityRequired,
  gradcam,
  setGradcam,
  device,
  setDevice,
  disabled,
}: {
  runQualityGate: boolean;
  setRunQualityGate: (v: boolean) => void;
  qualityRequired: boolean;
  setQualityRequired: (v: boolean) => void;
  gradcam: boolean;
  setGradcam: (v: boolean) => void;
  device: "cpu" | "mps" | "auto";
  setDevice: (v: "cpu" | "mps" | "auto") => void;
  disabled: boolean;
}) {
  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-sm font-bold">Analysis Options</CardTitle>
      </CardHeader>
      <CardContent className="pt-0 space-y-5">
        <div className="flex items-center gap-3">
          <input
            id="run-quality-gate"
            type="checkbox"
            checked={runQualityGate}
            onChange={(e) => setRunQualityGate(e.target.checked)}
            disabled={disabled}
            className="rounded border-border text-accent focus:ring-accent"
          />
          <Label htmlFor="run-quality-gate" className="text-sm cursor-pointer disabled:opacity-50">
            Enable quality gate
          </Label>
          <input
            id="quality-required"
            type="checkbox"
            checked={qualityRequired}
            onChange={(e) => setQualityRequired(e.target.checked)}
            disabled={disabled || !runQualityGate}
            className="rounded border-border text-accent focus:ring-accent"
          />
          <Label htmlFor="quality-required" className="text-sm cursor-pointer disabled:opacity-50">
            Require quality pass to infer
          </Label>
        </div>
        <div className="flex items-center gap-3">
          <input
            id="gradcam"
            type="checkbox"
            checked={gradcam}
            onChange={(e) => setGradcam(e.target.checked)}
            disabled={disabled}
            className="rounded border-border text-accent focus:ring-accent"
          />
          <Label htmlFor="gradcam" className="text-sm cursor-pointer disabled:opacity-50">
            Generate Grad-CAM visualization
          </Label>
        </div>
        <div className="space-y-2">
          <Label htmlFor="device" className="text-xs text-secondary">
            Device
          </Label>
          <Select nativeId="device" value={device} onValueChange={(v: string) => setDevice(v as "cpu" | "mps" | "auto")} disabled={disabled}>
            <option value="auto">Auto (MPS if available, else CPU)</option>
            <option value="cpu">CPU</option>
            <option value="mps">MPS (Apple Silicon)</option>
          </Select>
        </div>
        <p className="text-xs text-muted">
          Analysis runs on the backend. Checkpoint and calibration are server-configured.
        </p>
      </CardContent>
    </Card>
  );
}

function Select({
  nativeId,
  value,
  onValueChange,
  disabled,
  children,
}: {
  nativeId: string;
  value: string;
  onValueChange: (v: string) => void;
  disabled: boolean;
  children?: React.ReactNode;
}) {
  return (
    <select id={nativeId} value={value} onChange={(e) => onValueChange(e.target.value)} disabled={disabled} className="flex h-11 w-full rounded-md border border-border bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-accent disabled:cursor-not-allowed disabled:opacity-50">
      {children}
      <option value="auto">Auto (MPS if available, else CPU)</option>
      <option value="cpu">CPU</option>
      <option value="mps">MPS (Apple Silicon)</option>
    </select>
  );
}

function Results({
  result,
  onDownloadJson,
  previewUrl,
}: {
  result: PredictionResponse;
  onDownloadJson: () => void;
  previewUrl: string | null;
}) {
  const rawBars = result.raw_probabilities.map((p, i) => ({ grade: i, probability: p, isPredicted: i === result.predicted_class }));
  const calibratedBars = result.calibrated_probabilities
    ? result.calibrated_probabilities.map((p, i) => ({ grade: i, probability: p, isPredicted: i === result.predicted_class }))
    : null;

  return (
    <Card className="border-border bg-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-lg font-bold">Analysis Results</CardTitle>
          <p className="text-xs text-secondary">Grade {result.predicted_grade}</p>
        </div>
        <Badge
          variant={result.referable ? "destructive" : "success"}
          className="text-xs font-bold"
        >
          {result.referable ? "Referable DR (grade ≥ 2)" : "Not referable (grade < 2)"}
        </Badge>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-5">
            <h4 className="text-sm font-bold text-foreground">Prediction</h4>
            <div className="flex items-center gap-3 text-3xl font-bold text-foreground">
              Grade {result.predicted_grade}
            </div>
            <p className="text-xs text-secondary">
              Referable probability: {result.referable_probability.toFixed(3)}
            </p>
            <div className="flex items-center gap-3 text-xs text-secondary">
              <span>Architecture:</span> <span className="font-medium">{result.model_architecture ?? "unknown"}</span>
            </div>
            <div className="flex items-center gap-3 text-xs text-secondary">
              <span>Checkpoint:</span> <span className="font-medium truncate max-w-[200px]">{result.model_checkpoint}</span>
            </div>
          </div>

          <div className="space-y-5">
            <h4 className="text-sm font-bold text-foreground">Class probabilities</h4>
            <div className="space-y-4">
              {rawBars.map((bar) => (
                <ProbabilityBar key={bar.grade} grade={bar.grade} probability={bar.probability} isPredicted={bar.isPredicted} />
              ))}
            </div>
            {calibratedBars && (
              <>
                <Separator className="my-4" />
                <h4 className="text-sm font-bold text-foreground">Calibrated probabilities</h4>
                <div className="space-y-4">
                  {calibratedBars.map((bar) => (
                    <ProbabilityBar key={bar.grade} grade={bar.grade} probability={bar.probability} isPredicted={bar.isPredicted} />
                  ))}
                </div>
                <p className="text-xs text-secondary">
                  Temperature: {result.temperature?.toFixed(3) ?? "N/A"}
                </p>
              </>
            )}
          </div>
        </div>

        {result.quality && (
          <div className="mt-8">
            <QualityPanel quality={result.quality} />
            {result.quality_blocked && (
              <p className="mt-3 text-sm text-destructive">
                Inference was blocked because quality gating was required and the image did not pass.
              </p>
            )}
          </div>
        )}

        {result.gradcam_saved && result.gradcam_url && previewUrl && (
          <div className="mt-8">
            <GradCamPanel originalImageUrl={previewUrl ?? ""} gradcamUrl={resolveGradCamUrl(result.gradcam_url) ?? ""} />
          </div>
        )}

        <div className="mt-8 flex items-center gap-3">
          <Button variant="outline" onClick={onDownloadJson} className="flex-1">
            Download JSON Report
          </Button>
        </div>
        <p className="mt-4 text-xs text-secondary leading-relaxed">
          {result.disclaimer}
        </p>
      </CardContent>
    </Card>
  );
}
