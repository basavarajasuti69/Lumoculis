"use client";

import { useState } from "react";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { downloadGradCam, resolveGradCamUrl } from "@/lib/api";

export function GradCamPanel({
  originalImageUrl,
  gradcamUrl,
}: {
  originalImageUrl: string;
  gradcamUrl: string;
}) {
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const handleDownload = async () => {
    setDownloading(true);
    setDownloadError(null);
    try {
      await downloadGradCam(gradcamUrl);
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : "Failed to download Grad-CAM.");
    } finally {
      setDownloading(false);
    }
  };

  return (
    <Card className="border-border bg-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-sm font-bold">Grad-CAM Visualization</CardTitle>
        {gradcamUrl && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleDownload}
            disabled={downloading}
          >
            {downloading ? "Downloading…" : (
              <>
                <Download className="mr-1 h-4 w-4" /> Download
              </>
            )}
          </Button>
        )}
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="flex flex-col items-center gap-3">
            <div className="w-full aspect-square rounded-lg border border-border bg-surface-elevated overflow-hidden flex items-center justify-center">
              <img
                src={originalImageUrl}
                alt="Original retinal image"
                className="max-h-full max-w-full object-contain"
              />
            </div>
            <p className="text-xs text-center text-secondary">Original</p>
          </div>
          <div className="flex flex-col items-center gap-3">
            <div className="w-full aspect-square rounded-lg border border-border bg-surface-elevated overflow-hidden flex items-center justify-center">
              <img
                src={resolveGradCamUrl(gradcamUrl) ?? ""}
                alt="Grad-CAM visualization"
                className="max-h-full max-w-full object-contain"
              />
            </div>
            <p className="text-xs text-center text-secondary">Grad-CAM sensitivity</p>
          </div>
        </div>
        {downloadError && (
          <p className="mt-3 text-xs text-destructive">{downloadError}</p>
        )}
        <p className="mt-4 text-xs text-secondary leading-relaxed">
          Grad-CAM highlights image regions that influenced the model&apos;s prediction. It does not
          establish disease location or causality.
        </p>
      </CardContent>
    </Card>
  );
}
