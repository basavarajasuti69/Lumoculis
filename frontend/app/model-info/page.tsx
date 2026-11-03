"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ResearchDisclaimer } from "@/components/ResearchDisclaimer";
import { fetchModelInfo, type ModelInfo } from "@/lib/api";

export default function ModelInfo() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    fetchModelInfo()
      .then((data) => {
        if (mounted) setInfo(data);
      })
      .catch((err) => {
        if (mounted) setError(err instanceof Error ? err.message : "Failed to load model info");
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)] bg-background">
      <div className="max-w-[1200px] mx-auto px-4 py-8 lg:px-8 lg:py-12">
        <div className="flex flex-col lg:flex-row lg:items-start gap-8">
          <div className="flex-1 lg:w-1/2">
            <Card className="border-border bg-card">
              <CardHeader>
                <CardTitle className="text-lg font-bold">Model Information</CardTitle>
              </CardHeader>
              <CardContent className="pt-0 space-y-4">
                {loading && <p className="text-sm text-secondary">Loading model information…</p>}
                {!loading && error && (
                  <p className="text-sm text-destructive">{error}</p>
                )}
                {!loading && info && (
                  <>
                    <p className="text-sm text-secondary">
                      Fetched from the backend at{" "}
                      <span className="font-mono text-xs">/model-info</span>. Only metadata returned by
                      the backend is displayed.
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <InfoRow label="Status" value={info.model_loaded ? "loaded" : "not loaded"} />
                      <InfoRow label="Architecture" value={String(info.architecture ?? "—")} />
                      <InfoRow label="Classes" value={String(info.num_classes ?? "—")} />
                      <InfoRow label="Image size" value={String(info.image_size ?? "—")} />
                      <InfoRow label="Device" value={info.device} />
                      <InfoRow label="Calibration" value={info.calibration_loaded ? "loaded" : "not loaded"} />
                      <InfoRow label="Epoch" value={String(info.epoch ?? "—")} />
                      <InfoRow label="Validation QWK" value={String(info.val_qwk ?? "—")} />
                    </div>
                    {info.error && (
                      <p className="mt-2 text-xs text-destructive">{info.error}</p>
                    )}
                    <p className="mt-4 text-xs text-muted leading-relaxed">
                      <strong className="text-secondary">Model status:</strong> indicates whether the
                      checkpoint loaded successfully at backend startup.{" "}
                      <strong className="text-secondary">Architecture:</strong> the neural network
                      backbone used for feature extraction.{" "}
                      <strong className="text-secondary">Classes:</strong> the number of output
                      severity grades (0–4).{" "}
                      <strong className="text-secondary">Image size:</strong> the input resolution
                      expected by the model.{" "}
                      <strong className="text-secondary">Device:</strong> the compute device used for
                      inference.{" "}
                      <strong className="text-secondary">Calibration:</strong> whether temperature
                      scaling was loaded.{" "}
                      <strong className="text-secondary">Epoch:</strong> the training epoch of the
                      saved checkpoint.{" "}
                      <strong className="text-secondary">Validation QWK:</strong> quadratic weighted
                      kappa on the historical validation set (experimental metric).
                    </p>
                  </>
                )}
              </CardContent>
            </Card>
          </div>

          <div className="flex-1 lg:w-1/2">
            <Card className="border-border bg-card">
              <CardHeader>
                <CardTitle className="text-lg font-bold">Configuration</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <p className="text-sm text-secondary">
                  Configure the backend with environment variables:
                </p>
                <ul className="text-sm text-secondary space-y-2 list-disc pl-5">
                  <li>`LUMOCULIS_CHECKPOINT` — path to the model checkpoint</li>
                  <li>`LUMOCULIS_CALIBRATION` — optional temperature-scaling JSON</li>
                  <li>`LUMOCULIS_DEVICE` — cpu, mps, or auto</li>
                  <li>`LUMOCULIS_MAX_UPLOAD_MB` — maximum upload size</li>
                  <li>`LUMOCULIS_CORS_ORIGINS` — allowed frontend origins</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>

        <div className="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Card className="border-border bg-card">
            <CardHeader>
              <CardTitle className="text-lg font-bold">DR Grade Explanations</CardTitle>
            </CardHeader>
            <CardContent className="pt-0 space-y-3">
              <GradeRow grade="0" label="No diabetic retinopathy" />
              <GradeRow grade="1" label="Mild non-proliferative diabetic retinopathy" />
              <GradeRow grade="2" label="Moderate non-proliferative diabetic retinopathy" />
              <GradeRow grade="3" label="Severe non-proliferative diabetic retinopathy" />
              <GradeRow grade="4" label="Proliferative diabetic retinopathy" />
              <p className="mt-4 text-xs text-muted leading-relaxed">
                Grades 2–4 are considered <strong className="text-secondary">referable DR</strong> in
                this project&apos;s classification logic (grade ≥ 2). This is a research screening
                flag, not a confirmed clinical finding.
              </p>
            </CardContent>
          </Card>

          <Card className="border-border bg-card">
            <CardHeader>
              <CardTitle className="text-lg font-bold">Model Evaluation</CardTitle>
            </CardHeader>
            <CardContent className="pt-0 space-y-3">
              <p className="text-sm text-secondary leading-relaxed">
                <strong className="text-foreground">Quadratic Weighted Kappa (QWK)</strong> measures
                agreement between predicted and actual ordinal grades. Higher values indicate
                better agreement. It is an experimental validation metric, not clinical accuracy.
              </p>
              <p className="text-sm text-secondary leading-relaxed">
                <strong className="text-foreground">Calibration:</strong>{" "}
                {info?.calibration_loaded
                  ? "Temperature scaling is loaded and applied to raw logits to produce calibrated probabilities."
                  : "No calibration file is loaded; raw model probabilities are used."}
              </p>
              <p className="text-xs text-muted leading-relaxed">
                Calibration is intended to improve the reliability of predicted probabilities. It
                does not change the predicted grade or model weights.
              </p>
            </CardContent>
          </Card>
        </div>

        <ResearchDisclaimer className="mx-auto mt-12 max-w-[960px] px-4 lg:px-8" />

        <div className="mx-auto max-w-[960px] px-4 lg:px-8">
          <Link href="/" className="text-sm text-accent hover:text-accent-hover underline-offset-4 hover:underline">
            ← Back to overview
          </Link>
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-sm border-b border-border pb-2">
      <span className="text-secondary">{label}</span>
      <span className="text-foreground font-medium">{value}</span>
    </div>
  );
}

function GradeRow({ grade, label }: { grade: string; label: string }) {
  return (
    <div className="flex items-center gap-3 text-sm border-b border-border pb-2">
      <span className="inline-flex h-6 w-6 items-center justify-center rounded-md border border-border bg-surface-elevated text-accent font-bold text-xs">
        {grade}
      </span>
      <span className="text-secondary">{label}</span>
    </div>
  );
}
