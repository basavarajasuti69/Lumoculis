import { AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function ResearchDisclaimer({ className }: { className?: string }) {
  return (
    <Card className={`border-border bg-card text-foreground ${className ?? ""}`}>
      <CardHeader className="flex flex-row items-start gap-3">
        <AlertTriangle className="h-5 w-5 text-accent shrink-0" aria-hidden="true" />
        <CardTitle className="text-sm font-bold">Research prototype — not a medical device</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-base leading-relaxed text-secondary">
          Lumoculis is an experimental research prototype for AI-assisted diabetic retinopathy
          grading. It is not clinically validated, is not a medical device, and must not replace
          professional medical assessment. Image-quality thresholds are heuristic, and Grad-CAM
          highlights model-relevant regions but does not prove disease location or causality.
        </p>
      </CardContent>
    </Card>
  );
}
