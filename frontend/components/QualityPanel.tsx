import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";


function formatMeasurement(measured: unknown): string {
  if (measured === null || measured === undefined) return "";
  if (typeof measured === "number") return measured.toFixed(3).replace(/\.?0+$/, "");
  if (typeof measured === "string") return measured;
  if (typeof measured === "boolean") return measured ? "true" : "false";
  if (Array.isArray(measured)) {
    return measured.map(formatMeasurement).join(", ");
  }
  if (typeof measured === "object") {
    return Object.entries(measured as Record<string, unknown>)
      .map(([key, value]) => `${key}: ${formatMeasurement(value)}`)
      .join(", ");
  }
  return String(measured);
}

export function QualityPanel({ quality }: { quality: { status: string; decoded: boolean; image_width?: number | null; image_height?: number | null; reasons: string[]; checks: Array<{ name: string; status: string; measured: unknown; threshold: unknown; reason: string; hard_failure: boolean }> } }) {
  const statusVariant = quality.status === "pass" ? "success" : quality.status === "warning" ? "warning" : "destructive";

  return (
    <Card className="border-border bg-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-sm font-bold">Image Quality Assessment</CardTitle>
        <Badge variant={statusVariant}>{quality.status}</Badge>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-2">
          {quality.reasons.map((reason, i) => (
            <div key={i} className="flex items-start gap-2 text-sm">
              <span className="text-secondary">{reason}</span>
            </div>
          ))}
        </div>
        <Separator className="my-4" />
        <div className="text-xs text-secondary space-y-1">
          {quality.checks.map((check) => (
            <div key={check.name} className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <span className="font-medium text-foreground">{check.name}</span>
                <Badge variant="outline">{check.status}</Badge>
              </div>
              <div className="text-xs">
                {check.reason}
                {check.measured !== undefined && check.measured !== null && (
                  <span className="text-muted"> — measured: {formatMeasurement(check.measured)}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
