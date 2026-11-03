const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface QualityCheckResult {
  name: string;
  status: string;
  measured: unknown;
  threshold: unknown;
  reason: string;
  hard_failure: boolean;
}

export interface QualityResult {
  status: string;
  decoded: boolean;
  image_width: number | null;
  image_height: number | null;
  reasons: string[];
  thresholds: Record<string, unknown>;
  checks: QualityCheckResult[];
}

export interface PredictionResponse {
  predicted_class: number;
  predicted_grade: number;
  referable: boolean;
  referable_probability: number;
  raw_probabilities: number[];
  calibrated_probabilities: number[] | null;
  temperature: number | null;
  temperature_applied: boolean;
  quality: QualityResult | null;
  quality_blocked: boolean;
  gradcam_saved: boolean | null;
  gradcam_url: string | null;
  model_checkpoint: string;
  model_architecture: string | null;
  image_width: number;
  image_height: number;
  filename: string | null;
  disclaimer: string;
}

export interface ModelInfo {
  model_loaded: boolean;
  checkpoint_configured: boolean;
  checkpoint_path: string | null;
  device: string;
  architecture: string | null;
  num_classes: number | null;
  image_size: number | null;
  epoch: number | null;
  val_qwk: number | null;
  calibration_loaded: boolean;
  error: string | null;
  disclaimer: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  checkpoint_configured: boolean;
  device: string;
  version: string;
  error: string | null;
}

export interface PredictOptions {
  runQualityGate?: boolean;
  qualityRequired?: boolean;
  gradcam?: boolean;
  device?: "cpu" | "mps" | "auto";
}

class ApiError extends Error {
  public quality: unknown;
  public errorCode: string | null;

  constructor(public status: number, message: string, public detail?: unknown) {
    super(message);
    this.name = "ApiError";
    // FastAPI wraps HTTPException detail in a "detail" key.
    const unwrapped =
      detail && typeof detail === "object" && "detail" in detail
        ? (detail as { detail: unknown }).detail
        : detail;
    this.errorCode =
      unwrapped && typeof unwrapped === "object" && "error" in unwrapped
        ? String((unwrapped as { error: unknown }).error)
        : null;
    this.quality =
      unwrapped && typeof unwrapped === "object" && "quality" in unwrapped
        ? (unwrapped as { quality: unknown }).quality
        : null;
  }
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail: unknown;
    try {
      detail = await res.json();
    } catch {
      detail = res.statusText;
    }
    // FastAPI wraps HTTPException detail in a "detail" key.
    const unwrapped =
      detail && typeof detail === "object" && "detail" in detail
        ? (detail as { detail: unknown }).detail
        : detail;
    const message =
      (unwrapped && typeof unwrapped === "object" && "error" in unwrapped
        ? String((unwrapped as { error: unknown }).error)
        : null) ||
      (unwrapped && typeof unwrapped === "object" && "message" in unwrapped
        ? String((unwrapped as { message: unknown }).message)
        : null) ||
      `Request failed with status ${res.status}`;
    throw new ApiError(res.status, message, detail);
  }
  return res;
}

export async function predictImage(
  file: File,
  options: PredictOptions = {}
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file, file.name);
  formData.append("run_quality_gate", String(options.runQualityGate ?? false));
  formData.append("quality_required", String(options.qualityRequired ?? false));
  formData.append("gradcam", String(options.gradcam ?? false));
  if (options.device) {
    formData.append("device", options.device);
  }

  const res = await request("/predict", {
    method: "POST",
    body: formData,
  });
  return (await res.json()) as PredictionResponse;
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await request("/model-info", { method: "GET" });
  return (await res.json()) as ModelInfo;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await request("/health", { method: "GET" });
  return (await res.json()) as HealthResponse;
}

export function resolveGradCamUrl(url: string | null | undefined): string | null {
  if (!url) return null;
  try {
    return new URL(url, API_BASE_URL).href;
  } catch {
    return url;
  }
}

export async function downloadGradCam(url: string | null | undefined): Promise<void> {
  const resolved = resolveGradCamUrl(url);
  if (!resolved) {
    throw new Error("Grad-CAM URL is not available.");
  }
  const response = await fetch(resolved);
  if (!response.ok) {
    throw new Error(`Failed to download Grad-CAM (HTTP ${response.status}).`);
  }
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = "lumoculis-gradcam.png";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(objectUrl);
}

export { API_BASE_URL, ApiError };
