import Link from "next/link";
import { Image as ImageIcon, BarChart3, ShieldCheck, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ResearchDisclaimer } from "@/components/ResearchDisclaimer";
import { FeatureCard } from "@/components/FeatureCard";
import Footer from "@/components/Footer";

const FEATURES = [
  {
    title: "FIVE-CLASS CLASSIFICATION",
    description: "Classifies retinal fundus images into diabetic retinopathy severity grades 0–4.",
    icon: <BarChart3 />,
  },
  {
    title: "EXPLAINABLE AI",
    description: "Optional Grad-CAM visualizations show model-sensitive image regions.",
    icon: <ImageIcon />,
  },
  {
    title: "IMAGE QUALITY",
    description: "Optional heuristic image-quality assessment before grading.",
    icon: <ShieldCheck />,
  },
  {
    title: "MODEL EVALUATION",
    description: "Quadratic weighted kappa, accuracy, MAE, balanced accuracy, macro F1, and referable DR AUROC.",
    icon: <Play />,
  },
];

const STEPS = [
  { step: "01", title: "Upload", description: "Select a supported retinal fundus image (PNG, JPEG)." },
  { step: "02", title: "Analyze", description: "Run the inference pipeline on the uploaded image." },
  { step: "03", title: "Review", description: "Inspect grades, probabilities, quality, and Grad-CAM." },
];

// CSS/SVG-based retinal-signal illustration — concentric rings, fine lines, subtle red highlights.
const RetinalSignal = () => (
  <svg
    className="w-full h-full block"
    viewBox="0 0 400 400"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <defs>
      <radialGradient id="glow" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stopColor="#DA291C" stopOpacity="0.15" />
        <stop offset="60%" stopColor="#DA291C" stopOpacity="0.05" />
        <stop offset="100%" stopColor="#DA291C" stopOpacity="0" />
      </radialGradient>
    </defs>

    <circle cx="200" cy="200" r="180" fill="url(#glow)" />

    {[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map((angle) => (
      <line
        key={angle}
        x1={200 + 160 * Math.cos((angle * Math.PI) / 180)}
        y1={200 + 160 * Math.sin((angle * Math.PI) / 180)}
        x2={200 + 185 * Math.cos((angle * Math.PI) / 180)}
        y2={200 + 185 * Math.sin((angle * Math.PI) / 180)}
        stroke="#353535"
        strokeWidth={1}
      />
    ))}

    <circle cx="200" cy="200" r="60" stroke="#DA291C" strokeWidth={1.5} strokeOpacity="0.6" />
    <circle cx="200" cy="200" r="100" stroke="#DA291C" strokeWidth={1} strokeOpacity="0.4" />
    <circle cx="200" cy="200" r="140" stroke="#353535" strokeWidth={0.5} strokeOpacity="0.5" />
    <circle cx="200" cy="200" r="175" stroke="#353535" strokeWidth={0.5} strokeOpacity="0.3" />

    <circle cx="200" cy="200" r="12" fill="#DA291C" />
    <circle cx="200" cy="200" r="4" fill="#F5F5F5" />

    <path
      d="M 200 200 L 200 30 A 170 170 0 0 1 320 200"
      stroke="#DA291C"
      strokeWidth={1}
      strokeOpacity="0.5"
      fill="none"
    />
  </svg>
);

export default function Home() {
  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)] bg-background">
      {/* Hero */}
      <section className="relative isolate px-4 py-16 lg:px-8 lg:py-32 overflow-hidden bg-background">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top_left,#222222_0%,#111111_60%)]" />

        <div className="max-w-[1200px] mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-16 items-center">
            {/* Text column */}
            <div className="space-y-8">
              <div className="inline-flex items-center gap-2 rounded-md border border-border bg-card px-3 py-1 text-xs font-bold text-foreground uppercase tracking-tighter">
                <span className="flex h-1.5 w-1.5 rounded-full bg-accent" />
                Research Prototype
              </div>
              <h1 className="mt-2 text-6xl font-bold tracking-tight text-foreground lg:text-7xl leading-[1.05]">
                See beyond the image.<br />
                <span className="text-accent">Understand the signal.</span>
              </h1>
              <p className="mt-6 max-w-none lg:max-w-[62ch] text-lg text-secondary leading-relaxed">
                Lumoculis is an AI-assisted research prototype that classifies retinal fundus images
                into five diabetic retinopathy severity grades (0–4) and flags referable DR (grade ≥ 2).
              </p>
              <div className="mt-12 flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                <Button asChild size="lg" className="w-full sm:w-auto">
                  <Link href="/analyze">ANALYZE AN IMAGE</Link>
                </Button>
                <Button variant="outline" asChild size="lg" className="w-full sm:w-auto">
                  <Link href="/model-info">EXPLORE THE MODEL</Link>
                </Button>
              </div>
            </div>

            {/* Visual column */}
            <div className="hidden lg:block flex items-center justify-center">
              <div className="relative w-full max-w-[520px] aspect-square">
                <div className="absolute inset-0 bg-gradient-to-tr from-accent/10 to-transparent rounded-full blur-[80px]" />
                <div className="relative z-10 w-full h-full rounded-full border border-border overflow-hidden bg-card">
                  <RetinalSignal />
                  <div className="absolute bottom-6 left-1/2 -translate-x-1/2 text-center">
                    <p className="text-xs font-bold text-foreground uppercase tracking-tighter">
                      Retinal Signal
                    </p>
                    <p className="mt-1 text-xs text-muted">
                      Upload a fundus photograph to begin
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-24 border-y border-border bg-background">
        <div className="max-w-[1200px] mx-auto px-4 lg:px-8">
          <div className="text-center lg:text-left">
            <h2 className="text-4xl font-bold tracking-tight text-foreground">Technical highlights</h2>
            <p className="mt-2 text-secondary">
              Capabilities implemented in the current research prototype.
            </p>
          </div>
          <div className="mt-16 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
            {FEATURES.map((feature) => (
              <FeatureCard key={feature.title} {...feature} />
            ))}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="py-24 bg-background">
        <div className="max-w-[1200px] mx-auto px-4 lg:px-8">
          <div className="text-center lg:text-left">
            <h2 className="text-4xl font-bold tracking-tight text-foreground">How it works</h2>
          </div>
          <div className="mt-16 grid grid-cols-1 sm:grid-cols-3 gap-12">
            {STEPS.map((step) => (
              <div key={step.step} className="flex flex-col gap-4">
                <span className="inline-flex h-12 w-12 items-center justify-center rounded-md border border-border bg-card text-accent font-bold text-lg">
                  {step.step}
                </span>
                <h3 className="text-base font-bold text-foreground">{step.title}</h3>
                <p className="text-sm text-secondary leading-relaxed">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <ResearchDisclaimer className="mx-auto mt-24 max-w-[960px] px-4 lg:px-8" />

      <Footer />
    </div>
  );
}
