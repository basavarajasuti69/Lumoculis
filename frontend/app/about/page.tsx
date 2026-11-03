import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ResearchDisclaimer } from "@/components/ResearchDisclaimer";

export default function About() {
  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)] bg-background">
      <div className="max-w-[1100px] mx-auto px-4 py-8 lg:px-8 lg:py-12 space-y-8">
        {/* Project Overview */}
        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-lg font-bold">Project Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-secondary leading-relaxed">
              Lumoculis is an AI-assisted research prototype for diabetic retinopathy (DR) grading.
              It classifies retinal fundus images into five severity grades (0–4) and flags
              referable DR (grade ≥ 2) using a deep-learning model. The application is designed
              for research and educational exploration of retinal image analysis workflows.
            </p>
          </CardContent>
        </Card>

        {/* Dataset */}
        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-lg font-bold">Dataset</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-secondary leading-relaxed">
              The model was trained and evaluated on the APTOS 2019 dataset for diabetic
              retinopathy grading. The dataset provides retinal fundus images labeled with
              five DR severity grades.
            </p>
          </CardContent>
        </Card>

        {/* Technology Stack */}
        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-lg font-bold">Technology Stack</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-secondary space-y-2 list-disc pl-5">
              <li>Python, PyTorch, torchvision, timm (EfficientNet-B0)</li>
              <li>FastAPI, Pydantic (backend API)</li>
              <li>Next.js, React, TypeScript (frontend)</li>
              <li>Tailwind CSS (styling)</li>
              <li>OpenCV, NumPy, scikit-learn (image processing and metrics)</li>
              <li>MLflow (experiment tracking)</li>
              <li>Streamlit (legacy interface)</li>
            </ul>
          </CardContent>
        </Card>

        {/* Capabilities */}
        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-lg font-bold">Capabilities</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-secondary space-y-2 list-disc pl-5">
              <li>Five-class DR severity classification (grades 0–4).</li>
              <li>Referable DR flagging (grade ≥ 2).</li>
              <li>Image quality assessment (resolution, exposure, blur, retinal field of view).</li>
              <li>Optional Grad-CAM visualization for model explainability.</li>
              <li>Prediction report export (JSON).</li>
            </ul>
          </CardContent>
        </Card>

        {/* About & Workflow (existing) */}
        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-lg font-bold">About Lumoculis</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-secondary leading-relaxed">
              Lumoculis is a research prototype for AI-assisted diabetic retinopathy (DR) grading. It
              classifies retinal fundus images into five severity grades (0–4) and flags referable DR
              (grade ≥ 2) using a deep-learning model trained on APTOS 2019.
            </p>
            <p className="mt-3 text-sm text-secondary leading-relaxed">
              The frontend is a Next.js application that calls a FastAPI backend reusing the existing
              PyTorch inference pipeline. No model code, weights, or training logic were changed for
              this interface.
            </p>
          </CardContent>
        </Card>

        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-lg font-bold">Workflow</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <ol className="list-decimal pl-5 text-sm text-secondary space-y-2 leading-relaxed">
              <li>Upload a PNG or JPEG retinal fundus image.</li>
              <li>Optionally enable image-quality assessment and Grad-CAM visualization.</li>
              <li>Run inference through the FastAPI backend.</li>
              <li>Review the predicted grade, probabilities, quality assessment, and Grad-CAM.</li>
              <li>Download the JSON prediction report for your records.</li>
            </ol>
          </CardContent>
        </Card>

        {/* Limitations */}
        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-lg font-bold">Limitations</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-secondary space-y-2 list-disc pl-5">
              <li>Research prototype, not a medical device.</li>
              <li>Not clinically validated.</li>
              <li>Image-quality checks are heuristic.</li>
              <li>Grad-CAM does not prove disease location or causality.</li>
              <li>Predictions must not replace professional medical assessment.</li>
            </ul>
          </CardContent>
        </Card>

        <ResearchDisclaimer className="mx-auto max-w-[960px]" />
        <Link href="/" className="text-sm text-accent hover:text-accent-hover underline-offset-4 hover:underline">
          ← Back to overview
        </Link>
      </div>
    </div>
  );
}
