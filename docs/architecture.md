# Lumoculis Architecture

## Overview

The production website separates prediction from explainability. Prediction runs through the Render-hosted FastAPI/ONNX service. Optional Grad-CAM runs through a separate Modal/PyTorch service, reached through a same-origin Next.js route on Vercel.

## Production prediction flow

1. A user uploads a retinal fundus image in the Next.js frontend.
2. The frontend sends the original image to the Render prediction API.
3. The API validates and preprocesses the image, runs ONNX inference, and returns the predicted grade, class probabilities and other enabled results.
4. The frontend displays the prediction.

## Optional Grad-CAM flow

1. If the user enables Grad-CAM, the browser creates a compressed JPEG copy of the uploaded image. The original image remains unchanged for prediction.
2. The browser sends the compressed copy and predicted class to the Vercel route at POST /api/gradcam.
3. The Vercel route validates the request and applies an Upstash Redis sliding-window limit of 10 requests per IP per minute when Upstash is configured.
4. Vercel forwards the image to the Modal POST /gradcam endpoint using a server-side Bearer token.
5. Modal authenticates the request, downloads the PyTorch checkpoint from private Amazon S3, loads EfficientNet-B0, applies the project preprocessing, and computes Grad-CAM for the requested class.
6. Modal returns a PNG panel containing the processed original image, heatmap and overlay.
7. Vercel returns the PNG to the browser, where the frontend displays and allows downloading it.

Grad-CAM is optional. A Grad-CAM failure is reported separately and does not invalidate a successful prediction.

## Component responsibilities

| Component | Responsibility |
|---|---|
| Next.js frontend on Vercel | Upload UI, prediction display, optional Grad-CAM request, image compression |
| Vercel /api/gradcam route | Server-side proxy, input validation, authentication forwarding and rate limiting |
| Render FastAPI service | Production prediction API and ONNX inference |
| Modal service | CPU-based PyTorch Grad-CAM generation |
| Private Amazon S3 bucket | Stores the PyTorch checkpoint used by Modal |
| Upstash Redis | Shared rate-limit state for the Vercel route |

## Model and preprocessing

- Model family: EfficientNet-B0.
- Number of output classes: five.
- Checkpoint input size: 384 × 384 pixels.
- Class grades: 0, 1, 2, 3 and 4.
- Project definition of referable DR: grade 2 or higher.
- Modal reads model name, class count and image size from the checkpoint configuration.
- Grad-CAM uses the repository preprocessing and explainability modules.

The ONNX prediction artifact and PyTorch Grad-CAM checkpoint are separate deployment artifacts. Keep them version-aligned when updating the model.

## Security boundaries

- Keep the Modal Bearer token in server-side Vercel environment variables only. Never expose it through a NEXT_PUBLIC variable.
- The Modal service checks the Bearer token before processing the uploaded image.
- Keep the S3 bucket private and use an AWS identity with read-only access to the required checkpoint object.
- Keep local environment files, cloud credentials, model checkpoints and datasets out of Git.
- Configure Render CORS with exact trusted frontend origins.
- Configure both Upstash credentials in production. The current route intentionally operates without rate limiting when credentials are absent. If credentials are configured but Redis is unavailable, requests should fail rather than silently bypass the limit.

## Operational behavior

- Modal is configured for CPU inference, one maximum container and scale-down after inactivity.
- The Vercel route has a 90-second maximum duration and an 85-second upstream timeout.
- The browser compresses the image copy used for Grad-CAM to stay below Vercel's request-body limit. The original image remains available for prediction.
- Grad-CAM responses are PNG images and use no-store caching.

## Scientific interpretation

Grad-CAM highlights image regions that influence the model output. It does not prove disease location, pathology or causality. It is an explainability aid for research, not a clinical localization tool.
