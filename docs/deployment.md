# Production Deployment Guide

This guide documents the deployed Lumoculis architecture. Never commit secret values to this repository.

## 1. Frontend — Vercel

Deploy the Next.js application from the frontend directory.

Configure these Vercel environment variables:

| Variable | Purpose |
|---|---|
| NEXT_PUBLIC_API_URL | Base URL of the Render prediction API |
| MODAL_GRADCAM_URL | HTTPS URL of the Modal /gradcam endpoint |
| MODAL_GRADCAM_TOKEN | Bearer token shared with the Modal secret |
| UPSTASH_REDIS_REST_URL | Upstash Redis REST endpoint |
| UPSTASH_REDIS_REST_TOKEN | Upstash Redis REST token |

Set variables for the deployment environments you use (Production and, if needed, Preview). Redeploy after changing environment variables.

MODAL_GRADCAM_TOKEN and UPSTASH_REDIS_REST_TOKEN must remain server-side. Never expose them through NEXT_PUBLIC variables or client-side code.

### Upstash rate limiting

The /api/gradcam route uses Upstash Redis with a sliding-window policy of 10 requests per IP per minute. The limiter is distributed across serverless instances.

The current route creates the limiter only when both Upstash environment variables are present. If either is absent, the route runs without rate limiting. Configure both variables in production. If Redis is configured but unavailable, the request should fail rather than bypass the limit.

### Request size

The browser compresses the image copy used for Grad-CAM before sending it through the Vercel route. This avoids Vercel's serverless request-body size ceiling. The original image is still used for prediction.

## 2. Prediction API — Render

Deploy the FastAPI prediction service separately.

Configure LUMOCULIS_CORS_ORIGINS with the exact frontend origin(s), comma-separated if supported by the backend. For the primary website:

    LUMOCULIS_CORS_ORIGINS=https://lumoculis.vercel.app

Do not use a wildcard origin for a credentialed production application. Save the environment change and allow Render to redeploy.

The frontend NEXT_PUBLIC_API_URL must point to the deployed Render API base URL.

## 3. Grad-CAM service — Modal

The Modal app is defined in deploy/modal/gradcam.py.

The deployed ASGI service exposes:

- POST /gradcam?target_class=0..4
- Multipart file field: file
- Authentication: Authorization: Bearer <token>
- Accepted media types: JPEG and PNG
- Maximum upload size: 10 MiB
- Maximum image dimensions: 4096 × 4096
- Successful response: image/png

The Modal app uses the Modal secret named:

    lumoculis-gradcam-secrets

The secret must contain these keys:

- MODAL_GRADCAM_TOKEN
- LUMOCULIS_S3_BUCKET
- AWS_DEFAULT_REGION
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY

Use a dedicated AWS identity with the minimum permissions required to read the checkpoint object. Never place AWS credentials in source control.

The token in the Modal secret must match the Vercel MODAL_GRADCAM_TOKEN. If the token is rotated, update both services. Secret changes are consumed by newly started Modal containers.

## 4. Model artifact — Amazon S3

The Modal service downloads the PyTorch checkpoint from:

    s3://<LUMOCULIS_S3_BUCKET>/models/best.pt

Keep the bucket private. Confirm the object exists and is the intended checkpoint before deploying a model update. The ONNX prediction artifact is managed separately by the Render prediction service.

## 5. Production verification

After deploying environment changes, test the public website:

1. Open the Analyze page.
2. Submit a retinal image with Grad-CAM disabled; confirm prediction works.
3. Submit an image with Grad-CAM enabled; confirm prediction and visualization both appear.
4. Test a larger image to verify browser-side compression avoids the Vercel request-body limit.
5. Download the Grad-CAM visualization and JSON prediction report.
6. Confirm a Grad-CAM failure does not hide a successful prediction.
7. Send requests above the configured threshold and verify the proxy returns HTTP 429.
8. Confirm the Modal token is not present in browser JavaScript, page source or browser-originated network request headers.
9. Check Render, Vercel, Modal and Upstash logs for errors.

## 6. Cost and operations

Modal is configured for CPU-only execution, one maximum container and scale-down after inactivity. Monitor Modal compute and storage separately; a compute spend limit should not be assumed to cap every possible storage charge.

Monitor Upstash usage and rate-limit analytics. Adjust the per-IP threshold only after observing legitimate traffic and abuse patterns.

## 7. Secret rotation

If a token or cloud credential is exposed:

1. Revoke or rotate it at its source.
2. Update the corresponding secret in the consuming service.
3. Redeploy or restart affected services as required.
4. Verify that the old credential no longer works.
5. Review logs for unexpected use.

Never include actual secret values in documentation, issue reports, screenshots or commits.
