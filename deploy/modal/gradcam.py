import io
import os

import boto3
import cv2
import modal
import numpy as np
import torch
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from typing import Optional
from fastapi.responses import Response
from PIL import Image

app = modal.App("lumoculis-gradcam")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "timm>=0.9",
        "numpy>=1.24",
        "opencv-python-headless>=4.8",
        "Pillow>=10",
        "fastapi>=0.100",
        "python-multipart",
        "boto3",
    )
    .add_local_dir("src", remote_path="/root/src")
)

web = FastAPI()
_model = None
_gradcam_class = None
_overlay_heatmap = None


def load_model():
    global _model, _gradcam_class, _overlay_heatmap
    if _model is not None:
        return

    import sys
    sys.path.insert(0, "/root/src")

    from lumoculis.model import build_model
    from lumoculis.explain import GradCAM, overlay_heatmap

    bucket = os.environ["LUMOCULIS_S3_BUCKET"]
    region = os.environ["AWS_DEFAULT_REGION"]
    client = boto3.client("s3", region_name=region)

    checkpoint_path = "/tmp/best.pt"
    client.download_file(bucket, "models/best.pt", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "efficientnet_b0")
    num_classes = int(config.get("num_classes", 5))
    image_size = int(config.get("image_size", 384))

    model = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _model = model
    _gradcam_class = GradCAM
    _overlay_heatmap = overlay_heatmap
    globals()["IMAGE_SIZE"] = image_size


@web.post("/gradcam")
async def generate_gradcam(
    file: UploadFile = File(...),
    target_class: int = Query(..., ge=0, le=4),
    authorization: Optional[str] = Header(default=None),
):
    expected = os.environ["MODAL_GRADCAM_TOKEN"]
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Use JPEG or PNG.")

    raw = await file.read(10 * 1024 * 1024 + 1)
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB.")

    try:
        load_model()

        array = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Invalid image.")

        height, width = bgr.shape[:2]
        if width > 4096 or height > 4096 or width < 32 or height < 32:
            raise ValueError("Image dimensions are not supported.")

        from lumoculis.preprocess import (
            IMAGENET_MEAN, IMAGENET_STD, preprocess
        )

        processed = preprocess(bgr, size=globals()["IMAGE_SIZE"])
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0)

        with _gradcam_class(_model, target_layer=None) as cam:
            heatmap, used_class = cam.compute_heatmap(
                tensor, target_class=target_class
            )

        overlay = _overlay_heatmap(rgb, heatmap)

        # Return a side-by-side original / heatmap / overlay PNG.
        panel = Image.new("RGB", (rgb.shape[1] * 3, rgb.shape[0]))
        panel.paste(Image.fromarray(rgb), (0, 0))
        panel.paste(Image.fromarray(
            cv2.applyColorMap(
                np.uint8(255 * heatmap), cv2.COLORMAP_JET
            )[:, :, ::-1]
        ), (rgb.shape[1], 0))
        panel.paste(Image.fromarray(overlay), (rgb.shape[1] * 2, 0))

        output = io.BytesIO()
        panel.save(output, format="PNG")
        return Response(
            content=output.getvalue(),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Grad-CAM failed: {type(exc).__name__}"
        ) from exc


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("lumoculis-gradcam-secrets")],
    cpu=1,
    memory=2048,
    timeout=90,
    max_containers=1,
    scaledown_window=10,
)
@modal.asgi_app()
def serve():
    return web
