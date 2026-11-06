import { NextRequest, NextResponse } from "next/server";
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

export const runtime = "nodejs";

export const maxDuration = 90;

const MAX_FILE_SIZE = 10 * 1024 * 1024;

const ratelimit =
  process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN
    ? new Ratelimit({
        redis: Redis.fromEnv(),
        limiter: Ratelimit.slidingWindow(10, "1 m"),
        analytics: true,
        prefix: "lumoculis:gradcam",
      })
    : null;

export async function POST(request: NextRequest) {
  try {
    if (ratelimit) {
      const ip =
        request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ??
        request.headers.get("x-real-ip") ??
        "anonymous";
      const { success, limit, reset, remaining } = await ratelimit.limit(ip);
      if (!success) {
        return NextResponse.json(
          { error: "Too many requests. Please try again later." },
          {
            status: 429,
            headers: {
              "X-RateLimit-Limit": String(limit),
              "X-RateLimit-Remaining": String(remaining),
              "X-RateLimit-Reset": String(reset),
              "Retry-After": String(Math.ceil((reset - Date.now()) / 1000)),
            },
          }
        );
      }
    }

    const modalUrl = process.env.MODAL_GRADCAM_URL;
    const modalToken = process.env.MODAL_GRADCAM_TOKEN;

    if (!modalUrl || !modalToken) {
      return NextResponse.json(
        { error: "Grad-CAM service is not configured." },
        { status: 503 }
      );
    }

    const endpoint = new URL(modalUrl);

    if (
      endpoint.protocol !== "https:" ||
      !endpoint.hostname.endsWith(".modal.run")
    ) {
      return NextResponse.json(
        { error: "Invalid Grad-CAM service configuration." },
        { status: 500 }
      );
    }

    const formData = await request.formData();
    const file = formData.get("file");
    const targetClass = Number(formData.get("target_class"));

    if (!(file instanceof File)) {
      return NextResponse.json(
        { error: "An image file is required." },
        { status: 400 }
      );
    }

    if (!["image/jpeg", "image/png"].includes(file.type)) {
      return NextResponse.json(
        { error: "Only JPEG and PNG images are supported." },
        { status: 415 }
      );
    }

    if (file.size === 0 || file.size > MAX_FILE_SIZE) {
      return NextResponse.json(
        { error: "Image must be between 1 byte and 10 MB." },
        { status: 413 }
      );
    }

    if (
      !Number.isInteger(targetClass) ||
      targetClass < 0 ||
      targetClass > 4
    ) {
      return NextResponse.json(
        { error: "Invalid target class." },
        { status: 400 }
      );
    }

    const modalForm = new FormData();
    modalForm.append("file", file, file.name);

    endpoint.searchParams.set("target_class", String(targetClass));

    const response = await fetch(endpoint.toString(), {
      method: "POST",
      headers: {
        Authorization: `Bearer ${modalToken}`,
      },
      body: modalForm,
      cache: "no-store",
      signal: AbortSignal.timeout(85000),
    });

    if (!response.ok) {
      console.error("Modal Grad-CAM returned:", response.status);

      return NextResponse.json(
        { error: "Grad-CAM generation failed." },
        { status: 502 }
      );
    }

    const contentType = response.headers.get("content-type");

    if (!contentType?.includes("image/png")) {
      return NextResponse.json(
        { error: "Grad-CAM service returned an invalid response." },
        { status: 502 }
      );
    }

    const image = await response.arrayBuffer();

    return new Response(image, {
      status: 200,
      headers: {
        "Content-Type": "image/png",
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    console.error(
      "Grad-CAM route error:",
      error instanceof Error ? error.message : "Unknown error"
    );

    return NextResponse.json(
      { error: "Unable to generate Grad-CAM visualization." },
      { status: 500 }
    );
  }
}
