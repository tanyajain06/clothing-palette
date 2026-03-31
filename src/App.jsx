import { useRef, useState } from "react";
import { FilesetResolver, ImageSegmenter } from "@mediapipe/tasks-vision";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";

const CLOTHES_CATEGORY = 4;

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((v) =>
        Math.max(0, Math.min(255, Math.round(v)))
          .toString(16)
          .padStart(2, "0")
      )
      .join("")
  );
}

function colorDistance(a, b) {
  return (
    (a[0] - b[0]) ** 2 +
    (a[1] - b[1]) ** 2 +
    (a[2] - b[2]) ** 2
  );
}

function kMeans(points, k = 5, iterations = 8) {
  if (!points.length) return [];

  const actualK = Math.min(k, points.length);
  let centers = [];

  for (let i = 0; i < actualK; i += 1) {
    const idx = Math.floor((i * points.length) / actualK);
    centers.push([...points[idx]]);
  }

  for (let iter = 0; iter < iterations; iter += 1) {
    const groups = Array.from({ length: actualK }, () => []);

    for (const point of points) {
      let bestIndex = 0;
      let bestDist = Infinity;

      for (let i = 0; i < centers.length; i += 1) {
        const dist = colorDistance(point, centers[i]);
        if (dist < bestDist) {
          bestDist = dist;
          bestIndex = i;
        }
      }

      groups[bestIndex].push(point);
    }

    centers = groups.map((group, i) => {
      if (group.length === 0) return centers[i];
      const sum = group.reduce(
        (acc, p) => [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]],
        [0, 0, 0]
      );
      return [
        sum[0] / group.length,
        sum[1] / group.length,
        sum[2] / group.length,
      ];
    });
  }

  const counts = Array(centers.length).fill(0);

  for (const point of points) {
    let bestIndex = 0;
    let bestDist = Infinity;

    for (let i = 0; i < centers.length; i += 1) {
      const dist = colorDistance(point, centers[i]);
      if (dist < bestDist) {
        bestDist = dist;
        bestIndex = i;
      }
    }

    counts[bestIndex] += 1;
  }

  return centers
    .map((center, i) => ({
      rgb: center,
      hex: rgbToHex(center[0], center[1], center[2]),
      share: counts[i] / points.length,
    }))
    .sort((a, b) => b.share - a.share);
}

function dedupeSimilarColors(colors, threshold = 28) {
  const kept = [];

  for (const color of colors) {
    const tooClose = kept.some((existing) => {
      return Math.sqrt(colorDistance(color.rgb, existing.rgb)) < threshold;
    });

    if (!tooClose) {
      kept.push(color);
    }
  }

  return kept;
}

async function loadImageFromFile(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await img.decode();
  return img;
}

export default function App() {
  const [palette, setPalette] = useState([]);
  const [status, setStatus] = useState("Upload a clothing photo.");
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingTime, setProcessingTime] = useState(null);

  const imageCanvasRef = useRef(null);
  const maskedCanvasRef = useRef(null);
  const segmenterRef = useRef(null);

  async function getSegmenter() {
    if (segmenterRef.current) return segmenterRef.current;

    setIsLoadingModel(true);
    setStatus("Loading segmentation model...");

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    const segmenter = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
      },
      runningMode: "IMAGE",
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });

    segmenterRef.current = segmenter;
    setIsLoadingModel(false);
    return segmenter;
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setPalette([]);
    setProcessingTime(null);

    if (!file.type.startsWith("image/")) {
      setStatus("Sorry, please only upload an image file.");
      return;
    }

    setIsProcessing(true);
    setStatus("Loading image...");

    const startTime = performance.now();

    try {
      const image = await loadImageFromFile(file);
      const segmenter = await getSegmenter();

      const maxSide = 768;
      const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
      const width = Math.max(1, Math.round(image.width * scale));
      const height = Math.max(1, Math.round(image.height * scale));

      const imageCanvas = imageCanvasRef.current;
      const maskedCanvas = maskedCanvasRef.current;

      const imageCtx = imageCanvas.getContext("2d", { willReadFrequently: true });
      const maskedCtx = maskedCanvas.getContext("2d", { willReadFrequently: true });

      imageCanvas.width = width;
      imageCanvas.height = height;
      maskedCanvas.width = width;
      maskedCanvas.height = height;

      imageCtx.clearRect(0, 0, width, height);
      maskedCtx.clearRect(0, 0, width, height);
      imageCtx.drawImage(image, 0, 0, width, height);

      setStatus("Segmenting clothing from image...");

      const result = segmenter.segment(imageCanvas);
      const categoryMask = result.categoryMask;
      const maskData = categoryMask.getAsUint8Array();

      const imageData = imageCtx.getImageData(0, 0, width, height);
      const pixels = imageData.data;
      const maskedImageData = imageCtx.createImageData(width, height);

      const clothingPixels = [];
      const sampleStride = 2;

      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const idx = y * width + x;
          const pixelOffset = idx * 4;
          const category = maskData[idx];

          if (category === CLOTHES_CATEGORY) {
            maskedImageData.data[pixelOffset] = pixels[pixelOffset];
            maskedImageData.data[pixelOffset + 1] = pixels[pixelOffset + 1];
            maskedImageData.data[pixelOffset + 2] = pixels[pixelOffset + 2];
            maskedImageData.data[pixelOffset + 3] = 255;

            if (x % sampleStride === 0 && y % sampleStride === 0) {
              const r = pixels[pixelOffset];
              const g = pixels[pixelOffset + 1];
              const b = pixels[pixelOffset + 2];

              const max = Math.max(r, g, b);
              const min = Math.min(r, g, b);

              if (max > 20) {
                clothingPixels.push([r, g, b]);

                if (max - min > 35) {
                  clothingPixels.push([r, g, b]);
                }
              }
            }
          } else {
            maskedImageData.data[pixelOffset] = 0;
            maskedImageData.data[pixelOffset + 1] = 0;
            maskedImageData.data[pixelOffset + 2] = 0;
            maskedImageData.data[pixelOffset + 3] = 0;
          }
        }
      }

      maskedCtx.putImageData(maskedImageData, 0, 0);

      if (clothingPixels.length < 30) {
        setStatus("Error with num of clothing pixels. Upload a clearer outfit photo.");
        setPalette([]);
        setProcessingTime(Math.round(performance.now() - startTime));
        return;
      }

      setStatus("Extracting clothing colors...");

      const clustered = kMeans(clothingPixels, 5, 10).filter(
        (c) => c.share > 0.03
      );

      const cleanPalette = dedupeSimilarColors(clustered).slice(0, 5);

      setPalette(cleanPalette);
      setProcessingTime(Math.round(performance.now() - startTime));
      setStatus("Done.");
    } catch (error) {
      console.error(error);
      setStatus("Error while processing the image.");
      setPalette([]);
      setProcessingTime(null);
    } finally {
      setIsProcessing(false);
      setIsLoadingModel(false);
    }
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0b1020",
        color: "white",
        padding: "32px",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
        <h1 style={{ fontSize: "52px", marginBottom: "16px", textAlign: "center" }}>
          Tanya's Closet
        </h1>

        <div style={{ textAlign: "center", marginBottom: "20px" }}>
          <input type="file" accept="image/*" onChange={handleUpload} />
        </div>

        <div style={{ textAlign: "center", marginBottom: "10px", color: "#cbd5e1" }}>
          {isLoadingModel ? "Loading model..." : status}
        </div>

        {processingTime !== null && (
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              marginBottom: "24px",
            }}
          >
            <div
              style={{
                background: "#1f2937",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "999px",
                padding: "10px 18px",
                fontWeight: "700",
                fontSize: "14px",
                color: "white",
                boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
              }}
            >
              ⚡ Extraction Time: {processingTime} ms
            </div>
          </div>
        )}

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "24px",
            alignItems: "start",
          }}
        >
          <div>
            <h2 style={{ marginBottom: "12px" }}>Original</h2>
            <canvas
              ref={imageCanvasRef}
              style={{
                width: "100%",
                maxWidth: "480px",
                background: "#111827",
                borderRadius: "12px",
              }}
            />
          </div>

          <div>
            <h2 style={{ marginBottom: "12px" }}>Clothing Only</h2>
            <canvas
              ref={maskedCanvasRef}
              style={{
                width: "100%",
                maxWidth: "480px",
                background: "#111827",
                borderRadius: "12px",
              }}
            />
          </div>
        </div>

        <div style={{ marginTop: "32px" }}>
          <h2 style={{ marginBottom: "16px" }}>Palette</h2>

          <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
            {palette.map((color) => (
              <div key={color.hex} style={{ textAlign: "center" }}>
                <div
                  style={{
                    width: "100px",
                    height: "100px",
                    borderRadius: "12px",
                    background: color.hex,
                    border: "1px solid rgba(255,255,255,0.12)",
                  }}
                />
                <div style={{ marginTop: "8px", fontSize: "14px", fontWeight: "600" }}>
                  {color.hex}
                </div>
                <div style={{ fontSize: "12px", color: "#94a3b8" }}>
                  {(color.share * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        {isProcessing && (
          <div style={{ marginTop: "20px", color: "#94a3b8", textAlign: "center" }}>
            Processing...
          </div>
        )}
      </div>
    </div>
  );
}