const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

export interface Prediction {
  probability: number;
  cui: string;
  name: string;
}

export interface PredictResponse {
  model: string;
  latencyMs: number;
  predictions: Prediction[];
  rawOutput: string;
}

export interface MedicalAnalysisResult {
  caption: string;
  concepts: string[];
  cui: {
    code: string;
    term: string;
    probability: number;
  }[];
  recommendation: {
    specialist: string;
    nextStep: string;
    actions: string[];
    urgency: string;
  };
  model: string;
  latencyMs: number;
}

export interface StatusResponse {
  ready: boolean;
  files: {
    name: string;
    sizeBytes: number;
    sizeFormatted: string;
    category: string;
    url: string;
  }[];
  comparison: Record<string, string | number>[];
  histories: Record<string, { epoch: number; train_loss: number; f1_samples: number }[]>;
  configs: Record<string, unknown>;
  confusionMatrices: {
    cui: string;
    name: string;
    tn: number;
    fp: number;
    fn: number;
    tp: number;
  }[];
  sampleCases: {
    id: string;
    image: string;
    caption: string;
    true_cuis: { cui: string; name: string }[];
  }[];
}

/**
 * Convert a File to base64 data URL string.
 */
function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/**
 * Determine specialist recommendation based on detected concepts.
 */
function getRecommendation(concepts: string[]) {
  const conceptsLower = concepts.map((c) => c.toLowerCase()).join(" ");

  if (
    conceptsLower.includes("lung") ||
    conceptsLower.includes("chest") ||
    conceptsLower.includes("pulmonary") ||
    conceptsLower.includes("thorax")
  ) {
    return {
      specialist: "Pulmonologist / Respiratory Specialist",
      nextStep:
        "Discuss the AI-generated findings with a qualified healthcare professional for thoracic evaluation.",
      actions: [
        "Keep the original medical image and AI report.",
        "Share the results with your pulmonologist or radiologist.",
        "Do not self-diagnose or self-medicate based on the AI result.",
      ],
      urgency:
        "Routine consultation recommended unless you have acute respiratory symptoms.",
    };
  }

  if (
    conceptsLower.includes("brain") ||
    conceptsLower.includes("skull") ||
    conceptsLower.includes("cranial") ||
    conceptsLower.includes("cerebr")
  ) {
    return {
      specialist: "Neurologist / Neuroradiologist",
      nextStep:
        "Consult a neurologist to evaluate the AI-identified neurological features.",
      actions: [
        "Keep the original medical image and AI report.",
        "Share with your neurologist for detailed assessment.",
        "Do not self-diagnose based on the AI result.",
      ],
      urgency:
        "Schedule consultation at earliest convenience; seek emergency care for sudden symptoms.",
    };
  }

  if (
    conceptsLower.includes("bone") ||
    conceptsLower.includes("fracture") ||
    conceptsLower.includes("skeletal") ||
    conceptsLower.includes("joint")
  ) {
    return {
      specialist: "Orthopedic Specialist / Radiologist",
      nextStep:
        "Have the findings reviewed by an orthopedic specialist or musculoskeletal radiologist.",
      actions: [
        "Preserve the original image and AI analysis.",
        "Consult with your orthopedic doctor.",
        "Avoid self-treatment based on AI findings alone.",
      ],
      urgency:
        "Routine follow-up unless experiencing acute pain or trauma.",
    };
  }

  return {
    specialist: "General Radiologist / Primary Care Physician",
    nextStep:
      "Discuss the AI-generated findings with a qualified healthcare professional.",
    actions: [
      "Keep the original medical image and report.",
      "Share the results with your doctor.",
      "Do not self-medicate based on the AI result.",
    ],
    urgency:
      "Routine medical consultation recommended unless you have severe or sudden symptoms.",
  };
}

function checkImageIsNonMedical(file: File): Promise<boolean> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        if (!ctx) return resolve(false);
        const w = 64;
        const h = 64;
        canvas.width = w;
        canvas.height = h;
        ctx.drawImage(img, 0, 0, w, h);
        const data = ctx.getImageData(0, 0, w, h).data;
        let diffSum = 0;
        let saturatedCount = 0;
        for (let i = 0; i < data.length; i += 4) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          const max = Math.max(r, g, b);
          const min = Math.min(r, g, b);
          diffSum += (max - min);
          if (max - min > 35) {
            saturatedCount++;
          }
        }
        const avgDiff = diffSum / (w * h);
        const satRatio = saturatedCount / (w * h);
        // Everyday color photos (grass, nature, cows, sunsets, colorful scenes)
        if (avgDiff > 16 && satRatio > 0.22) {
          resolve(true);
        } else {
          resolve(false);
        }
      } catch {
        resolve(false);
      }
    };
    img.onerror = () => resolve(false);
    img.src = URL.createObjectURL(file);
  });
}

/**
 * Analyze a medical image using the trained ROCOv2 models via the Flask backend.
 */
export async function analyzeMedicalImage(
  file: File,
  model: string = "fusion",
  caption: string = ""
): Promise<MedicalAnalysisResult> {
  console.log("Sending image for AI analysis:", file.name, "model:", model);

  // Client-side guardrail: Check if the input is an everyday non-medical color photograph
  const isNonMedical = await checkImageIsNonMedical(file);
  if (isNonMedical) {
    throw new Error(
      "⚠️ Non-Medical Image Detected: The uploaded image appears to be an everyday color photograph (e.g. animal, nature, or object). This diagnostic AI is strictly calibrated for radiology modalities (CT, MRI, X-Ray, Ultrasound, Angiogram). Please upload a valid clinical radiograph."
    );
  }

  const imageBase64 = await fileToBase64(file);

  const response = await fetch(`${API_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      imageBase64,
      model,
      caption,
      topn: 15,
    }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(
      (err as { error?: string }).error || `Analysis failed (${response.status})`
    );
  }

  const data: PredictResponse = await response.json();

  // Transform predictions into the UI format
  const concepts = data.predictions
    .filter((p) => p.probability >= 0.1)
    .slice(0, 8)
    .map((p) => p.name || p.cui);

  const cuiMappings = data.predictions.slice(0, 10).map((p) => ({
    code: p.cui,
    term: p.name || p.cui,
    probability: p.probability,
  }));

  // Generate caption from top predictions
  const topNames = data.predictions
    .slice(0, 4)
    .map((p) => p.name || p.cui)
    .filter(Boolean);

  const generatedCaption =
    topNames.length > 0
      ? `Medical image analysis identified: ${topNames.join(", ")}. Analysis performed using ${data.model.toUpperCase()} model with ${data.predictions.length} concept${data.predictions.length !== 1 ? "s" : ""} detected.`
      : "Medical image analyzed. No high-confidence concepts were detected.";

  const recommendation = getRecommendation(concepts);

  return {
    caption: generatedCaption,
    concepts,
    cui: cuiMappings,
    recommendation,
    model: data.model,
    latencyMs: data.latencyMs,
  };
}

/**
 * Fetch the training/model status from the backend.
 */
export async function getModelStatus(): Promise<StatusResponse> {
  const response = await fetch(`${API_URL}/api/status`);
  if (!response.ok) {
    throw new Error("Failed to fetch model status");
  }
  return response.json();
}