import { useState, type ChangeEvent } from "react";
import { analyzeMedicalImage } from "../api/medicalApi";
import type { MedicalAnalysisResult } from "../api/medicalApi";

function Analyze() {
  const [image, setImage] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [showResult, setShowResult] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState("fusion");
  const [caption, setCaption] = useState("");
  const [result, setResult] = useState<MedicalAnalysisResult | null>(null);

  const models = [
    { value: "fusion", label: "ViT + Text Fusion", desc: "Best accuracy — uses image + caption" },
    { value: "vit", label: "ViT-B/16", desc: "Vision Transformer" },
    { value: "resnet50", label: "ResNet-50", desc: "Pretrained CNN" },
    { value: "cnn", label: "Simple CNN", desc: "Trained from scratch" },
  ];

  const handleImageUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    const allowedTypes = ["image/jpeg", "image/png"];
    if (!allowedTypes.includes(selectedFile.type)) {
      setError("Invalid file type. Please upload a JPG, JPEG, or PNG image.");
      event.target.value = "";
      return;
    }

    const maxSize = 5 * 1024 * 1024;
    if (selectedFile.size > maxSize) {
      setError("Image is too large. Please upload an image smaller than 5 MB.");
      event.target.value = "";
      return;
    }

    setFile(selectedFile);
    setImage(URL.createObjectURL(selectedFile));
    setShowResult(false);
    setResult(null);
    setError(null);
  };

  const removeImage = () => {
    setImage(null);
    setFile(null);
    setShowResult(false);
    setIsAnalyzing(false);
    setResult(null);
    setError(null);
  };

  const loadSample = async (url: string, sampleCaption: string) => {
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const sampleFile = new File([blob], url.split("/").pop() || "sample.jpg", { type: "image/jpeg" });
      setFile(sampleFile);
      setImage(url);
      setCaption(sampleCaption);
      setShowResult(false);
      setResult(null);
      setError(null);
    } catch (e) {
      console.error("Failed to load sample image:", e);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const analysisResult = await analyzeMedicalImage(file, selectedModel, caption);
      setResult(analysisResult);
      setShowResult(true);
    } catch (err) {
      console.error("Medical image analysis failed:", err);
      setError(
        err instanceof Error
          ? err.message
          : "Unable to analyze the image. Please try again."
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <section className="analyze-page" id="analyze">

      {/* HEADING */}
      <div className="analyze-heading">
        <p>MEDICAL IMAGE ANALYSIS</p>
        <h1>Analyze Your Image</h1>
        <span>
          Upload a medical image for AI-powered analysis using trained ROCOv2 models.
        </span>
      </div>

      {/* UPLOAD */}
      {!image && (
        <div className="upload-container">
          <label className="upload-box">
            <div className="upload-icon">+</div>
            <h2>Upload Medical Image</h2>
            <p>Click here to select an image or drag & drop</p>
            <small>JPG / JPEG / PNG</small>
            <input
              type="file"
              accept="image/png, image/jpeg"
              onChange={handleImageUpload}
            />
          </label>

          <div className="sample-picks-container">
            <span>Or test with a sample scan:</span>
            <div className="sample-buttons">
              <button
                type="button"
                className="sample-btn"
                onClick={() => loadSample("/samples/ROCOv2_2023_test_000001.jpg", "CT chest axial view showing a huge ascending aortic aneurysm (*).")}
              >
                🫁 Thoracic CT
              </button>
              <button
                type="button"
                className="sample-btn"
                onClick={() => loadSample("/samples/ROCOv2_2023_test_000009.jpg", "Early sagittal T2-weighted MRI scan")}
              >
                🧠 Brain MRI
              </button>
              <button
                type="button"
                className="sample-btn"
                onClick={() => loadSample("/samples/ROCOv2_2023_test_000003.jpg", "Digitally subtracted angiogram demonstrates active extravasation")}
              >
                🩸 DSA Angiogram
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ERROR */}
      {error && (
        <div className="analysis-error">
          <div className="error-badge">⚠️ IMAGE DOMAIN NOTICE</div>
          <p>{error}</p>
          <button
            type="button"
            className="error-sample-btn"
            onClick={() => loadSample("/samples/ROCOv2_2023_test_000001.jpg", "CT chest axial view showing a huge ascending aortic aneurysm (*).")}
          >
            Load Valid Thoracic CT Scan →
          </button>
        </div>
      )}

      {/* IMAGE PREVIEW + MODEL SELECTOR */}
      {image && !showResult && (
        <div className="preview-section">
          <div className="image-preview">
            <img src={image} alt="Uploaded medical image" />
          </div>

          {/* Model Selection */}
          <div className="model-selector">
            <span className="model-selector-label">SELECT AI MODEL</span>
            <div className="model-options">
              {models.map((m) => (
                <button
                  key={m.value}
                  className={`model-option ${selectedModel === m.value ? "model-option-active" : ""}`}
                  onClick={() => setSelectedModel(m.value)}
                >
                  <strong>{m.label}</strong>
                  <small>{m.desc}</small>
                </button>
              ))}
            </div>
          </div>

          {/* Caption input for fusion model */}
          {selectedModel === "fusion" && (
            <div className="caption-input-container">
              <span className="caption-input-label">OPTIONAL: RADIOLOGY CAPTION</span>
              <input
                type="text"
                className="caption-input"
                placeholder="e.g. Chest X-ray showing bilateral lung fields..."
                value={caption}
                onChange={(e) => setCaption(e.target.value)}
              />
              <small className="caption-hint">
                The Fusion model uses both image and text for better accuracy.
              </small>
            </div>
          )}

          <div className="preview-buttons">
            <button className="remove-button" onClick={removeImage}>
              REMOVE IMAGE
            </button>
            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={isAnalyzing}
            >
              {isAnalyzing
                ? "ANALYZING..."
                : `ANALYZE WITH ${selectedModel.toUpperCase()} →`}
            </button>
          </div>
        </div>
      )}

      {/* RESULTS */}
      {image && showResult && result && (
        <>
          <div className="results-title">
            <p>AI ANALYSIS COMPLETE</p>
            <h2>Analysis Results</h2>
            {result.latencyMs && (
              <small className="latency-badge">
                {result.model.toUpperCase()} · {result.latencyMs}ms
              </small>
            )}
          </div>

          <div className="results-section">

            {/* RESULT IMAGE */}
            <div className="result-image">
              <img src={image} alt="Analyzed medical image" />
            </div>

            {/* RESULT CONTENT */}
            <div className="result-content">

              {/* AI CAPTION */}
              <div className="result-card">
                <span>AI GENERATED CAPTION</span>
                <h2>{result.caption}</h2>
              </div>

              {/* MEDICAL CONCEPTS */}
              <div className="result-card">
                <span>MEDICAL CONCEPTS</span>
                <div className="concept-list">
                  {result.concepts.map((concept) => (
                    <div key={concept}>{concept}</div>
                  ))}
                </div>
              </div>

              {/* CUI MAPPING WITH PROBABILITIES */}
              <div className="result-card">
                <span>CUI MAPPING — CONFIDENCE SCORES</span>
                {result.cui.map((item) => (
                  <div className="cui-row" key={item.code}>
                    <strong>{item.code}</strong>
                    <p>{item.term}</p>
                    <div className="probability-bar-container">
                      <div
                        className="probability-bar"
                        style={{ width: `${Math.round(item.probability * 100)}%` }}
                      />
                      <span className="probability-value">
                        {(item.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* RECOMMENDATION */}
              <div className="result-card recommendation-card">
                <span>RECOMMENDED NEXT STEP</span>
                <h3>{result.recommendation.specialist}</h3>
                <p className="recommendation-next">
                  {result.recommendation.nextStep}
                </p>
                <div className="recommendation-actions">
                  {result.recommendation.actions.map((action) => (
                    <div key={action}>→ {action}</div>
                  ))}
                </div>
                <small>{result.recommendation.urgency}</small>
              </div>

            </div>

            {/* NEW ANALYSIS */}
            <button className="new-analysis-button" onClick={removeImage}>
              ANALYZE ANOTHER IMAGE →
            </button>

          </div>
        </>
      )}

    </section>
  );
}

export default Analyze;