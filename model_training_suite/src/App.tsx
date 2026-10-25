import React, { useState, useEffect, useMemo } from 'react';
import {
  Download,
  Play,
  RefreshCw,
  Copy,
  Check,
  Search,
  ChevronRight,
  Layers,
  BarChart3,
  FileArchive,
  Upload
} from 'lucide-react';

interface ArtifactFile {
  name: string;
  sizeBytes: number;
  sizeFormatted: string;
  category: 'weights' | 'config' | 'vocab' | 'metrics' | 'script';
  url: string;
}

interface ComparisonRow {
  model: string;
  f1_samples: number;
  f1_micro: number;
  precision_micro: number;
  recall_micro: number;
  threshold: number;
  params_M: number;
}

interface EpochMetric {
  epoch: number;
  train_loss: number;
  f1_samples: number;
  f1_micro: number;
  precision_micro: number;
  recall_micro: number;
}

interface ConfusionMatrixItem {
  cui: string;
  name: string;
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

interface SampleCase {
  id: string;
  image: string;
  caption: string;
  true_cuis: Array<{ cui: string; name: string }>;
}

interface StatusPayload {
  ready: boolean;
  files: ArtifactFile[];
  comparison: ComparisonRow[];
  histories: Record<string, EpochMetric[]>;
  configs: Record<string, {
    model: string;
    threshold: number;
    num_labels: number;
    text_dim: number;
    test_metrics: Record<string, number>;
  }>;
  confusionMatrices: ConfusionMatrixItem[];
  sampleCases: SampleCase[];
  trainLog: string;
}

interface PredictionResult {
  model: string;
  latencyMs: number;
  rawOutput: string;
  command: string;
  predictions: Array<{
    probability: number;
    cui: string;
    name: string;
  }>;
}

const MODEL_LABELS: Record<string, { title: string; arch: string; stroke: string }> = {
  cnn: {
    title: 'SimpleCNN (From Scratch)',
    arch: '5-Block Conv2d + AdaptiveAvgPool2d → Linear(256, 500)',
    stroke: '#64748B',
  },
  resnet50: {
    title: 'ResNet-50 (ImageNet V2)',
    arch: 'Bottleneck ConvNet → Dropout(0.3) + Linear(2048, 500)',
    stroke: '#0284C7',
  },
  vit: {
    title: 'ViT-B/16 (Patch16-224)',
    arch: '12-Layer Vision Transformer → Linear(768, 500)',
    stroke: '#4F46E5',
  },
  fusion: {
    title: 'ViT-B/16 + Caption Fusion',
    arch: 'Concat[ViT(768), TF-IDF MLP(4000→512→256)] → Linear(1024, 500)',
    stroke: '#059669',
  },
};

export default function App() {
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [activeSection, setActiveSection] = useState<'artifacts' | 'telemetry' | 'confusion' | 'inference'>('artifacts');
  const [artifactFilter, setArtifactFilter] = useState<'all' | 'weights' | 'config' | 'vocab' | 'metrics' | 'script'>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<'cnn' | 'resnet50' | 'vit' | 'fusion'>('fusion');
  const [selectedSampleId, setSelectedSampleId] = useState<string>('');
  const [captionInput, setCaptionInput] = useState<string>('');
  const [topN, setTopN] = useState<number>(8);
  const [uploadedBase64, setUploadedBase64] = useState<string | null>(null);
  const [uploadedName, setUploadedName] = useState<string>('');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [copiedCmd, setCopiedCmd] = useState<boolean>(false);
  const [inspectConfigModel, setInspectConfigModel] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data: StatusPayload = await res.json();
        setStatus(data);
        if (!selectedSampleId && data.sampleCases && data.sampleCases.length > 0) {
          setSelectedSampleId(data.sampleCases[0].id);
          setCaptionInput(data.sampleCases[0].caption);
        }
      }
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  useEffect(() => {
    if (status && !status.ready) {
      const timer = setInterval(fetchStatus, 4000);
      return () => clearInterval(timer);
    }
  }, [status?.ready]);

  const runLivePrediction = async (
    overrideModel?: 'cnn' | 'resnet50' | 'vit' | 'fusion',
    overrideSampleId?: string,
    overrideCaption?: string
  ) => {
    setPredicting(true);
    try {
      const m = overrideModel || selectedModel;
      const sId = overrideSampleId !== undefined ? overrideSampleId : selectedSampleId;
      const cap = overrideCaption !== undefined ? overrideCaption : captionInput;

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: m,
          sampleId: uploadedBase64 ? undefined : sId,
          imageBase64: uploadedBase64 || undefined,
          caption: cap,
          topn: topN,
        }),
      });
      if (res.ok) {
        const data: PredictionResult = await res.json();
        setPrediction(data);
      }
    } finally {
      setPredicting(false);
    }
  };

  useEffect(() => {
    if (status?.ready && selectedSampleId && !prediction && !predicting) {
      runLivePrediction(selectedModel, selectedSampleId, captionInput);
    }
  }, [status?.ready, selectedSampleId]);

  const handleSampleChange = (sample: SampleCase) => {
    setUploadedBase64(null);
    setUploadedName('');
    setSelectedSampleId(sample.id);
    setCaptionInput(sample.caption);
    runLivePrediction(selectedModel, sample.id, sample.caption);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadedName(file.name);
    const reader = new FileReader();
    reader.onload = () => {
      setUploadedBase64(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const filteredFiles = useMemo(() => {
    if (!status?.files) return [];
    return status.files
      .filter((f) => artifactFilter === 'all' || f.category === artifactFilter)
      .filter((f) => f.name.toLowerCase().includes(searchQuery.toLowerCase()))
      .sort((a, b) => {
        if (a.name.endsWith('.zip')) return -1;
        if (b.name.endsWith('.zip')) return 1;
        if (a.category === 'weights' && b.category !== 'weights') return -1;
        if (b.category === 'weights' && a.category !== 'weights') return 1;
        return a.name.localeCompare(b.name);
      });
  }, [status?.files, artifactFilter, searchQuery]);

  const sortedComparison = useMemo(() => {
    if (!status?.comparison) return [];
    return [...status.comparison].sort((a, b) => b.f1_samples - a.f1_samples);
  }, [status?.comparison]);

  const bestModelRow = sortedComparison[0] || null;
  const activeSample = status?.sampleCases?.find((s) => s.id === selectedSampleId) || null;

  const copyCommand = (cmd: string) => {
    navigator.clipboard.writeText(cmd);
    setCopiedCmd(true);
    setTimeout(() => setCopiedCmd(false), 1800);
  };

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-900 flex flex-col">
      {/* Top Bar Contract */}
      <header className="flex items-center justify-between gap-8 px-6 py-3.5 bg-white border-b border-slate-200 sticky top-0 z-30">
        <a href="#top" className="text-base font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0">
          ROCOv2 Workbench
        </a>

        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600">
          <button
            onClick={() => setActiveSection('artifacts')}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              activeSection === 'artifacts' ? 'text-slate-900 underline underline-offset-8 decoration-2 decoration-blue-600' : ''
            }`}
          >
            Trained Artifacts
          </button>
          <button
            onClick={() => setActiveSection('telemetry')}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              activeSection === 'telemetry' ? 'text-slate-900 underline underline-offset-8 decoration-2 decoration-blue-600' : ''
            }`}
          >
            Benchmarks
          </button>
          <button
            onClick={() => setActiveSection('confusion')}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              activeSection === 'confusion' ? 'text-slate-900 underline underline-offset-8 decoration-2 decoration-blue-600' : ''
            }`}
          >
            Confusion Matrices
          </button>
          <button
            onClick={() => setActiveSection('inference')}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              activeSection === 'inference' ? 'text-slate-900 underline underline-offset-8 decoration-2 decoration-blue-600' : ''
            }`}
          >
            Live Inference
          </button>
        </nav>

        <div className="flex items-center gap-3 shrink-0">
          <a
            href="/api/download/rocov2_model_bundle.zip"
            download="rocov2_model_bundle.zip"
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 rounded hover:bg-blue-700 transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="w-3.5 h-3.5" />
            Download Bundle (.zip)
          </a>
        </div>
      </header>

      <main id="top" className="flex-1 max-w-[1440px] w-full mx-auto px-6 py-6 space-y-6">
        <section className="bg-white border border-slate-200 p-5">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 pb-5 border-b border-slate-200">
            <div>
              <div className="flex items-center gap-2 text-xs text-slate-500 font-mono mb-1">
                <span>ZENODO RECORD 10821435</span>
                <span aria-hidden="true">·</span>
                <span>UMLS CONCEPT DETECTION</span>
                <span aria-hidden="true">·</span>
                <span>
                  {status?.ready ? (
                    <span className="text-emerald-700 font-semibold">● All 4 Models Trained & Verified</span>
                  ) : (
                    <span className="text-amber-700 font-semibold">▲ Training Pipeline Active</span>
                  )}
                </span>
              </div>
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                ROCOv2 Multimodal Concept Detection — Trained Model Repository
              </h1>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              <button
                onClick={fetchStatus}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
                Refresh Telemetry
              </button>
              <a
                href="/api/download/comparison.csv"
                download="comparison.csv"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded transition-colors whitespace-nowrap shrink-0"
              >
                <Download className="w-3.5 h-3.5" />
                comparison.csv
              </a>
              <a
                href="/api/download/label_vocab.json"
                download="label_vocab.json"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded transition-colors whitespace-nowrap shrink-0"
              >
                <Download className="w-3.5 h-3.5" />
                label_vocab.json
              </a>
              <a
                href="/api/download/tfidf.pkl"
                download="tfidf.pkl"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded transition-colors whitespace-nowrap shrink-0"
              >
                <Download className="w-3.5 h-3.5" />
                tfidf.pkl
              </a>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 pt-5">
            <div>
              <div className="text-xs tracking-wider uppercase text-slate-500 font-medium">ROCOv2 Corpus Scale</div>
              <div className="mt-1 flex items-baseline">
                <span className="text-2xl font-mono font-bold text-slate-900 tabular-nums">79,789</span>
                <span className="text-xs uppercase font-mono text-slate-500 ml-1.5">radiology pairs</span>
              </div>
              <div className="mt-1 text-xs font-mono text-slate-500">59,958 train · 9,904 valid · 9,927 test</div>
            </div>

            <div>
              <div className="text-xs tracking-wider uppercase text-slate-500 font-medium">Target Concept Vocabulary</div>
              <div className="mt-1 flex items-baseline">
                <span className="text-2xl font-mono font-bold text-slate-900 tabular-nums">500</span>
                <span className="text-xs uppercase font-mono text-slate-500 ml-1.5">UMLS CUIs</span>
              </div>
              <div className="mt-1 text-xs font-mono text-slate-500">4,000 sublinear TF-IDF (1–2 ngram) features</div>
            </div>

            <div>
              <div className="text-xs tracking-wider uppercase text-slate-500 font-medium">Best Test F1 (Samples)</div>
              <div className="mt-1 flex items-baseline">
                <span className="text-2xl font-mono font-bold text-emerald-700 tabular-nums">
                  {bestModelRow ? (bestModelRow.f1_samples * 100).toFixed(2) : '—'}
                </span>
                <span className="text-xs uppercase font-mono text-slate-500 ml-1.5">%</span>
              </div>
              <div className="mt-1 text-xs font-mono text-slate-600">
                Model: {bestModelRow ? bestModelRow.model.toUpperCase() : 'Training...'} (th={bestModelRow?.threshold ?? '—'})
              </div>
            </div>

            <div>
              <div className="text-xs tracking-wider uppercase text-slate-500 font-medium">Packaged Model Checkpoints</div>
              <div className="mt-1 flex items-baseline">
                <span className="text-2xl font-mono font-bold text-slate-900 tabular-nums">
                  {status?.files?.filter((f) => f.name.endsWith('.pt')).length || 0} / 4
                </span>
                <span className="text-xs uppercase font-mono text-slate-500 ml-1.5">FP16 weights (.pt)</span>
              </div>
              <div className="mt-1 text-xs font-mono text-slate-500">cnn.pt · resnet50.pt · vit.pt · fusion.pt</div>
            </div>
          </div>
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          <aside className="lg:col-span-4 space-y-6">
            <div className="bg-white border border-slate-200 p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-200 pb-3">
                <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                  01. Trained Model Checkpoints
                </h2>
                <span className="text-xs font-mono text-slate-500">PyTorch FP16</span>
              </div>

              <div className="space-y-3">
                {(['fusion', 'vit', 'resnet50', 'cnn'] as const).map((mKey) => {
                  const meta = MODEL_LABELS[mKey];
                  const row = status?.comparison?.find((r) => r.model === mKey);
                  const ptFile = status?.files?.find((f) => f.name === `${mKey}.pt`);
                  const cfgFile = status?.files?.find((f) => f.name === `${mKey}_config.json`);

                  return (
                    <div
                      key={mKey}
                      className={`p-3.5 border transition-colors ${
                        selectedModel === mKey
                          ? 'border-blue-600 bg-blue-50/20'
                          : 'border-slate-200 bg-slate-50/50 hover:border-slate-300'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <div className="text-xs font-mono text-slate-500">
                            {mKey.toUpperCase()} · {row ? `${row.params_M.toFixed(1)}M params` : 'ready'}
                          </div>
                          <div className="text-sm font-semibold text-slate-900 mt-0.5">{meta.title}</div>
                        </div>
                        {row && (
                          <div className="text-right font-mono">
                            <div className="text-sm font-bold text-slate-900 tabular-nums">
                              {(row.f1_samples * 100).toFixed(1)}%
                            </div>
                            <div className="text-[11px] text-slate-500">F1 (samples)</div>
                          </div>
                        )}
                      </div>

                      <div className="mt-2 text-xs text-slate-600 font-mono truncate" title={meta.arch}>
                        {meta.arch}
                      </div>

                      <div className="mt-3 pt-2.5 border-t border-slate-200/80 flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          {ptFile ? (
                            <a
                              href={ptFile.url}
                              download={ptFile.name}
                              className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-mono font-medium text-white bg-slate-900 hover:bg-slate-800 rounded whitespace-nowrap shrink-0"
                            >
                              <Download className="w-3 h-3" />
                              {ptFile.name} ({ptFile.sizeFormatted})
                            </a>
                          ) : (
                            <span className="text-xs font-mono text-amber-700">Building {mKey}.pt...</span>
                          )}

                          {cfgFile && (
                            <button
                              onClick={() => setInspectConfigModel(inspectConfigModel === mKey ? null : mKey)}
                              className="px-2 py-1 text-xs font-mono text-slate-600 hover:text-slate-900 bg-white border border-slate-200 rounded whitespace-nowrap shrink-0 cursor-pointer"
                            >
                              config.json
                            </button>
                          )}
                        </div>

                        <button
                          onClick={() => {
                            setSelectedModel(mKey);
                            setActiveSection('inference');
                            runLivePrediction(mKey, selectedSampleId, captionInput);
                          }}
                          className="text-xs font-medium text-blue-600 hover:text-blue-800 inline-flex items-center gap-0.5 whitespace-nowrap shrink-0 cursor-pointer"
                        >
                          Test
                          <ChevronRight className="w-3.5 h-3.5" />
                        </button>
                      </div>

                      {inspectConfigModel === mKey && status?.configs?.[mKey] && (
                        <pre className="mt-2.5 p-2.5 bg-slate-900 text-slate-100 text-[11px] font-mono overflow-x-auto rounded">
                          {JSON.stringify(status.configs[mKey], null, 2)}
                        </pre>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="bg-white border border-slate-200 p-5 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-200 pb-2.5">
                <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                  02. Local Inference Command
                </h2>
                <button
                  onClick={() =>
                    copyCommand(
                      `pip install torch torchvision timm scikit-learn pandas pillow joblib\npython predict.py --model_dir rocov2_model --model ${selectedModel} --image sample.jpg --caption "${captionInput.slice(0, 45)}..."`
                    )
                  }
                  className="inline-flex items-center gap-1 text-xs font-mono text-slate-600 hover:text-slate-900 cursor-pointer"
                >
                  {copiedCmd ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                  {copiedCmd ? 'Copied' : 'Copy'}
                </button>
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                Unzip <code className="font-mono text-slate-900">rocov2_model_bundle.zip</code> and run{' '}
                <code className="font-mono text-slate-900">predict.py</code> offline—no dataset or internet connection required.
              </p>
              <pre className="p-3 bg-slate-900 text-slate-100 text-xs font-mono overflow-x-auto leading-relaxed">
{`# 1. Install runtime dependencies
pip install torch torchvision timm scikit-learn pandas pillow joblib

# 2. Run multi-label CUI prediction
python predict.py \\
  --model_dir rocov2_model \\
  --model ${selectedModel} \\
  --image xray.jpg \\
  --caption "chest x-ray showing..."`}
              </pre>
            </div>
          </aside>

          <div className="lg:col-span-8 space-y-6">
            <div className="bg-white border border-slate-200 p-2 flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-1">
                <button
                  onClick={() => setActiveSection('artifacts')}
                  className={`inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    activeSection === 'artifacts'
                      ? 'bg-slate-900 text-white'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  <FileArchive className="w-3.5 h-3.5" />
                  All Trained Files ({status?.files?.length || 0})
                </button>
                <button
                  onClick={() => setActiveSection('telemetry')}
                  className={`inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    activeSection === 'telemetry'
                      ? 'bg-slate-900 text-white'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  <BarChart3 className="w-3.5 h-3.5" />
                  Training Curves & Metrics
                </button>
                <button
                  onClick={() => setActiveSection('confusion')}
                  className={`inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    activeSection === 'confusion'
                      ? 'bg-slate-900 text-white'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  <Layers className="w-3.5 h-3.5" />
                  12-Concept Confusion Matrices
                </button>
                <button
                  onClick={() => setActiveSection('inference')}
                  className={`inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    activeSection === 'inference'
                      ? 'bg-slate-900 text-white'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  <Play className="w-3.5 h-3.5" />
                  Live PyTorch Evaluator
                </button>
              </div>
            </div>

            {activeSection === 'artifacts' && (
              <section className="bg-white border border-slate-200 p-5 space-y-5">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">
                  <div>
                    <h2 className="text-base font-bold text-slate-900">
                      Complete Trained Model Files & Artifacts Directory
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Every file generated by <code className="font-mono">train_rocov2.py</code> and{' '}
                      <code className="font-mono">package_model.py</code> ready for direct download.
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-1 bg-slate-100 p-1 rounded">
                    {(['all', 'weights', 'config', 'vocab', 'metrics', 'script'] as const).map((cat) => (
                      <button
                        key={cat}
                        onClick={() => setArtifactFilter(cat)}
                        className={`px-2.5 py-1 text-xs font-medium rounded capitalize transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                          artifactFilter === cat
                            ? 'bg-white text-slate-900 shadow-xs'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        {cat}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="relative">
                  <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Filter files by filename (e.g. fusion.pt, tfidf.pkl, label_vocab.json, comparison.csv)..."
                    className="w-full pl-9 pr-4 py-2 text-xs font-mono bg-slate-50 border border-slate-200 focus:outline-none focus:border-blue-600"
                  />
                </div>

                <div className="overflow-x-auto border border-slate-200">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200 text-[11px] font-mono uppercase text-slate-500">
                        <th className="py-2.5 px-4">Filename</th>
                        <th className="py-2.5 px-4">Artifact Type</th>
                        <th className="py-2.5 px-4 text-right">Size</th>
                        <th className="py-2.5 px-4 text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 text-xs">
                      {filteredFiles.map((file) => (
                        <tr key={file.name} className="hover:bg-slate-50/80">
                          <td className="py-2.5 px-4 font-mono font-medium text-slate-900">{file.name}</td>
                          <td className="py-2.5 px-4 text-slate-500 font-mono">{file.category}</td>
                          <td className="py-2.5 px-4 text-right font-mono tabular-nums text-slate-700">
                            {file.sizeFormatted}
                          </td>
                          <td className="py-2.5 px-4 text-right">
                            <a
                              href={file.url}
                              download={file.name}
                              className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-mono font-medium text-blue-600 hover:text-white hover:bg-blue-600 border border-blue-600 rounded transition-colors whitespace-nowrap shrink-0"
                            >
                              <Download className="w-3 h-3" />
                              Download
                            </a>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="pt-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                      Test Split Evaluation Summary (comparison.csv)
                    </h3>
                    <span className="text-xs font-mono text-slate-500">
                      Threshold tuned on validation F1 (samples)
                    </span>
                  </div>

                  <div className="overflow-x-auto border border-slate-200">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="bg-slate-50 border-b border-slate-200 text-[11px] font-mono uppercase text-slate-500">
                          <th className="py-2.5 px-4">Model</th>
                          <th className="py-2.5 px-4 text-right">F1 (Samples)</th>
                          <th className="py-2.5 px-4 text-right">F1 (Micro)</th>
                          <th className="py-2.5 px-4 text-right">Precision (Micro)</th>
                          <th className="py-2.5 px-4 text-right">Recall (Micro)</th>
                          <th className="py-2.5 px-4 text-right">Threshold</th>
                          <th className="py-2.5 px-4 text-right">Params (M)</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 text-xs font-mono tabular-nums">
                        {sortedComparison.map((row, idx) => (
                          <tr
                            key={row.model}
                            className={idx === 0 ? 'bg-emerald-50/40 font-semibold' : 'hover:bg-slate-50'}
                          >
                            <td className="py-2.5 px-4 text-slate-900 uppercase">
                              {row.model} {idx === 0 && '★'}
                            </td>
                            <td className="py-2.5 px-4 text-right text-slate-900">{row.f1_samples.toFixed(4)}</td>
                            <td className="py-2.5 px-4 text-right text-slate-700">{row.f1_micro.toFixed(4)}</td>
                            <td className="py-2.5 px-4 text-right text-slate-700">{row.precision_micro.toFixed(4)}</td>
                            <td className="py-2.5 px-4 text-right text-slate-700">{row.recall_micro.toFixed(4)}</td>
                            <td className="py-2.5 px-4 text-right text-slate-700">{row.threshold.toFixed(2)}</td>
                            <td className="py-2.5 px-4 text-right text-slate-700">{row.params_M.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </section>
            )}

            {activeSection === 'telemetry' && (
              <section className="bg-white border border-slate-200 p-5 space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-slate-200">
                  <div>
                    <h2 className="text-base font-bold text-slate-900">
                      Epoch-by-Epoch Training Loss & Validation F1 Telemetry
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Interactive SVG curves from <code className="font-mono">*_history.csv</code> and rendered Matplotlib plots.
                    </p>
                  </div>
                  <a
                    href="/api/download/loss_curves.png"
                    download="loss_curves.png"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded whitespace-nowrap shrink-0"
                  >
                    <Download className="w-3.5 h-3.5" />
                    loss_curves.png
                  </a>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="border border-slate-200 p-4 bg-slate-50/40">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                        BCE With Logits Training Loss
                      </span>
                      <span className="text-xs font-mono text-slate-500">Lower is better</span>
                    </div>
                    <svg viewBox="0 0 360 200" className="w-full h-48 overflow-visible">
                      {[0, 0.25, 0.5, 0.75, 1].map((t, idx) => (
                        <line key={idx} x1={36} y1={20 + t * 150} x2={340} y2={20 + t * 150} stroke="#E2E8F0" strokeWidth="1" />
                      ))}
                      {(['cnn', 'resnet50', 'vit', 'fusion'] as const).map((mKey) => {
                        const hist = status?.histories?.[mKey] || [];
                        if (hist.length === 0) return null;
                        const maxLoss = Math.max(
                          0.05,
                          ...Object.values(status?.histories || {})
                            .flat()
                            .map((h) => h.train_loss)
                        );
                        const pts = hist
                          .map((h, i) => {
                            const x = 36 + (i / Math.max(1, hist.length - 1)) * 304;
                            const y = 170 - Math.min(1, h.train_loss / maxLoss) * 145;
                            return `${x},${y}`;
                          })
                          .join(' ');
                        return (
                          <polyline
                            key={mKey}
                            fill="none"
                            stroke={MODEL_LABELS[mKey].stroke}
                            strokeWidth="2.2"
                            points={pts}
                          />
                        );
                      })}
                    </svg>
                  </div>

                  <div className="border border-slate-200 p-4 bg-slate-50/40">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                        Validation F1 (Samples) Progression
                      </span>
                      <span className="text-xs font-mono text-slate-500">Higher is better</span>
                    </div>
                    <svg viewBox="0 0 360 200" className="w-full h-48 overflow-visible">
                      {[0, 0.25, 0.5, 0.75, 1].map((t, idx) => (
                        <line key={idx} x1={36} y1={20 + t * 150} x2={340} y2={20 + t * 150} stroke="#E2E8F0" strokeWidth="1" />
                      ))}
                      {(['cnn', 'resnet50', 'vit', 'fusion'] as const).map((mKey) => {
                        const hist = status?.histories?.[mKey] || [];
                        if (hist.length === 0) return null;
                        const maxF1 = Math.max(
                          0.4,
                          ...Object.values(status?.histories || {})
                            .flat()
                            .map((h) => h.f1_samples)
                        );
                        const pts = hist
                          .map((h, i) => {
                            const x = 36 + (i / Math.max(1, hist.length - 1)) * 304;
                            const y = 170 - Math.min(1, h.f1_samples / maxF1) * 145;
                            return `${x},${y}`;
                          })
                          .join(' ');
                        return (
                          <polyline
                            key={mKey}
                            fill="none"
                            stroke={MODEL_LABELS[mKey].stroke}
                            strokeWidth="2.2"
                            points={pts}
                          />
                        );
                      })}
                    </svg>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                      PyTorch Training Execution Log
                    </span>
                    <span className="text-xs font-mono text-slate-500">train_rocov2.py output</span>
                  </div>
                  <pre className="p-4 bg-slate-900 text-slate-100 text-xs font-mono overflow-x-auto max-h-64 leading-relaxed">
                    {status?.trainLog || 'Training log ready.'}
                  </pre>
                </div>
              </section>
            )}

            {activeSection === 'confusion' && (
              <section className="bg-white border border-slate-200 p-5 space-y-5">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-slate-200">
                  <div>
                    <h2 className="text-base font-bold text-slate-900">
                      Per-Concept Confusion Matrices (12 Most Frequent Test CUIs)
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Evaluated on the top-performing model ({bestModelRow?.model.toUpperCase() || 'FUSION'}) at optimal threshold.
                    </p>
                  </div>
                  <a
                    href="/api/download/confusion_matrices.png"
                    download="confusion_matrices.png"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded whitespace-nowrap shrink-0"
                  >
                    <Download className="w-3.5 h-3.5" />
                    confusion_matrices.png
                  </a>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {(status?.confusionMatrices || []).map((cm) => {
                    const total = cm.tn + cm.fp + cm.fn + cm.tp;
                    const acc = total > 0 ? ((cm.tn + cm.tp) / total) * 100 : 0;
                    return (
                      <div key={cm.cui} className="border border-slate-200 p-3.5 bg-slate-50/40">
                        <div className="flex items-center justify-between gap-2 mb-2">
                          <span className="text-xs font-mono font-bold text-blue-700">{cm.cui}</span>
                          <span className="text-[11px] font-mono text-slate-500 tabular-nums">
                            Acc: {acc.toFixed(1)}%
                          </span>
                        </div>
                        <div className="text-xs font-semibold text-slate-900 truncate mb-3" title={cm.name}>
                          {cm.name}
                        </div>
                        <div className="grid grid-cols-2 gap-1.5 text-center font-mono text-xs tabular-nums">
                          <div className="p-2 bg-white border border-slate-200">
                            <div className="text-[10px] text-slate-400 uppercase">TN (0→0)</div>
                            <div className="font-bold text-slate-800 mt-0.5">{cm.tn}</div>
                          </div>
                          <div className="p-2 bg-amber-50/50 border border-amber-200/70">
                            <div className="text-[10px] text-amber-700 uppercase">FP (0→1)</div>
                            <div className="font-bold text-amber-900 mt-0.5">{cm.fp}</div>
                          </div>
                          <div className="p-2 bg-rose-50/50 border border-rose-200/70">
                            <div className="text-[10px] text-rose-700 uppercase">FN (1→0)</div>
                            <div className="font-bold text-rose-900 mt-0.5">{cm.fn}</div>
                          </div>
                          <div className="p-2 bg-emerald-50/60 border border-emerald-200">
                            <div className="text-[10px] text-emerald-700 uppercase">TP (1→1)</div>
                            <div className="font-bold text-emerald-900 mt-0.5">{cm.tp}</div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            )}

            {activeSection === 'inference' && (
              <section className="bg-white border border-slate-200 p-5 space-y-5">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
                  <div>
                    <h2 className="text-base font-bold text-slate-900">
                      Live PyTorch Model Inference Workbench (predict.py)
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Executes your trained <code className="font-mono">{selectedModel}.pt</code> checkpoint in real time on ROCOv2 test cases or custom uploads.
                    </p>
                  </div>

                  <div className="flex items-center gap-1 bg-slate-100 p-1 rounded">
                    {(['cnn', 'resnet50', 'vit', 'fusion'] as const).map((m) => (
                      <button
                        key={m}
                        onClick={() => {
                          setSelectedModel(m);
                          runLivePrediction(m, selectedSampleId, captionInput);
                        }}
                        className={`px-3 py-1 text-xs font-mono font-semibold uppercase rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                          selectedModel === m
                            ? 'bg-white text-slate-900 shadow-xs'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-600">
                      Select ROCOv2 Test Case or Upload Image
                    </span>
                    <label className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-medium text-blue-600 border border-blue-600 rounded hover:bg-blue-50 cursor-pointer">
                      <Upload className="w-3.5 h-3.5" />
                      {uploadedName ? `Uploaded: ${uploadedName}` : 'Upload Custom Image'}
                      <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
                    </label>
                  </div>

                  <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
                    {(status?.sampleCases || []).slice(0, 6).map((sc) => (
                      <button
                        key={sc.id}
                        onClick={() => handleSampleChange(sc)}
                        className={`p-1.5 border text-left transition-colors cursor-pointer ${
                          !uploadedBase64 && selectedSampleId === sc.id
                            ? 'border-blue-600 bg-blue-50/30'
                            : 'border-slate-200 hover:border-slate-300'
                        }`}
                      >
                        <div className="aspect-square bg-slate-900 overflow-hidden mb-1">
                          <img
                            src={sc.image}
                            alt={sc.id}
                            referrerPolicy="no-referrer"
                            className="w-full h-full object-cover"
                          />
                        </div>
                        <div className="text-[10px] font-mono text-slate-700 truncate">{sc.id}</div>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-12 gap-6 pt-2">
                  <div className="md:col-span-5 space-y-4">
                    <div className="aspect-square bg-slate-900 border border-slate-200 relative overflow-hidden flex items-center justify-center">
                      {uploadedBase64 ? (
                        <img
                          src={uploadedBase64}
                          alt="Uploaded study"
                          referrerPolicy="no-referrer"
                          className="w-full h-full object-contain"
                        />
                      ) : activeSample ? (
                        <img
                          src={activeSample.image}
                          alt={activeSample.id}
                          referrerPolicy="no-referrer"
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <span className="text-xs font-mono text-slate-400">Radiology Study Preview</span>
                      )}
                      <div className="absolute bottom-2 left-2 bg-black/75 text-white px-2 py-0.5 text-[11px] font-mono">
                        224×224 Normalized Tensor
                      </div>
                    </div>

                    <div className="space-y-1.5">
                      <label className="block text-xs font-semibold text-slate-700">
                        Radiology Caption Text (Used by Fusion TF-IDF Encoder)
                      </label>
                      <textarea
                        rows={3}
                        value={captionInput}
                        onChange={(e) => setCaptionInput(e.target.value)}
                        placeholder="Enter radiology findings caption..."
                        className="w-full p-2.5 text-xs font-mono bg-slate-50 border border-slate-200 focus:outline-none focus:border-blue-600"
                      />
                    </div>

                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-slate-600">Top-N:</span>
                        <select
                          value={topN}
                          onChange={(e) => setTopN(Number(e.target.value))}
                          className="px-2 py-1 text-xs font-mono border border-slate-200 bg-white"
                        >
                          <option value={5}>5</option>
                          <option value={8}>8</option>
                          <option value={12}>12</option>
                        </select>
                      </div>

                      <button
                        onClick={() => runLivePrediction()}
                        disabled={predicting}
                        className="flex-1 py-2 px-4 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded inline-flex items-center justify-center gap-1.5 cursor-pointer"
                      >
                        <Play className="w-3.5 h-3.5" />
                        {predicting ? 'Running Inference...' : `Run ${selectedModel.toUpperCase()} Inference`}
                      </button>
                    </div>
                  </div>

                  <div className="md:col-span-7 space-y-4">
                    <div className="flex items-center justify-between border-b border-slate-200 pb-2">
                      <span className="text-xs font-bold uppercase tracking-wider text-slate-800">
                        Predicted UMLS Concepts (CUIs)
                      </span>
                      {prediction && (
                        <span className="text-xs font-mono text-slate-500 tabular-nums">
                          Model: {prediction.model.toUpperCase()} · {prediction.latencyMs} ms
                        </span>
                      )}
                    </div>

                    {prediction?.predictions && prediction.predictions.length > 0 ? (
                      <div className="space-y-2.5">
                        {prediction.predictions.map((pred) => {
                          const isTrueHit = activeSample?.true_cuis?.some((tc) => tc.cui === pred.cui);
                          return (
                            <div key={pred.cui} className="p-3 border border-slate-200 bg-slate-50/50 space-y-1.5">
                              <div className="flex items-center justify-between text-xs">
                                <div className="flex items-center gap-2">
                                  <span className="font-mono font-bold text-blue-700">{pred.cui}</span>
                                  <span className="font-semibold text-slate-900">{pred.name}</span>
                                  {!uploadedBase64 && isTrueHit && (
                                    <span className="text-[11px] font-mono text-emerald-700">· Ground Truth Match</span>
                                  )}
                                </div>
                                <span className="font-mono font-bold text-slate-900 tabular-nums">
                                  {(pred.probability * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="w-full h-1.5 bg-slate-200 overflow-hidden">
                                <div
                                  className="h-full bg-blue-600"
                                  style={{ width: `${Math.min(100, pred.probability * 100)}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="p-8 border border-slate-200 text-center text-xs font-mono text-slate-500">
                        {predicting ? 'Executing predict.py on PyTorch checkpoint...' : 'Click "Run Inference" to evaluate.'}
                      </div>
                    )}
                  </div>
                </div>
              </section>
            )}
          </div>
        </div>
      </main>

      <footer className="mt-12 border-t border-slate-200 bg-white px-6 py-4">
        <div className="max-w-[1440px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div>ROCOv2 Multimodal Concept Detection Suite · Research & Educational Use Only</div>
          <div className="flex items-center gap-4 font-mono">
            <a href="/api/download/train_rocov2.py" download className="hover:text-slate-900">
              train_rocov2.py
            </a>
            <span>·</span>
            <a href="/api/download/predict.py" download className="hover:text-slate-900">
              predict.py
            </a>
            <span>·</span>
            <a href="/api/download/package_model.py" download className="hover:text-slate-900">
              package_model.py
            </a>
            <span>·</span>
            <a href="/api/download/README.md" download className="hover:text-slate-900">
              README.md
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
