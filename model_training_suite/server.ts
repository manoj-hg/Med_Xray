import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import fs from 'fs';
import { execFile } from 'child_process';

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '15mb' }));

const MODEL_DIR = path.resolve(process.cwd(), 'public/rocov2_model');

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(i >= 2 ? 2 : 1)} ${sizes[i]}`;
}

app.get('/api/status', (_req, res) => {
  try {
    const logPath = '/tmp/train.log';
    const trainLog = fs.existsSync(logPath) ? fs.readFileSync(logPath, 'utf-8') : '';

    const files: Array<{
      name: string;
      sizeBytes: number;
      sizeFormatted: string;
      category: 'weights' | 'config' | 'vocab' | 'metrics' | 'script';
      url: string;
    }> = [];

    if (fs.existsSync(MODEL_DIR)) {
      const entries = fs.readdirSync(MODEL_DIR);
      for (const entry of entries) {
        const fullPath = path.join(MODEL_DIR, entry);
        const stat = fs.statSync(fullPath);
        if (stat.isFile()) {
          let category: 'weights' | 'config' | 'vocab' | 'metrics' | 'script' = 'metrics';
          if (entry.endsWith('.pt') || entry.endsWith('.zip')) category = 'weights';
          else if (entry.endsWith('_config.json')) category = 'config';
          else if (entry === 'label_vocab.json' || entry === 'tfidf.pkl' || entry === 'cui_mapping.csv') category = 'vocab';
          else if (entry.endsWith('.py') || entry === 'README.md') category = 'script';

          files.push({
            name: entry,
            sizeBytes: stat.size,
            sizeFormatted: formatBytes(stat.size),
            category,
            url: `/api/download/${encodeURIComponent(entry)}`,
          });
        }
      }
    }

    // Parse comparison.csv if available
    const comparison: Array<Record<string, string | number>> = [];
    const cmpPath = path.join(MODEL_DIR, 'comparison.csv');
    if (fs.existsSync(cmpPath)) {
      const lines = fs.readFileSync(cmpPath, 'utf-8').trim().split('\n');
      if (lines.length > 1) {
        const headers = lines[0].split(',');
        for (let i = 1; i < lines.length; i++) {
          const vals = lines[i].split(',');
          const row: Record<string, string | number> = {};
          headers.forEach((h, idx) => {
            const v = vals[idx];
            row[h] = isNaN(Number(v)) ? v : Number(v);
          });
          comparison.push(row);
        }
      }
    }

    // Parse histories
    const histories: Record<string, Array<Record<string, number>>> = {};
    for (const m of ['cnn', 'resnet50', 'vit', 'fusion']) {
      const hPath = path.join(MODEL_DIR, `${m}_history.csv`);
      if (fs.existsSync(hPath)) {
        const lines = fs.readFileSync(hPath, 'utf-8').trim().split('\n');
        if (lines.length > 1) {
          const headers = lines[0].split(',');
          histories[m] = lines.slice(1).map((line) => {
            const vals = line.split(',');
            const obj: Record<string, number> = {};
            headers.forEach((h, idx) => {
              obj[h] = Number(vals[idx]);
            });
            return obj;
          });
        }
      }
    }

    // Parse configs
    const configs: Record<string, unknown> = {};
    for (const m of ['cnn', 'resnet50', 'vit', 'fusion']) {
      const cPath = path.join(MODEL_DIR, `${m}_config.json`);
      if (fs.existsSync(cPath)) {
        configs[m] = JSON.parse(fs.readFileSync(cPath, 'utf-8'));
      }
    }

    // Confusion matrices & sample cases
    const cmPath = path.join(MODEL_DIR, 'confusion_matrices.json');
    const confusionMatrices = fs.existsSync(cmPath) ? JSON.parse(fs.readFileSync(cmPath, 'utf-8')) : [];

    const scPath = path.join(MODEL_DIR, 'sample_cases.json');
    const sampleCases = fs.existsSync(scPath) ? JSON.parse(fs.readFileSync(scPath, 'utf-8')) : [];

    res.json({
      ready: comparison.length === 4,
      files,
      comparison,
      histories,
      configs,
      confusionMatrices,
      sampleCases,
      trainLog,
    });
  } catch (err: unknown) {
    res.status(500).json({ error: err instanceof Error ? err.message : 'Status check failed' });
  }
});

app.get('/api/download/:filename', (req, res) => {
  const filename = path.basename(req.params.filename);
  const filePath = path.join(MODEL_DIR, filename);
  if (!fs.existsSync(filePath)) {
    res.status(404).json({ error: 'File not found' });
    return;
  }
  res.download(filePath, filename);
});

app.post('/api/predict', (req, res) => {
  const { model = 'fusion', sampleId, caption = '', topn = 10, imageBase64 } = req.body;
  const allowedModels = ['cnn', 'resnet50', 'vit', 'fusion'];
  const selectedModel = allowedModels.includes(model) ? model : 'fusion';

  let imagePath = '';
  let tempFile = '';

  if (imageBase64 && typeof imageBase64 === 'string') {
    const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, '');
    tempFile = `/tmp/upload_${Date.now()}.jpg`;
    fs.writeFileSync(tempFile, Buffer.from(base64Data, 'base64'));
    imagePath = tempFile;
  } else if (sampleId) {
    const candidate = path.join(MODEL_DIR, 'samples', `${path.basename(sampleId)}.jpg`);
    if (fs.existsSync(candidate)) {
      imagePath = candidate;
    }
  }

  if (!imagePath) {
    // Fallback to first sample
    const samplesDir = path.join(MODEL_DIR, 'samples');
    if (fs.existsSync(samplesDir)) {
      const imgs = fs.readdirSync(samplesDir).filter((f) => f.endsWith('.jpg'));
      if (imgs.length > 0) imagePath = path.join(samplesDir, imgs[0]);
    }
  }

  if (!imagePath || !fs.existsSync(imagePath)) {
    res.status(400).json({ error: 'No valid image found for inference' });
    return;
  }

  const predictScript = path.resolve(process.cwd(), 'predict.py');
  const args = [
    predictScript,
    '--model_dir',
    MODEL_DIR,
    '--model',
    selectedModel,
    '--image',
    imagePath,
    '--caption',
    caption || '',
    '--topn',
    String(topn || 10),
  ];

  const t0 = Date.now();
  execFile('python3', args, { timeout: 60000 }, (error, stdout, stderr) => {
    if (tempFile && fs.existsSync(tempFile)) {
      try {
        fs.unlinkSync(tempFile);
      } catch {
        // ignore
      }
    }

    if (error) {
      res.status(500).json({ error: stderr || error.message });
      return;
    }

    const lines = stdout.trim().split('\n');
    const predictions: Array<{ probability: number; cui: string; name: string }> = [];
    for (const line of lines) {
      const m = line.match(/^\s*([0-9.]+)\s+(C\d+)\s*(.*)$/);
      if (m) {
        predictions.push({
          probability: Number(m[1]),
          cui: m[2],
          name: m[3].trim() || m[2],
        });
      }
    }

    res.json({
      model: selectedModel,
      latencyMs: Date.now() - t0,
      rawOutput: stdout,
      command: `python predict.py --model_dir rocov2_model --model ${selectedModel} --image ${path.basename(imagePath)}${selectedModel === 'fusion' && caption ? ` --caption "${caption.slice(0, 60)}..."` : ''} --topn ${topn}`,
      predictions,
    });
  });
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
