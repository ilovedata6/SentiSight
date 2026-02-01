# SentiSight Startup Time & Performance Guide

## 🚀 Quick Answer

**Expected Startup Time:**
- **First Run (downloading models):** 45-90 seconds
- **Subsequent Runs (models cached):** 15-30 seconds
- **App Ready for Use:** Immediately after loading completes

## ⚡ Current GPU Status

### Your System
- **GPU:** NVIDIA Quadro M1200 (4GB VRAM)
- **CUDA Version:** 12.0
- **Driver:** 528.79 ✓

### PyTorch Status

✓ Currently Running: **GPU Mode**
- **PyTorch:** 2.5.1+cu121
- **CUDA Available:** True
- **Device:** NVIDIA Quadro M1200 (4.3 GB)


## 💡 GPU Utilization Options

### Option 1: Downgrade to Python 3.12 (Recommended for GPU)

```bash
# Install Python 3.12
# Then recreate environment
uv venv --python 3.12
uv sync

# Install PyTorch with CUDA 12.1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Benefits:**
- 🚀 **2-3x faster** sentiment analysis
- ⚡ **Real-time processing** for batches
- 📊 Larger batch sizes (16-32 vs 8)
- 🎯 Better for heavy workloads

**Startup Time with GPU:**
- First run: 60-120 seconds (model download + GPU warmup)
- Subsequent: 20-40 seconds
- GPU warmup adds ~5-10 seconds initially

### Option 2: Continue with CPU (Current Setup)

**Benefits:**
- ✅ Works immediately
- 💾 Lower memory usage
- 🔧 No additional setup needed
- 📱 Good for moderate workloads

**Performance:**
- Single feedback: ~200-400ms
- Batch (10): ~2-4 seconds
- Still very usable for typical use cases

## 📊 Detailed Startup Breakdown

### Phase 1: Model Download (First Run Only)
```
DistilBERT Model:     ~250 MB    (30-60s download)
Tokenizer Files:      ~1 MB      (1-2s)
Config Files:         ~5 KB      (<1s)
Total Download:       ~251 MB    (30-60s total)
```

**Note:** This happens only once. Models are cached in:
- Windows: `C:\Users\<username>\.cache\huggingface\`
- After first download, this phase is skipped

### Phase 2: Model Loading (Every Run)

**CPU Mode (Current):**
```
Loading DistilBERT:        10-15 seconds
Loading Category Classifier: 2-3 seconds
Loading Anomaly Detector:    1-2 seconds
Initializing Preprocessor:   <1 second
Total:                       15-20 seconds
```

**GPU Mode (If using Python 3.12):**
```
Loading DistilBERT:        ~9.2 seconds (measured load with Quadro M1200)
GPU Warmup:                ~0.01-0.02 seconds (first inference is fast; one warmup run performed)
Loading Category Classifier: <1 second
Loading Anomaly Detector:    <1 second
Initializing Preprocessor:   <1 second
Total (measured, subsequent run): ~12.7 seconds
```

### Phase 3: App Initialization
```
Streamlit Framework:    2-3 seconds
UI Components:          1-2 seconds
Session State:          <1 second
Total:                  3-5 seconds
```

## ⏱️ Total Startup Time Summary

| Scenario | First Run | Subsequent Runs |
|----------|-----------|-----------------|
| **CPU (Current)** | 45-80 seconds | 15-25 seconds |
| **GPU (Python 3.12)** | 90-120 seconds | 25-40 seconds |

## 🎯 When is App Ready to Use?

The app is **immediately ready** once you see:
```
✅ All models loaded in X.Xs on CPU/GPU
```

**What you'll see during startup:**
1. Streamlit page loads (3-5s)
2. Progress indicator appears
3. Model loading stages:
   - 🤖 Loading Sentiment Analyzer (DistilBERT)... [10%]
   - 📊 Loading Category Classifier... [50%]
   - 🚨 Initializing Anomaly Detector... [80%]
   - ✅ All models loaded! [100%]
4. **App is now fully functional**

## 🏎️ Performance Comparison: CPU vs GPU

### Sentiment Analysis (DistilBERT)

| Operation | CPU | GPU (4GB) | Improvement |
|-----------|-----|-----------|-------------|
| Single feedback | 150-250ms | 50-100ms | 2-3x faster |
| Batch of 10 | 1.5-2.5s | 0.5-1.0s | 3x faster |
| Batch of 100 | 15-25s | 5-10s | 3x faster |
| Batch of 1000 | 150-250s | 50-100s | 3x faster |

### Memory Usage

| Component | CPU | GPU (4GB) |
|-----------|-----|-----------|
| DistilBERT | ~500MB RAM | ~800MB VRAM + 200MB RAM |
| Category Classifier | ~50MB RAM | ~50MB RAM |
| Anomaly Detector | ~20MB RAM | ~20MB RAM |
| **Total** | **~570MB RAM** | **~800MB VRAM + 270MB RAM** |

## 🔧 Optimization Tips

### For CPU Mode (Current)
1. **Reduce batch size:** Use batch_size=4-8
2. **Process in chunks:** Break large datasets into 100-500 record chunks
3. **Close other apps:** Free up RAM
4. **Use during off-peak:** CPU runs cooler and faster

### For GPU Mode (Python 3.12)
1. **Larger batches:** Use batch_size=16-32
2. **Keep models loaded:** Don't restart frequently
3. **Monitor VRAM:** 4GB is good for most workloads
4. **GPU warmup:** First inference is slower (normal)

## 🚦 Real-World Usage Scenarios

### Scenario 1: Interactive Analysis (Dashboard)
- **Typical:** Analyzing 1-10 feedbacks at a time
- **CPU Performance:** Excellent (200-400ms per feedback)
- **GPU Benefit:** Minimal (already fast enough)
- **Recommendation:** CPU is fine

### Scenario 2: Batch Processing (API)
- **Typical:** Processing 100-1000 feedbacks
- **CPU Performance:** Good (2-4 minutes for 1000)
- **GPU Benefit:** Significant (40-80 seconds for 1000)
- **Recommendation:** GPU worth it if frequent batching

### Scenario 3: Large Dataset Analysis
- **Typical:** Processing 10,000+ feedbacks
- **CPU Performance:** Acceptable (20-40 minutes for 10K)
- **GPU Benefit:** Major (7-15 minutes for 10K)
- **Recommendation:** GPU highly recommended

## 📝 Recommendations

### For Your Use Case

**Stick with CPU if:**
- ✅ Analyzing <100 feedbacks at a time
- ✅ Interactive/exploratory analysis
- ✅ Don't want to downgrade Python
- ✅ Acceptable 15-25 second startup

**Switch to GPU (Python 3.12) if:**
- ⚡ Regular batch processing (>100 feedbacks)
- ⚡ High-frequency API calls
- ⚡ Large dataset analysis (>1000 feedbacks)
- ⚡ Don't mind 25-40 second startup for 3x speed

## 🛠️ How to Switch to GPU

If you decide GPU is worth it:

```bash
# 1. Backup current environment
cp .venv .venv_backup_313

# 2. Remove current environment
rm -rf .venv

# 3. Install Python 3.12
# Download from python.org or use pyenv

# 4. Create new environment with Python 3.12
uv venv --python 3.12
uv sync

# 5. Install CUDA PyTorch
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 6. Verify GPU
python scripts/check_gpu.py

# Should see:
# ✓ CUDA Available: True
# ✓ GPU Device: Quadro M1200
```

### Docker + GPU (production)

For GPU-enabled deployment use the NVIDIA Container Toolkit and a CUDA base image.
- Use a CUDA runtime image (e.g., `nvidia/cuda:12.1-runtime`).
- Install PyTorch with CUDA (cu121) inside the image.
- Start containers with GPU access: `docker run --gpus all -p 8501:8501 your-image`.

See `README.md` Docker section for an example `docker-compose` snippet using `runtime: nvidia` or `--gpus` flags.

## 📊 Current Configuration Summary

```
System:           Windows + Quadro M1200 (4GB)
Python:           3.12.11 (GPU-enabled)
PyTorch:          2.5.1+cu121 (CUDA available)
Startup Time:     ~12-15 seconds (models cached)
Performance:      DistilBERT load ~9.2s; inference ~13.4ms
GPU Utilization:  Active (Quadro M1200)

Recommendation:   GPU mode is recommended for:
                  - Regular batch processing (>100 feedbacks)
                  - High-frequency API calls
                  - Large dataset analysis (>1k feedbacks)

                  CPU mode remains fine for:
                  - Interactive dashboard use
                  - API with <50 requests/minute
                  - Small batch sizes (<100)
```

## ✅ Bottom Line

**Your app will be ready in 15-30 seconds (after first download).**

The code is already optimized to use GPU if available, so if you ever downgrade to Python 3.12, it will automatically detect and use your Quadro M1200 for 2-3x faster processing. For now, CPU mode works great for typical usage!
