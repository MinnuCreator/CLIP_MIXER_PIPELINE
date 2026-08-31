# FrameForge — Three-Way Multimodal AI Video Generation Pipeline (V3)

FrameForge is an AI video generation system that synchronizes voiceovers with up to **700 images** and optional **prompt descriptions** to produce synchronized MP4 videos.

---

## 🌟 Key Capabilities in V3

### 1. Three-Way Multimodal Alignment (Pixels + Prompts + Voiceover)
- **Voiceover $\leftrightarrow$ Image Pixels ($S_{vi}$):** Joint visual-text CLIP embeddings (`clip-ViT-B-32`) mapping narration text to image pixel semantics.
- **Voiceover $\leftrightarrow$ Image Prompt ($S_{vp}$):** Dense semantic embeddings (`all-MiniLM-L6-v2`) scoring narration relevance against image descriptions.
- **Prompt $\leftrightarrow$ Image Consistency ($S_{pi}$):** Semantic validation between image prompts and visual pixel features, penalizing hallucinations or mismatched labels.
- **Normalized Linear Combination:**
  $$\text{Score} = w_{vi} S_{vi} + w_{vp} S_{vp} + w_{pi} S_{pi} + w_{ent} S_{ent} + w_{q} (Q - 0.5)$$

### 2. Flexible Prompt Mapping (JSON / CSV / Auto-Captioned)
- **JSON File:** `{"IMG_0001.jpg": "stormy ocean with huge waves", ...}`
- **CSV File:** `filename,prompt` or headerless CSV.
- **Zero-Prompt Fallback:** Images without prompts are automatically analyzed for visual concepts, dominant lighting, and color profiles (`["ocean", "storm", "night"]`, `["golden", "sunrise"]`).

### 3. Global Sequence Optimizer (DP / Viterbi Path Search)
- Solves global sequence lattice over Top-$K$ candidate visuals, balancing semantic relevance, topic continuity ($+0.08$), and duplicate avoidance ($-0.20$).

### 4. Dual-Path FFmpeg Video Renderer
- **FAST PATH:** Concat Demuxer (`concat.txt`) with pre-scaled normalized images in $O(1)$ CLI length, rendering 700 scenes in seconds.
- **TRANSITION PATH:** Staged filter graph for crossfades/fades.
- **Exact Duration & Sync:** Strict `ffprobe` verification with $\le 0.05\text{s}$ tolerance.

---

## 📊 Quality & Performance Benchmarks

### Accuracy Comparison (Voice→Image vs Voice→Image+Prompt)
| Mode | Top-1 Accuracy | Top-5 Recall | Top-10 Recall | Description |
|---|---|---|---|---|
| **Version A (Voice → Image Only)** | $10.0\%$ | $100.0\%$ | $100.0\%$ | Pixel features only with arbitrary filenames (`IMG_0001.jpg`) |
| **Version B (Voice → Image + Prompt V3)** | **$70.0\%$** | **$100.0\%$** | **$100.0\%$** | **$7.0\times$ accuracy improvement** using 3-Way Multimodal Alignment |

### Scalability Benchmarks (700 Images / 80 Scenes)
- **Cold Run:** $\approx 5.0\text{s}$
- **Warm Cached Run:** $\approx 1.2\text{s}$ ($4.1\times$ speedup)
- **FFmpeg Concat Render:** $\approx 8.1\text{s}$ for 48s 1080x1920 video ($6\times$ faster than realtime)

---

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Start the Local Server
```powershell
python -m frameforge.server
```
Open **`http://127.0.0.1:8000`** in your browser.

### 3. Run the Full Test Suite
```powershell
python -m unittest discover -s tests -v
```
