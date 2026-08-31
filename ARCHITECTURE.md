# FrameForge V2 — Technical Architecture & Deep Dive

## 1. System Philosophy

FrameForge V2 is built around three core architectural pillars:
1. **Local-First & Zero Paid APIs:** All transcription, embedding generation, semantic similarity scoring, and rendering run entirely on the local machine with no external network dependencies or token costs.
2. **700-Image High-Throughput Scalability:** Designed from the ground up to process large asset sets without Windows CLI string limits, file descriptor exhaustion, or browser DOM freezes.
3. **Multi-Level Persistent Caching:** Every expensive operation (audio transcription, image metadata extraction, perceptual hashing, prompt vector embedding, and image aspect ratio normalization) is keyed by stable SHA-256 hashes and cached to disk.

---

## 2. End-to-End Pipeline Stages

```text
[Voiceover Audio]                      [700 Image Assets]
       │                                       │
       ▼                                       ▼
Streaming SHA-256 Hasher               Streaming SHA-256 + dHash
       │                                       │
       ▼                                       ▼
Transcript Cache Check                 Metadata & Quality Analysis
   ├─ Hit ──> Load JSONSegments            ├─ Hit ──> Load JSON Metadata
   └─ Miss ─> faster-whisper (int8)        └─ Miss ─> PIL Perceptual Metrics
       │                                       │
       ▼                                       ▼
Semantic Segmentation                  Singleton Embedding Engine
(Punctuation & Pause >= 0.7s)          (384-d all-MiniLM-L6-v2)
       │                                       │
       ▼                                       ▼
Narration Segments Matrix              In-Memory Vector Index
       │                                       │
       └──────────────┬────────────────────────┘
                      ▼
            Stage 1: Top-K Retrieval (NumPy Matrix Dot Product)
                      │
                      ▼
            Stage 2: Multi-Factor Reranking
            (Semantic + Entity + Quality + Continuity - Duplicates - Recency)
                      │
                      ▼
            Temporal Boundary Reconciliation & Short-Scene Merging
                      │
                      ▼
            Gap-Free Validated Timeline (0.0s -> T_audio)
                      │
                      ▼
            Pre-Scaled Image Normalization Cache (1080x1920 / 540x960)
                      │
                      ▼
            FFmpeg Concat Demuxer Engine (ffconcat protocol)
                      │
                      ▼
            FFprobe Codec & Duration Verification (<= 0.15s tolerance)
                      │
                      ▼
            Final MP4 Video Output
```

---

## 3. Mathematical Formulation of Two-Stage Matching

### Stage 1: Candidate Vector Retrieval
Given narration query vector $\mathbf{q} \in \mathbb{R}^{384}$ and $M \le 700$ image prompt vectors stored in matrix $\mathbf{V} \in \mathbb{R}^{M \times 384}$:
$$\mathbf{s}_{\text{sem}} = \mathbf{q} \mathbf{V}^T$$
Top-$K$ candidate indices $C_q = \{c_1, c_2, \dots, c_K\}$ are selected via `np.argpartition` in $O(M)$ time.

### Stage 2: Multi-Factor Reranking
For each candidate $j \in C_q$:
$$\text{Score}(q, j) = w_{\text{sem}} \cdot \mathbf{s}_{\text{sem}}[j] + w_{\text{ent}} \cdot E(q, j) + w_{\text{qual}} \cdot (Q_j - 0.5) + B_{\text{cont}} - P_{\text{dup}} - P_{\text{recent}}$$

Where:
- $w_{\text{sem}} = 0.55$: Dense semantic similarity weight.
- $w_{\text{ent}} = 0.15$: Salient entity / noun keyword token overlap $E(q, j) = \frac{|\text{Tokens}(q) \cap \text{Tokens}(j)|}{\max(1, |\text{Tokens}(q)|)}$.
- $w_{\text{qual}} = 0.10$: Image visual quality score $Q_j \in [0.1, 1.0]$ based on resolution, sharpness gradient, and luminance contrast.
- $B_{\text{cont}} = 0.08$: Topic continuity boost applied when candidate $j$ matches the previous scene's visual.
- $P_{\text{dup}} = 0.20$: Perceptual duplicate penalty applied when image $j$ has a near-identical dHash (Hamming distance $\le 4$) to an already-indexed image.
- $P_{\text{recent}} = 0.12 \cdot \left(1.0 - \frac{\text{recency}}{\text{window}}\right)$: Decaying recent-use penalty across a sliding window of the last $W=4$ scenes, preventing visual repetition when narration moves to new subjects.

---

## 4. Scalable Rendering: FFmpeg Concat Demuxer

### The Problem with Single-Command Filter Graphs
In standard architectures, rendering $N$ scenes generates $N$ `-loop 1 -t ... -i ...` CLI flags and a massive `filter_complex` string.
- At $N=700$, the CLI string exceeds 84,000 characters (crashing the Windows 32KB limit).
- FFmpeg keeps 700 active image decoders in RAM simultaneously, causing out-of-memory errors.

### The V2 Solution
1. **Parallel Background Normalization:** Images are pre-scaled and cropped into target resolutions (`.frameforge-cache/normalized/<hash>_1080x1920_cover.jpg`).
2. **Concat Demuxer Script (`concat.txt`):**
   ```text
   ffconcat version 1.0
   file 'C:/.../norm_001.jpg'
   duration 6.7800
   file 'C:/.../norm_002.jpg'
   duration 7.6200
   ...
   ```
3. **Constant $O(1)$ CLI Execution:**
   ```bash
   ffmpeg -y -f concat -safe 0 -i concat.txt -i voiceover.mp3 \
          -c:v libx264 -preset medium -crf 22 -pix_fmt yuv420p \
          -c:a aac -b:a 192k -shortest output.mp4
   ```
   FFmpeg streams image frames sequentially from disk with minimal memory overhead ($< 150\text{ MB}$ RAM).

---

## 5. Persistent Cache Layout

```text
.frameforge-cache/
├── transcripts/       # SHA-256 keyed JSON transcriptions & word timestamps
├── metadata/          # Perceptual hashes, dimensions & quality metrics
├── embeddings/        # JSON/NPY 384-dimensional prompt vectors
└── normalized/        # Render-ready pre-scaled JPEG visuals
```
