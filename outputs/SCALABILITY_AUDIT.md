# FrameForge scalability audit (Phase 0)

Date: 2026-08-21

## Current architecture

FrameForge is a local Python application using `ThreadingHTTPServer`, a static browser UI, SQLite (`frameforge.db`), `faster-whisper`, `sentence-transformers`, and FFmpeg/FFprobe. Uploads are written to `uploads/`. A successful analysis persists the audio, transcript, timeline, and submitted image/prompt associations as one JSON timeline payload in `sync_jobs`. Rendering rehydrates that validated payload, renders a single FFmpeg filter graph, verifies it with FFprobe, and serves it from `static/renders/`.

The working synchronization path remains narration-driven: Whisper word/segment timestamps are semantically segmented, prompt/narration similarity selects images, reconciliation makes scene boundaries continuous, and validation enforces the exact audio endpoints. Existing tests: 15 passing.

## Actual bottlenecks and failure points

| Area | Current behavior | 25-minute / 500-image risk | Minimum change |
|---|---|---|---|
| Whisper | A `WhisperModel` is created for each uncached request. Audio SHA-256 uses `Path.read_bytes()`. | Repeated model load and full audio memory copy; request blocks browser/server thread. | Process-wide transcriber/model service; streaming hash; persist transcript metadata. |
| Transcript cache | JSON cache is keyed by audio hash, but the cache file is loaded fully and old entries require FFprobe. | Cache grows unmanaged; cache lookup may still read a large audio file into RAM. | Canonical `data/transcripts/` store, streaming hash, cache metadata/version. |
| Embeddings | `LocalSimilarity` constructs `SentenceTransformer` per alignment and computes all segment/prompt vectors on each request. | Model reload, duplicate prompts recomputed, Python dot-product matrix is O(segments × images × dimensions). | Singleton embedding service, hash-keyed vector cache, batched NumPy/model similarity, top-K candidates. |
| Semantic selection | Full score matrix is built even when most of 500 images are irrelevant. | Time and matrix memory grow linearly with all 500 prompts; no AUTO_SELECT / USE_ALL_IMAGES policy. | Batch matrix multiplication and configurable top-K/AUTO_SELECT policy. |
| Segmentation | `SemanticSegmenter.split()` calls `raw.index(source)` during iteration. | Avoidable O(n²) behavior for long narration. | Iterate with indices. |
| Timeline | In-memory scene list, reconciliation and validation are linear after scoring. | Reasonable for 500 images, but quality controls are not exposed. | Add configuration for semantic threshold and visual-change rate while retaining current reconciliation. |
| SQLite | One shared locked connection; JSON timeline is stored in one row; latest job orders only by timestamp. | Long JSON payloads, no job progress/history indexes, second-level timestamp ties. | Job table/state columns and indexes; separate transcript/embedding/cache records. |
| Uploads | Request body is read fully into memory; filenames can collide; no file limits or type validation. | 500 uploads can spike RAM and overwrite an existing file. | Unique server filenames, streamed multipart/chunk upload, limits and MIME/content checks. |
| Browser UI | Creates one card per image and uploads in a single `Promise.all`. | 500 cards/files can freeze UI and overload server memory/connections. | Pagination/virtual list, thumbnail/lazy preview, bounded upload concurrency, actual upload/job progress. |
| Rendering | A single FFmpeg graph is already used; source images stay file-based. | 500 `-loop` inputs and a huge concat graph may hit Windows command/filter limits; all scenes are scaled/re-encoded. | Generate filter script file, use staged concat only if required, bounded render job, preserve exact timeline durations. |
| Rendering lifecycle | Rendering occurs synchronously in HTTP request. | Long render causes timeout/disconnect and no resumability. | Persisted background job state, worker queue, status endpoint, retry from validated timeline. |
| Storage/layout | `uploads/`, `.frameforge-cache/`, `static/renders/` are present. | No defined quota/cleanup/temp/job isolation. | Introduce `data/{uploads,transcripts,embeddings,timelines,renders,temp}` incrementally with migration-safe references. |

## Memory and CPU profile

CPU-heavy work is local transcription, embedding inference, full similarity scoring, image scaling/encoding, and FFprobe/FFmpeg execution. Memory-heavy work is currently full audio hashing, uploaded request bodies, 500 browser cards, cached JSON parsing, and the similarity matrix. FFmpeg itself reads image files lazily rather than keeping all full-resolution source images in Python memory, which is a good baseline.

There are no paid network inference calls. SentenceTransformer may download model weights on first local setup; subsequent inference is local. The browser/server still uses local HTTP for upload and playback.

## Recommended minimum implementation order

1. **Phase 1:** reusable batched embedding service with deterministic prompt cache and duplicate detection.
2. **Phase 2:** vectorized top-K semantic matching with AUTO_SELECT / USE_ALL_IMAGES configuration.
3. **Phase 3:** streaming audio hash and persistent transcript cache metadata.
4. **Phases 4–5:** preserve timeline reconciliation; add configuration and large-input tests.
5. **Phases 6–9:** render filter-script support, persisted job states, worker/status API, resume from timeline.
6. **Phases 10–12:** storage layout, upload safeguards, responsive 500-image UI.

## Baseline and constraints

The repository contains the successful 48.04-second project data and rendered MP4. Baseline automated tests pass (15/15), but no 25-minute/500-image benchmark has been executed. FFmpeg/model availability is environment-dependent, so real large rendering must be marked blocked unless run on a machine with those dependencies, sufficient disk, and adequate CPU/RAM.
