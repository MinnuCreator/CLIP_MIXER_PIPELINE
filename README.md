# FrameForge

FrameForge is a local-first voice-over-to-image synchronization engine. It stores projects in SQLite, transcribes locally with `faster-whisper` when installed, aligns narration to image prompts with local sentence embeddings when available, and renders with local FFmpeg.

## Quick start

```powershell
python -m pip install -r requirements.txt
python -m frameforge.server
```

Open `http://127.0.0.1:8000`. FFmpeg must be installed and on `PATH` before rendering. No API keys or remote inference are used.

Enter the audio duration from your editor/player before analysis. This deliberate input avoids bundling or downloading a media-probing binary; FFmpeg is still required for the final render.

## Local model modes

- **semantic**: `sentence-transformers` is available; uses `all-MiniLM-L6-v2` locally (the initial model download is performed by the library if it is not already cached).
- **fallback**: deterministic token/phrase overlap; no model download.
- Transcription requires local `faster-whisper`; analysis accepts a saved/mock transcript for offline development and tests.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) and [PROJECT_AUDIT.md](PROJECT_AUDIT.md).
