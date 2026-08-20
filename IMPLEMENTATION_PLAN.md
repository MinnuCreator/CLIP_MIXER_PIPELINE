# FrameForge Phase 4 implementation plan

1. Establish a local-only modular architecture and SQLite timeline store. ✅
2. Implement local Whisper transcription with cached word timestamps and explicit dependency errors. ✅
3. Implement semantic transcript segmentation and prompt/image loading. ✅
4. Implement embedding alignment with deterministic offline fallback. ✅
5. Generate and validate a gap-free, editable audio-authoritative timeline. ✅
6. Build local FFmpeg rendering command generation, image fitting, cut/crossfade configuration, and motion effects. ✅
7. Expose project sync endpoints and a small browser UI for upload, preview, and timeline editing. ✅
8. Add offline unit tests and run them. ✅ (4 passed)
9. Run real end-to-end rendering once FFmpeg and a local Whisper model are installed. Blocked by baseline dependencies.
