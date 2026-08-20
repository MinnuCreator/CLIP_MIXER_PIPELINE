# Project audit

Date: 2026-08-21

This workspace was intentionally empty at the start of Phase 4. There is no Phase 3 code, database, API, frontend, test suite, or existing project convention to preserve. FrameForge is therefore bootstrapped as a modular Python application with a standard-library HTTP server and SQLite persistence. The service boundaries deliberately permit replacing the transport layer with an existing Phase 3 API later without changing transcription, alignment, timeline, or rendering logic.

Baseline: Python 3.14.7 present; Node.js and FFmpeg absent; no third-party Python packages installed.
