"""FrameForge V3 HTTP API and Web Server with Async Processing & Prompt File Ingestion."""
from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import DEFAULT_CONFIG, FrameForgeConfig
from .prompt_parser import PromptMappingParser
from .service import FrameForgeService
from .storage import TimelineStore

logger = logging.getLogger("frameforge.server")


def create_service(database_path: str | Path | None = None) -> FrameForgeService:
    """Factory helper creating a FrameForgeService instance with the given database path."""
    if database_path is not None:
        store = TimelineStore(database_path)
        cfg = FrameForgeConfig(db_path=Path(database_path))
        return FrameForgeService(store=store, config=cfg)
    return FrameForgeService()


class FrameForgeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, config: FrameForgeConfig | None = None, **kwargs):
        self.config = config or DEFAULT_CONFIG
        self.service = FrameForgeService(config=self.config)
        super().__init__(*args, directory=str(self.config.static_dir.resolve()), **kwargs)

    def _send_json(self, status: int, payload: dict | list):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _parse_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Health / diagnostics endpoint
        if path == "/api/health":
            try:
                vis = self.service.visual_status()
                # V5: add LLM status
                from .llm import GeminiBeatPlanner
                planner = GeminiBeatPlanner(self.config)
                vis["llm_status"] = {
                    "model": self.config.llm_model,
                    "configured": planner.is_available,
                    "api_key_set": bool(self.config.llm_api_key),
                    "enabled": self.config.llm_enabled,
                }
                # full_multimodal_ready reflects CLIP status only (LLM is optional)
                # Do NOT downgrade readiness when LLM is disabled.
                self._send_json(200, vis)
            except Exception as exc:
                self._send_json(200, {"status": "error", "error": str(exc)})
            return

        # Asynchronous job status endpoint
        if path.startswith("/api/jobs/"):
            job_id = path.split("/")[-1]
            job = self.service.job_manager.get_job(job_id)
            if not job:
                self._send_json(404, {"error": "Job not found"})
                return
            self._send_json(200, job.as_dict())
            return

        # Fetch latest timeline for project
        if path.startswith("/timeline/"):
            project_id = path.split("/")[-1]
            record = self.service.store.get_latest(project_id)
            if not record:
                self._send_json(404, {"error": "No timeline found for project."})
                return
            self._send_json(200, record["timeline"])
            return

        # Serve static renders
        if path.startswith("/renders/"):
            rel_path = path.replace("/renders/", "")
            file_path = self.config.renders_dir / rel_path
            if file_path.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(file_path.stat().st_size))
                self.end_headers()
                with open(file_path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return

        # Default static file serving
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # 1. Parse prompt file preview endpoint
        if path == "/api/prompts/parse":
            try:
                body = self._parse_json_body()
                raw_text = body.get("content", "")
                if raw_text.strip().startswith("{") or raw_text.strip().startswith("["):
                    mapping = PromptMappingParser.parse_json(raw_text)
                else:
                    mapping = PromptMappingParser.parse_csv(raw_text)

                self._send_json(200, {"status": "ok", "total_mapped": len(mapping), "sample": dict(list(mapping.items())[:10])})
            except Exception as exc:
                self._send_json(400, {"error": f"Failed to parse prompt file: {exc}"})
            return

        # 2. Upload file endpoint (chunked streaming)
        if path == "/upload":
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))

            if "multipart/form-data" in content_type:
                boundary = content_type.split("boundary=")[-1].strip().encode()
                data = self.rfile.read(length)
                parts = data.split(b"--" + boundary)

                saved_files = []
                for part in parts:
                    if b"Content-Disposition" in part and b"filename=" in part:
                        headers_part, file_body = part.split(b"\r\n\r\n", 1)
                        file_body = file_body.rstrip(b"\r\n")
                        disposition = headers_part.decode("utf-8", errors="replace")
                        for line in disposition.split("\r\n"):
                            if "filename=" in line:
                                fn = line.split("filename=")[-1].strip('"\r\n ')
                                clean_fn = Path(fn).name
                                self.config.uploads_dir.mkdir(parents=True, exist_ok=True)
                                out_file = self.config.uploads_dir / clean_fn
                                out_file.write_bytes(file_body)
                                saved_files.append(
                                    {
                                        "filename": clean_fn,
                                        "file_path": str(out_file.resolve()),
                                        "size": len(file_body),
                                    }
                                )

                self._send_json(200, {"uploaded": saved_files})
                return

        # 3. Asynchronous timeline analysis
        if path == "/analyze-async":
            try:
                body = self._parse_json_body()
                project_id = body.get("project_id", str(uuid.uuid4())[:8])
                audio_file = body["audio_file"]
                audio_dur = body.get("audio_duration")
                images = body["images"]
                transcript = body.get("transcript")
                prompts = body.get("prompts")

                llm_enabled = body.get("llm_enabled")

                job_id = self.service.analyze_async(
                    project_id,
                    audio_file,
                    audio_dur,
                    images,
                    transcript=transcript,
                    prompts=prompts,
                    llm_enabled=llm_enabled,
                )
                self._send_json(202, {"job_id": job_id, "project_id": project_id})
            except Exception as exc:
                logger.error("Analyze-async request failed: %s", exc)
                self._send_json(400, {"error": str(exc)})
            return

        # 4. Synchronous timeline analysis
        if path == "/analyze":
            try:
                body = self._parse_json_body()
                project_id = body.get("project_id", str(uuid.uuid4())[:8])
                audio_file = body["audio_file"]
                audio_dur = body.get("audio_duration")
                images = body["images"]
                transcript = body.get("transcript")
                prompts = body.get("prompts")

                llm_enabled = body.get("llm_enabled")

                result = self.service.analyze(
                    project_id,
                    audio_file,
                    audio_dur,
                    images,
                    transcript=transcript,
                    prompts=prompts,
                    llm_enabled=llm_enabled,
                )
                self._send_json(200, result)
            except Exception as exc:
                logger.error("Analyze request failed: %s", exc)
                self._send_json(400, {"error": str(exc)})
            return

        # 5. Asynchronous video rendering
        if path == "/render-async":
            try:
                body = self._parse_json_body()
                project_id = body["project_id"]
                preset = body.get("preset", "vertical")
                fit = body.get("fit", "cover")
                transition = body.get("transition", "cut")
                render_profile = body.get("render_profile", "balanced")
                render_path = body.get("render_path")
                output_name = f"{project_id}_{preset}_{int(uuid.uuid1().time)}.mp4"
                self.config.renders_dir.mkdir(parents=True, exist_ok=True)
                output_path = str((self.config.renders_dir / output_name).resolve())

                job_id = self.service.render_async(
                    project_id,
                    output_path,
                    preset=preset,
                    fit=fit,
                    transition=transition,
                    render_path=render_path,
                    render_profile=render_profile,
                )
                self._send_json(202, {"job_id": job_id, "output_url": f"/renders/{output_name}"})
            except Exception as exc:
                logger.error("Render-async request failed: %s", exc)
                self._send_json(400, {"error": str(exc)})
            return

        # 6. Synchronous video rendering
        if path == "/render":
            try:
                body = self._parse_json_body()
                project_id = body["project_id"]
                preset = body.get("preset", "vertical")
                fit = body.get("fit", "cover")
                transition = body.get("transition", "cut")
                render_profile = body.get("render_profile", "balanced")
                output_name = f"{project_id}_{preset}_{int(uuid.uuid1().time)}.mp4"
                self.config.renders_dir.mkdir(parents=True, exist_ok=True)
                output_path = str((self.config.renders_dir / output_name).resolve())

                result = self.service.render(
                    project_id,
                    output_path,
                    preset=preset,
                    fit=fit,
                    transition=transition,
                    render_profile=render_profile,
                )
                result["output_url"] = f"/renders/{output_name}"
                self._send_json(200, result)
            except Exception as exc:
                logger.error("Render request failed: %s", exc)
                self._send_json(400, {"error": str(exc)})
            return

        # 7. Update timeline manually
        if path.startswith("/timeline/"):
            try:
                project_id = path.split("/")[-1]
                body = self._parse_json_body()
                self.service.update(project_id, body)
                self._send_json(200, {"status": "updated"})
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})
            return

        self._send_json(404, {"error": "Endpoint not found"})


def run_server(host: str = "127.0.0.1", port: int = 8000, config: FrameForgeConfig | None = None):
    cfg = config or DEFAULT_CONFIG
    cfg.uploads_dir.mkdir(parents=True, exist_ok=True)
    cfg.renders_dir.mkdir(parents=True, exist_ok=True)

    def handler_factory(*args, **kwargs):
        return FrameForgeHandler(*args, config=cfg, **kwargs)

    server = ThreadingHTTPServer((host, port), handler_factory)
    logger.info("FrameForge V3 Server running on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down FrameForge server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_server()
