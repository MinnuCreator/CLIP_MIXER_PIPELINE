from __future__ import annotations
import json, logging, mimetypes, os, shutil
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from .models import ImageAsset, Scene, Timeline, TranscriptSegment, Word
from .renderer import FFmpegRenderer, RenderError
from .service import FrameForgeService
from .storage import TimelineStore
from .transcript import TranscriptionError

ROOT = Path(__file__).parent.parent; UPLOADS = ROOT / "uploads"; STATIC = ROOT / "static"


def create_service(database_path: Path) -> FrameForgeService:
    return FrameForgeService(TimelineStore(database_path))


service = create_service(ROOT / "frameforge.db")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("frameforge.server")

class App(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs): super().__init__(*args, directory=str(STATIC), **kwargs)
    def json(self, data, status=200):
        body = json.dumps(data).encode(); self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def read_json(self): return json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode())
    def do_GET(self):
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3:] == ["video-sync", "timeline"]:
            result = service.store.get_latest(parts[2]); return self.json(result or {"error": "No timeline exists."}, 404 if not result else 200)
        return super().do_GET()
    def do_POST(self):
        parts = urlparse(self.path).path.strip("/").split("/")
        if parts == ["api", "uploads"]:
            name = Path(self.headers.get("X-Filename", "upload.bin")).name
            if not name or name == ".": return self.json({"error":"Missing upload filename."}, 400)
            UPLOADS.mkdir(exist_ok=True)
            destination = UPLOADS / name
            destination.write_bytes(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            return self.json({"file_path": str(destination.resolve()), "filename": name})
        if len(parts) != 5 or parts[:2] != ["api", "projects"] or parts[3] != "video-sync": return self.json({"error":"Unknown endpoint"}, 404)
        try:
            body = self.read_json(); project_id, action = parts[2], parts[4]
            if action == "analyze":
                transcript = [TranscriptSegment(x["text"], x["start"], x["end"], [Word(**w) for w in x.get("words",[])]) for x in body.get("transcript", [])] or None
                # The backend measures uploaded audio itself. Do not allow an
                # empty browser number input (which JavaScript converts to 0)
                # to invalidate a successful Whisper transcription.
                supplied_duration = float(body["audio_duration"]) if transcript and body.get("audio_duration") is not None else None
                return self.json(service.analyze(project_id, body["audio_file"], supplied_duration, body["images"], transcript))
            if action == "render":
                render_dir = STATIC / "renders"; render_dir.mkdir(exist_ok=True)
                output = render_dir / f"{project_id}-synced.mp4"
                result = service.render(project_id, str(output), body.get("preset", "vertical"), body.get("fit", "cover"), body.get("transition", "cut"), body.get("ffmpeg_path"))
                return self.json({"status":"rendered", "output": result["output"], "url": f"/renders/{output.name}", "metadata": result["metadata"]})
        except (ValueError, KeyError, RenderError, TranscriptionError) as exc: return self.json({"error":str(exc)}, 400)
        except Exception:
            logger.exception("Unhandled error in POST %s", self.path)
            return self.json({"error":"The request could not be processed. Check server logs."}, 500)
    def do_PATCH(self):
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3:] == ["video-sync", "timeline"]:
            try: service.update(parts[2], self.read_json()); return self.json({"status":"saved"})
            except Exception: return self.json({"error":"Timeline could not be saved."}, 400)
        self.json({"error":"Unknown endpoint"}, 404)

if __name__ == "__main__":
    print("FrameForge available at http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), App).serve_forever()
