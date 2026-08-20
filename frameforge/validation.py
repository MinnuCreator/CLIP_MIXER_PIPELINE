from pathlib import Path
from .models import ImageAsset, Timeline


class TimelineValidationError(ValueError): pass


def validate_timeline(timeline: Timeline, images: list[ImageAsset], verify_files: bool = True, tolerance: float = 1e-5) -> Timeline:
    if timeline.audio_duration <= 0 or not timeline.scenes:
        raise TimelineValidationError("Timeline requires positive audio duration and at least one scene.")
    if verify_files and not Path(timeline.audio_file).is_file():
        raise TimelineValidationError("Audio file is missing.")
    image_map = {image.id: image for image in images}
    scenes = timeline.scenes
    # Exact authoritative endpoints; only sub-millisecond float noise is
    # normalized below. Meaningful gaps and overlaps must reach reconciliation.
    scenes[0].start, scenes[-1].end = 0.0, timeline.audio_duration
    for index, scene in enumerate(scenes):
        if scene.image_id not in image_map:
            raise TimelineValidationError(f"Unknown image reference: {scene.image_id}")
        if verify_files and not Path(image_map[scene.image_id].file_path).is_file():
            raise TimelineValidationError(f"Image file is missing: {image_map[scene.image_id].filename}")
        if scene.start < -tolerance or scene.end > timeline.audio_duration + tolerance:
            raise TimelineValidationError("Scene timestamp falls outside the audio duration.")
        if scene.end - scene.start <= tolerance: raise TimelineValidationError("All scene durations must be positive.")
        if index:
            previous = scenes[index - 1]
            if abs(scene.start - previous.end) <= tolerance: scene.start = previous.end
            else: raise TimelineValidationError("Timeline contains a gap or overlap.")
    return timeline
