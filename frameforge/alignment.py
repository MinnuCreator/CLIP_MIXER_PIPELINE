from __future__ import annotations

import math
import re

from .models import ImageAsset, Scene, TranscriptSegment


def _tokens(text: str) -> set[str]:
    return {x for x in re.findall(r"[a-z0-9]+", text.lower()) if len(x) > 1}


class LocalSimilarity:
    def __init__(self):
        self.method = "fallback"
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.method = "semantic"
        except Exception:
            pass

    def score_matrix(self, narration: list[str], prompts: list[str]) -> list[list[float]]:
        if self.model:
            vectors = self.model.encode(narration + prompts, normalize_embeddings=True)
            n = len(narration)
            return [[float(sum(a * b for a, b in zip(vectors[i], vectors[n + j]))) for j in range(len(prompts))] for i in range(n)]
        return [[self._fallback(a, b) for b in prompts] for a in narration]

    @staticmethod
    def _fallback(a: str, b: str) -> float:
        left, right = _tokens(a), _tokens(b)
        return len(left & right) / math.sqrt(len(left) * len(right)) if left and right else 0.0


class SceneAlignmentEngine:
    def __init__(self, minimum_scene_duration: float = 1.5, preferred_min_duration: float = 2.0, maximum_scene_duration: float = 12.0):
        self.minimum_scene_duration = minimum_scene_duration
        self.preferred_min_duration = preferred_min_duration
        self.maximum_scene_duration = maximum_scene_duration

    def align(self, segments: list[TranscriptSegment], images: list[ImageAsset], audio_duration: float) -> tuple[list[Scene], str]:
        if not images:
            raise ValueError("At least one image with a prompt is required.")
        if any(not x.prompt.strip() for x in images):
            raise ValueError("Every image requires a prompt.")
        similarity = LocalSimilarity()
        scores = similarity.score_matrix([s.text for s in segments], [i.prompt for i in images])
        choices: list[int] = []
        for row in scores:
            choices.append(max(range(len(images)), key=lambda j: row[j] + (0.06 if choices and j == choices[-1] else 0)))
        runs: list[tuple[int, int, int, float]] = []
        for index, image_index in enumerate(choices):
            if runs and runs[-1][0] == image_index:
                old = runs[-1]
                runs[-1] = (old[0], old[1], index, old[3] + scores[index][image_index])
            else:
                runs.append((image_index, index, index, scores[index][image_index]))
        scenes = []
        for number, (image_index, first, last, score) in enumerate(runs, 1):
            start = 0.0 if number == 1 else segments[first].start
            end = audio_duration if number == len(runs) else segments[last].end
            scenes.append(Scene(f"scene-{number:03d}", images[image_index].id, start, end, " ".join(s.text for s in segments[first:last + 1]), round(score / (last - first + 1), 4)))
        # Whisper segments may include pauses and may occasionally overlap.  The
        # semantic assignment above remains unchanged; this pass assigns every
        # unspoken interval to an adjacent selected image at a natural boundary.
        return self._merge_short(self._reconcile_boundaries(scenes, audio_duration)), similarity.method

    @staticmethod
    def _reconcile_boundaries(scenes: list[Scene], audio_duration: float) -> list[Scene]:
        """Make selected scenes a continuous partition of the actual audio.

        The next scene's Whisper start is the preferred boundary: a pause after
        a sentence stays with the preceding visual.  If timestamp candidates
        overlap or lie outside the audio, a midpoint is used deterministically.
        This is boundary reconciliation, not validation suppression.
        """
        if not scenes:
            return scenes
        epsilon = 1e-6
        for scene in scenes:
            scene.start = max(0.0, min(float(scene.start), audio_duration))
            scene.end = max(0.0, min(float(scene.end), audio_duration))
        scenes[0].start = 0.0
        scenes[-1].end = audio_duration
        index = 0
        while index < len(scenes) - 1:
            left, right = scenes[index], scenes[index + 1]
            lower = left.start + epsilon
            upper = right.end - epsilon
            preferred = right.start
            if upper <= lower:
                # No positive interval can be allocated to both selected
                # scenes after clamping.  Keep the earlier semantic choice and
                # absorb the impossible candidate into it.
                left.end = max(left.end, right.end)
                left.narration = (left.narration + " " + right.narration).strip()
                left.confidence = round((left.confidence + right.confidence) / 2, 4)
                scenes.pop(index + 1)
                continue
            if preferred < lower or preferred > upper:
                preferred = (left.end + right.start) / 2
            boundary = max(lower, min(preferred, upper))
            left.end = boundary
            right.start = boundary
            index += 1
        scenes[-1].end = audio_duration
        return scenes

    def _merge_short(self, scenes: list[Scene]) -> list[Scene]:
        index = 0
        while index < len(scenes):
            if len(scenes) > 1 and scenes[index].duration < self.minimum_scene_duration:
                target = index - 1 if index else 1
                left, right = sorted((index, target))
                keeper, removed = scenes[left], scenes[right]
                keeper.start, keeper.end = min(keeper.start, removed.start), max(keeper.end, removed.end)
                keeper.narration = (keeper.narration + " " + removed.narration).strip()
                scenes.pop(right)
                index = max(0, left - 1)
            else:
                index += 1
        for n, scene in enumerate(scenes, 1):
            scene.scene_id = f"scene-{n:03d}"
        return scenes
