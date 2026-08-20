import tempfile
import unittest
from pathlib import Path
from frameforge.alignment import SceneAlignmentEngine
from frameforge.models import ImageAsset, Timeline, TranscriptSegment, Word
from frameforge.validation import TimelineValidationError, validate_timeline
from frameforge.service import FrameForgeService


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.images = [ImageAsset("ship", "ship.png", "ship.png", "Titanic ship leaving Southampton"), ImageAsset("iceberg", "ice.png", "ice.png", "Titanic striking an iceberg at night")]
        self.segments = [TranscriptSegment("The Titanic departed Southampton.", 0, 3, [Word("The",0,.2), Word("Titanic",.2,.8), Word("departed",.8,1.2), Word("Southampton.",1.2,3)]), TranscriptSegment("Four days later it struck an iceberg at night.", 3, 7, [Word("Four",3,3.2), Word("iceberg",5,5.5)])]

    def test_alignment_uses_meaningful_timestamps(self):
        scenes, method = SceneAlignmentEngine().align(self.segments, self.images, 7)
        self.assertEqual((scenes[0].start, scenes[-1].end), (0.0, 7))
        self.assertEqual(scenes[0].image_id, "ship")
        self.assertEqual(scenes[-1].image_id, "iceberg")
        self.assertIn(method, {"semantic", "fallback"})

    def test_single_image_can_be_reused_as_one_continuous_scene(self):
        scenes, _ = SceneAlignmentEngine().align(self.segments, self.images[:1], 7)
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].duration, 7)

    def test_validation_normalizes_small_float_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio, image = Path(tmp)/"a.wav", Path(tmp)/"i.png"; audio.touch(); image.touch()
            assets=[ImageAsset("a", str(image), "i.png", "anything")]
            from frameforge.models import Scene
            line=Timeline(str(audio), 3, [Scene("1","a",0,1,"a",.5), Scene("2","a",1.000001,3,"b",.5)], "fallback")
            self.assertEqual(validate_timeline(line, assets).scenes[1].start, 1)

    def test_validation_rejects_gap(self):
        from frameforge.models import Scene
        line=Timeline("a.wav", 3, [Scene("1","a",0,1,"a",.5), Scene("2","a",2,3,"b",.5)], "fallback")
        with self.assertRaises(TimelineValidationError): validate_timeline(line, self.images, verify_files=False)

    def test_validation_rejects_material_overlap(self):
        from frameforge.models import Scene
        line = Timeline("a.wav", 3, [Scene("1", "ship", 0, 2, "a", .5), Scene("2", "iceberg", 1.5, 3, "b", .5)], "fallback")
        with self.assertRaises(TimelineValidationError):
            validate_timeline(line, self.images, verify_files=False)

    def _align(self, segments, duration):
        return SceneAlignmentEngine(minimum_scene_duration=.01).align(segments, self.images, duration)[0]

    def test_gap_between_segments_is_assigned_to_previous_visual(self):
        segments = [TranscriptSegment("Titanic departed Southampton.", 0, 3), TranscriptSegment("It struck an iceberg.", 4, 7)]
        scenes = self._align(segments, 7)
        self.assertEqual([(x.start, x.end) for x in scenes], [(0.0, 4), (4, 7)])

    def test_overlapping_segments_use_deterministic_next_start_boundary(self):
        segments = [TranscriptSegment("Titanic departed Southampton.", 0, 5), TranscriptSegment("It struck an iceberg.", 4, 7)]
        scenes = self._align(segments, 7)
        self.assertEqual([(x.start, x.end) for x in scenes], [(0.0, 4), (4, 7)])

    def test_audio_longer_than_narration_is_covered_by_last_selected_image(self):
        scenes = self._align(self.segments, 10)
        self.assertEqual(scenes[-1].end, 10)
        self.assertEqual(scenes[0].start, 0)

    def test_audio_shorter_than_timestamp_candidates_is_clamped_and_continuous(self):
        segments = [TranscriptSegment("Titanic departed Southampton.", 0, 4), TranscriptSegment("It struck an iceberg.", 4, 9)]
        scenes = self._align(segments, 5)
        self.assertEqual([(x.start, x.end) for x in scenes], [(0.0, 4), (4, 5)])

    def test_exactly_continuous_timeline_validates_and_ends_at_audio_duration(self):
        from frameforge.models import Scene
        line = Timeline("a.wav", 6, [Scene("1", "ship", 0, 2, "a", .5), Scene("2", "iceberg", 2, 6, "b", .5)], "fallback")
        valid = validate_timeline(line, self.images, verify_files=False)
        self.assertEqual(valid.scenes[-1].end, valid.audio_duration)

    def test_positive_measured_duration_reaches_timeline_generation(self):
        class Store:
            def save(self, *values): self.values = values
        store = Store()
        result = FrameForgeService(store).analyze("project", "voice.mp3", 7.25, [x.__dict__ for x in self.images], self.segments)
        self.assertEqual(result["audio"]["duration"], 7.25)
        self.assertEqual(result["scenes"][0]["start"], 0.0)
        self.assertEqual(result["scenes"][-1]["end"], 7.25)

if __name__ == "__main__": unittest.main()
