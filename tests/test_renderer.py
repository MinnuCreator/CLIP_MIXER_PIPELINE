import unittest
from frameforge.models import ImageAsset, Scene, Timeline
from frameforge.renderer import FFmpegRenderer


class RendererCommandTests(unittest.TestCase):
    def setUp(self):
        self.timeline = Timeline("voice.mp3", 5.5, [Scene("scene-001", "first", 0, 2.125, "first narration", .9), Scene("scene-002", "second", 2.125, 5.5, "second narration", .8)], "semantic")
        self.assets = [ImageAsset("first", "first.jpg", "first.jpg", "first"), ImageAsset("second", "second.jpg", "second.jpg", "second")]

    def test_command_uses_each_authoritative_scene_duration(self):
        command = FFmpegRenderer("ffmpeg").command(self.timeline, self.assets, "output.mp4")
        durations = [command[index + 1] for index, part in enumerate(command) if part == "-t"]
        self.assertEqual(durations, ["2.1250", "3.3750"])
        self.assertIn("concat=n=2:v=1:a=0[v]", command[command.index("-filter_complex") + 1])

    def test_command_maps_voiceover_and_h264_aac(self):
        command = FFmpegRenderer("ffmpeg").command(self.timeline, self.assets, "output.mp4", "vertical", "cover")
        self.assertIn("voice.mp3", command)
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertIn("scale=1080:1920", command[command.index("-filter_complex") + 1])
