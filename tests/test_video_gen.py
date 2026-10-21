import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))  # ComfyUI root, for comfy.model_management

import comfy.model_management as mm  # noqa: E402


def load_module():
    package_name = "comfy_openrouter_video_gen_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package
    modules = []
    for name in ("openrouter_shared", "node_video_gen"):
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


shared, node_module = load_module()
Node = node_module.OpenRouterVideoGenNode

SUBMITTED = {"id": "job-1", "status": "pending"}
PENDING = {"status": "in_progress"}
COMPLETED = {"status": "completed", "unsigned_urls": ["https://openrouter.ai/content/0"], "usage": {"USD": 0.5}}


def make_response(status=200, body=None, text="", content=b""):
    response = Mock()
    response.status_code = status
    response.json.return_value = body
    response.text = text
    response.content = content
    return response


def generate(**kwargs):
    args = dict(api_key="key", prompt="p", model="m", poll_interval=1, request_timeout=10)
    args.update(kwargs)
    return Node().generate_video(**args)


class VideoGenTests(unittest.TestCase):
    def setUp(self):
        patches = [
            patch.object(node_module.requests, "post", return_value=make_response(body=SUBMITTED)),
            patch.object(node_module.requests, "get"),
            patch.object(node_module.time, "sleep"),
            patch.object(shared, "fetch_credits", return_value="Remaining: $1.000"),
            patch.object(shared, "video_bytes_to_frames", return_value=(torch.zeros((2, 4, 4, 3)), {"fps": 24.0, "frame_count": 2})),
            patch.object(shared, "build_video_object", return_value="video"),
        ]
        self.post, self.get, self.sleep, *_ = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)

    def queue(self, *responses):
        self.get.side_effect = list(responses)

    def assert_resumable(self, regex):
        with self.assertRaisesRegex(RuntimeError, regex) as ctx:
            generate()
        self.assertIn("job-1", str(ctx.exception))
        self.assertIn("video_job_id", str(ctx.exception))

    def test_success_returns_video_and_stats(self):
        self.queue(make_response(body=PENDING), make_response(body=COMPLETED), make_response(content=b"mp4"))
        video, stats, credits = generate()
        self.assertEqual(video, "video")
        self.assertIn("Job ID: job-1", stats)
        self.assertIn("Cost: $0.500", stats)
        self.assertEqual(self.get.call_args_list[0].args[0], f"{shared.BASE_URL}/videos/job-1")
        self.assertEqual(self.get.call_args_list[2].args[0], "https://openrouter.ai/content/0")

    def test_missing_api_key_raises(self):
        with patch.object(shared, "get_api_key", return_value=""), self.assertRaisesRegex(RuntimeError, "API Key"):
            generate(api_key="")
        self.post.assert_not_called()

    def test_submit_http_error_raises_without_job_id(self):
        self.post.return_value = make_response(status=400, body={"error": {
            "message": "bad prompt", "metadata": {"provider_name": "Acme", "raw": "upstream said no"}}})
        with self.assertRaises(RuntimeError) as ctx:
            generate()
        message = str(ctx.exception)
        for part in ("400", "bad prompt", "Acme", "upstream said no"):
            self.assertIn(part, message)
        self.assertNotIn("video_job_id", message)

    def test_submit_without_job_id_raises(self):
        self.post.return_value = make_response(body={"status": "pending"})
        with self.assertRaisesRegex(RuntimeError, "no job ID"):
            generate()

    def test_submit_network_error_raises_without_job_id(self):
        self.post.side_effect = node_module.requests.ConnectionError("down")
        with self.assertRaises(node_module.requests.ConnectionError):
            generate()

    def test_frame_failure_raises_before_submit(self):
        with patch.object(shared, "image_to_base64", side_effect=ValueError("bad frame")), \
                self.assertRaisesRegex(ValueError, "bad frame"):
            generate(first_frame=torch.zeros((1, 4, 4, 3)))
        self.post.assert_not_called()

    def test_poll_http_error_carries_job_id(self):
        self.queue(make_response(status=500, body={"error": {"message": "upstream down"}}))
        self.assert_resumable("upstream down")

    def test_poll_network_error_carries_job_id(self):
        self.queue(node_module.requests.ConnectionError("reset"))
        self.assert_resumable("reset")

    def test_failed_job_keeps_provider_message(self):
        self.queue(make_response(body={"status": "failed", "error": "content policy violation"}))
        self.assert_resumable("failed.*content policy violation")

    def test_expired_job_raises(self):
        self.queue(make_response(body={"status": "expired"}))
        self.assert_resumable("expired")

    def test_timeout_carries_job_id(self):
        self.get.return_value = make_response(body=PENDING)
        self.assert_resumable("timed out")
        self.assertEqual(self.get.call_count, 10)

    def test_completed_without_url_raises(self):
        self.queue(make_response(body={"status": "completed", "unsigned_urls": []}))
        self.assert_resumable("no video URL")

    def test_download_http_error_carries_job_id(self):
        self.queue(make_response(body=COMPLETED), make_response(status=403, body={"error": {"message": "forbidden"}}))
        self.assert_resumable("forbidden")

    def test_undecodable_video_carries_job_id(self):
        self.queue(make_response(body=COMPLETED), make_response(content=b"mp4"))
        with patch.object(shared, "video_bytes_to_frames", side_effect=ValueError("bad codec")):
            self.assert_resumable("bad codec")

    def test_resume_skips_submit_and_media(self):
        self.queue(make_response(body=COMPLETED), make_response(content=b"mp4"))
        with patch.object(shared, "image_to_base64") as encode:
            video, stats, _ = generate(video_job_id=" job-9 ", first_frame=torch.zeros((1, 4, 4, 3)), prompt="")
        self.post.assert_not_called()
        encode.assert_not_called()
        self.assertEqual(video, "video")
        self.assertIn("Job ID: job-9", stats)
        self.assertEqual(self.get.call_args_list[0].args[0], f"{shared.BASE_URL}/videos/job-9")

    def test_resumed_job_failure_carries_the_same_job_id(self):
        self.queue(make_response(body={"status": "failed", "error": "boom"}))
        with self.assertRaisesRegex(RuntimeError, "job-9.*video_job_id"):
            generate(video_job_id="job-9")

    def test_interrupt_ends_polling(self):
        self.get.return_value = make_response(body=PENDING)
        self.sleep.side_effect = lambda _: mm.interrupt_current_processing(True)
        self.addCleanup(mm.interrupt_current_processing, False)
        with self.assertRaises(mm.InterruptProcessingException):
            generate(poll_interval=5)
        self.assertEqual(self.get.call_count, 1)

    def test_interrupt_before_first_poll(self):
        mm.interrupt_current_processing(True)
        self.addCleanup(mm.interrupt_current_processing, False)
        with self.assertRaises(mm.InterruptProcessingException):
            generate(video_job_id="job-9")
        self.get.assert_not_called()


class SchemaTests(unittest.TestCase):
    def test_video_job_id_is_the_last_optional_input(self):
        with patch.object(shared, "fetch_filtered_models", return_value=["m"]):
            inputs = Node.INPUT_TYPES()
        self.assertEqual(list(inputs["optional"])[-1], "video_job_id")
        self.assertNotIn("video_job_id", inputs["required"])
        self.assertEqual(inputs["optional"]["video_job_id"][0], "STRING")

    def test_video_job_id_changes_is_changed(self):
        base = dict(api_key="k", prompt="p", model="m")
        self.assertNotEqual(Node.IS_CHANGED(**base), Node.IS_CHANGED(**base, video_job_id="job-1"))
