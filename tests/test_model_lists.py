import contextlib
import importlib.util
import io
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))  # ComfyUI root, for comfy.utils


def load_modules():
    package_name = "comfy_openrouter_model_lists_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package
    modules = {}
    for name in ("openrouter_shared", "chat_manager", "node", "node_speech", "node_image_api", "node_decision"):
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


modules = load_modules()
shared = modules["openrouter_shared"]
Chat = modules["node"].OpenRouterNode
Speech = modules["node_speech"].OpenRouterSpeechNode
ImageAPI = modules["node_image_api"].OpenRouterImageAPINode
Decision = modules["node_decision"].OpenRouterDecisionNode
NODES = (Chat, Speech, ImageAPI, Decision)

MODELS = [
    {"id": "b/model", "architecture": {"modality": "text->text", "input_modalities": ["text"]}, "supported_parameters": {"n": {"max": 2}}},
    {"id": "a/model", "architecture": {"modality": "text->text", "input_modalities": ["text"]}, "supported_parameters": {}},
]


def fake_get(*args, **kwargs):
    response = Mock()
    response.json.return_value = {"data": MODELS}
    return response


class ModelListTests(unittest.TestCase):
    def setUp(self):
        for node in NODES:
            for name, value in (("models_cache", None), ("last_fetch_time", 0)):
                patcher = patch.object(node, name, value)
                patcher.start()
                self.addCleanup(patcher.stop)
        patcher = patch.object(ImageAPI, "model_params", {})
        patcher.start()
        self.addCleanup(patcher.stop)

    def wait_for_refresh(self):
        deadline = time.time() + 5
        while shared._refresh_running and time.time() < deadline:
            time.sleep(0.005)
        self.assertFalse(shared._refresh_running)

    def test_hanging_fetch_does_not_block_and_a_later_call_returns_the_fetched_list(self):
        release = threading.Event()
        self.addCleanup(self.wait_for_refresh)
        self.addCleanup(release.set)

        def hang(*args, **kwargs):
            release.wait(5)
            return fake_get()

        with patch.object(shared.requests, "get", side_effect=hang):
            start = time.time()
            for node in NODES:
                self.assertEqual(node.INPUT_TYPES()["required"]["model"][0], node._fallback_models)
            self.assertLess(time.time() - start, 1)
            release.set()
            self.wait_for_refresh()
            for node in NODES:
                self.assertEqual(node.INPUT_TYPES()["required"]["model"][0], ["a/model", "b/model"])

    def test_image_api_model_params_are_available_after_a_refresh(self):
        with patch.object(shared.requests, "get", side_effect=fake_get):
            ImageAPI.fetch_openrouter_models()
            self.wait_for_refresh()
        self.assertEqual(ImageAPI.model_params, {"b/model": {"n": {"max": 2}}, "a/model": {}})
        self.assertEqual(ImageAPI.max_images_per_call("b/model"), 2)

    def test_only_one_refresh_runs_at_a_time(self):
        running, peak = 0, 0
        lock = threading.Lock()

        def slow_get(*args, **kwargs):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.02)
            with lock:
                running -= 1
            return fake_get()

        with patch.object(shared.requests, "get", side_effect=slow_get) as get:
            for node in NODES:
                node.fetch_openrouter_models()
            self.wait_for_refresh()
        self.assertEqual((get.call_count, peak), (len(NODES), 1))
        self.assertTrue(all(node.models_cache == ["a/model", "b/model"] for node in NODES))

    def test_fresh_list_is_not_fetched_again(self):
        Speech.models_cache = ["x/speech"]
        Speech.last_fetch_time = time.time()
        with patch.object(shared.requests, "get", side_effect=fake_get) as get:
            self.assertEqual(Speech.fetch_openrouter_models(), ["x/speech"])
            self.wait_for_refresh()
        get.assert_not_called()

    def test_stale_list_is_returned_while_it_refreshes(self):
        Speech.models_cache = ["x/speech"]
        Speech.last_fetch_time = time.time() - Speech.cache_duration - 1
        release = threading.Event()

        def gated_get(*args, **kwargs):
            release.wait(5)
            return fake_get()

        with patch.object(shared.requests, "get", side_effect=gated_get):
            self.assertEqual(Speech.fetch_openrouter_models(), ["x/speech"])
            release.set()
            self.wait_for_refresh()
            self.assertEqual(Speech.fetch_openrouter_models(), ["a/model", "b/model"])

    def test_failed_refresh_keeps_the_previous_list_and_prints_one_line(self):
        Speech.models_cache = ["x/speech"]
        Speech.last_fetch_time = time.time() - Speech.cache_duration - 1
        out = io.StringIO()
        with patch.object(shared.requests, "get", side_effect=OSError("down")), contextlib.redirect_stdout(out):
            self.assertEqual(Speech.fetch_openrouter_models(), ["x/speech"])
            self.wait_for_refresh()
        self.assertEqual(Speech.models_cache, ["x/speech"])
        self.assertEqual(len(out.getvalue().strip().splitlines()), 1)

    def test_saved_model_missing_from_the_list_passes_validation(self):
        for node in NODES:
            self.assertTrue(node.VALIDATE_INPUTS(model="saved/model"))


if __name__ == "__main__":
    unittest.main()
