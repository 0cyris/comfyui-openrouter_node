import asyncio
import contextlib
import importlib.util
import inspect
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))

NODE_MODULES = ("node", "node_image_gen", "node_image_api", "node_decision",
                "node_speech", "node_transcription", "node_rerank", "node_video_gen")


def load_modules():
    package_name = "comfy_openrouter_api_key_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package
    modules = {}
    for name in ("openrouter_shared", "chat_manager") + NODE_MODULES:
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


with patch.object(requests, "get", side_effect=requests.ConnectionError):
    modules = load_modules()
shared = modules["openrouter_shared"]

SECRET = "sk-or-test-typed-in-workflow"


class KeyResolutionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.key_file = os.path.join(self.dir.name, "openrouter_api_key.json")
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("LLM_KEY", None)
        path = patch.object(shared, "API_KEY_FILE", self.key_file)
        path.start()
        self.addCleanup(path.stop)

    def write_key_file(self, content):
        with open(self.key_file, "w") as f:
            f.write(content)

    def test_takes_no_ui_value(self):
        self.assertEqual(list(inspect.signature(shared.get_api_key).parameters), [])

    def test_missing_key_is_empty(self):
        self.assertEqual(shared.get_api_key(), "")

    def test_environment_variable(self):
        os.environ["LLM_KEY"] = "  env-key \n"
        self.assertEqual(shared.get_api_key(), "env-key")

    def test_key_file(self):
        self.write_key_file(json.dumps({"api_key": " file-key "}))
        self.assertEqual(shared.get_api_key(), "file-key")

    def test_environment_variable_wins_over_key_file(self):
        os.environ["LLM_KEY"] = "env-key"
        self.write_key_file(json.dumps({"api_key": "file-key"}))
        self.assertEqual(shared.get_api_key(), "env-key")

    def test_blank_environment_variable_falls_back_to_key_file(self):
        os.environ["LLM_KEY"] = "  "
        self.write_key_file(json.dumps({"api_key": "file-key"}))
        self.assertEqual(shared.get_api_key(), "file-key")

    def test_unreadable_key_file_is_empty(self):
        self.write_key_file("not json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(shared.get_api_key(), "")


class NodeKeyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("LLM_KEY", None)
        path = patch.object(shared, "API_KEY_FILE", os.path.join(self.dir.name, "missing.json"))
        path.start()
        self.addCleanup(path.stop)

    def decide(self, api_key=SECRET):
        node = modules["node_decision"].OpenRouterDecisionNode()
        response = Mock(status_code=200)
        response.json.return_value = {"answers": {"decision": {"type": "noul", "noul": 0.9}}}
        with patch.object(shared.requests, "post", return_value=response) as post:
            asyncio.run(node.decide(
                api_key=api_key, model="m", question_type="noul", instructions="q", criteria="",
                true_guidance="", false_guidance="", state_text="s", threshold=0.5,
                request_timeout=120, retries=0))
        return post

    def chat(self, api_key=SECRET):
        node = modules["node"].OpenRouterNode()
        node.fetch_credits = Mock(return_value="")
        node.count_tokens = Mock(return_value=1)
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "done"}}], "usage": {}}
        with patch.object(shared.requests, "post", return_value=response) as post:
            result = asyncio.run(node.generate_response(
                api_key=api_key, system_prompt="s", user_message_box="u", model="m", web_search=False,
                cheapest=False, fastest=False, temperature=1.0, pdf_engine="auto", chat_mode=False))
        return result, post

    def test_ui_value_is_ignored(self):
        os.environ["LLM_KEY"] = "env-key"
        self.assertEqual(self.decide().call_args.kwargs["headers"]["Authorization"], "Bearer env-key")
        _, post = self.chat()
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer env-key")

    def test_ui_value_is_not_a_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "API key not provided"):
            self.decide()
        result, post = self.chat()
        post.assert_not_called()
        self.assertTrue(result[2].startswith("FAILED: API Key not provided"))


class PlaceholderTests(unittest.TestCase):
    def test_api_key_stays_the_first_required_input(self):
        with patch.object(requests, "get", side_effect=requests.ConnectionError), \
                contextlib.redirect_stdout(io.StringIO()):
            for name in NODE_MODULES:
                node_class = next(iter(modules[name].NODE_CLASS_MAPPINGS.values()))
                required = node_class.INPUT_TYPES()["required"]
                self.assertEqual(next(iter(required)), "api_key", name)
                self.assertEqual(required["api_key"][0], "STRING", name)


class PromptScrubTests(unittest.TestCase):
    def scrub(self, prompt):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = shared.blank_prompt_api_keys({"prompt": prompt}, {"OpenRouterNode", "OpenRouterDecisionNode"})
        return result["prompt"], out.getvalue()

    def test_typed_key_is_blanked_and_not_logged(self):
        prompt, printed = self.scrub({
            "1": {"class_type": "OpenRouterNode", "inputs": {"api_key": SECRET, "model": "m"}},
            "2": {"class_type": "OpenRouterDecisionNode", "inputs": {"api_key": ["5", 0]}},
        })
        self.assertEqual(prompt["1"]["inputs"], {"api_key": "", "model": "m"})
        self.assertEqual(prompt["2"]["inputs"], {"api_key": ""})
        self.assertEqual(len(printed.splitlines()), 1)
        self.assertIn("LLM_KEY", printed)
        self.assertNotIn(SECRET, printed)

    def test_blank_key_is_quiet(self):
        prompt, printed = self.scrub({"1": {"class_type": "OpenRouterNode", "inputs": {"api_key": ""}}})
        self.assertEqual(prompt["1"]["inputs"], {"api_key": ""})
        self.assertEqual(printed, "")

    def test_other_nodes_are_untouched(self):
        original = {"1": {"class_type": "SomethingElse", "inputs": {"api_key": "theirs"}}}
        prompt, printed = self.scrub(original)
        self.assertEqual(prompt["1"]["inputs"], {"api_key": "theirs"})
        self.assertEqual(printed, "")


if __name__ == "__main__":
    unittest.main()
