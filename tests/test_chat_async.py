import asyncio
import importlib.util
import inspect
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))


def load_module():
    package_name = "comfy_openrouter_chat_async_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package
    modules = []
    for name in ("openrouter_shared", "chat_manager", "node"):
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules[-1]


node_module = load_module()
Node = node_module.OpenRouterNode


def make_response(text="done"):
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
    }
    return response


def respond(node):
    return node.generate_response(
        api_key="test-key", system_prompt="system", user_message_box="hello", model="openai/gpt-4o",
        web_search=False, cheapest=False, fastest=False, temperature=1.0, pdf_engine="auto", chat_mode=False)


def make_node():
    node = Node()
    node.fetch_credits = Mock(return_value="Remaining: $1.000")
    node.count_tokens = Mock(return_value=1)
    return node


class ChatAsyncTests(unittest.TestCase):
    def test_generate_response_is_coroutine_function(self):
        self.assertTrue(inspect.iscoroutinefunction(Node.generate_response))

    def test_basic_call_returns_output_tuple(self):
        with patch.object(node_module.requests, "post", return_value=make_response("hi")):
            result = asyncio.run(respond(make_node()))

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], "hi")
        self.assertEqual(tuple(result[1].shape), (1, 1, 1, 3))
        self.assertIn("3", result[2])
        self.assertEqual(result[3], "Remaining: $1.000")

    def test_calls_run_concurrently(self):
        def slow_post(*args, **kwargs):
            time.sleep(0.5)
            return make_response()

        async def run_both():
            return await asyncio.gather(respond(make_node()), respond(make_node()))

        with patch.object(node_module.requests, "post", side_effect=slow_post):
            start = time.monotonic()
            results = asyncio.run(run_both())
            elapsed = time.monotonic() - start

        self.assertEqual([r[0] for r in results], ["done", "done"])
        self.assertLess(elapsed, 0.9)


if __name__ == "__main__":
    unittest.main()
