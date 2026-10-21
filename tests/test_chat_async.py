import asyncio
import importlib.util
import inspect
import os
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


def make_response(text="done", cost=None):
    response = Mock()
    usage = {"prompt_tokens": 2, "completion_tokens": 3}
    if cost is not None:
        usage["cost"] = cost
    response.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": usage,
    }
    return response


def respond(node, **kwargs):
    return node.generate_response(
        api_key="test-key", system_prompt="system", user_message_box="hello", model="openai/gpt-4o",
        web_search=False, cheapest=False, fastest=False, temperature=1.0, pdf_engine="auto", chat_mode=False,
        **kwargs)


def is_changed(**kwargs):
    return Node.IS_CHANGED(
        api_key="test-key", system_prompt="system", user_message_box="hello", model="openai/gpt-4o",
        web_search=False, cheapest=False, fastest=False, temperature=1.0, pdf_engine="auto", chat_mode=False,
        **kwargs)


def make_node():
    node = Node()
    node.fetch_credits = Mock(return_value="Remaining: $1.000")
    node.count_tokens = Mock(return_value=1)
    return node


def setUpModule():
    patcher = patch.dict(os.environ, {"LLM_KEY": "test-key"})
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class ChatAsyncTests(unittest.TestCase):
    def test_generate_response_is_coroutine_function(self):
        self.assertTrue(inspect.iscoroutinefunction(Node.generate_response))

    def test_basic_call_returns_output_tuple(self):
        with patch.object(node_module.requests, "post", return_value=make_response("hi")):
            result = asyncio.run(respond(make_node()))

        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], "hi")
        self.assertEqual(tuple(result[1].shape), (1, 1, 1, 3))
        self.assertIn("3", result[2])
        self.assertEqual(result[3], "Remaining: $1.000")

    def test_cost_output_is_last(self):
        self.assertEqual(Node.RETURN_TYPES[-1], "FLOAT")
        self.assertEqual(Node.RETURN_NAMES, ("Output", "image", "Stats", "Credits", "cost"))

    def test_cost_comes_from_usage(self):
        with patch.object(node_module.requests, "post", return_value=make_response(cost=0.00123)):
            result = asyncio.run(respond(make_node()))

        self.assertEqual(result[4], 0.00123)
        self.assertIn("Cost: 0.00123", result[2])

    def test_cost_is_zero_when_usage_has_none(self):
        with patch.object(node_module.requests, "post", return_value=make_response()):
            result = asyncio.run(respond(make_node()))

        self.assertEqual(result[4], 0.0)
        self.assertIn("Cost: 0.0", result[2])

    def test_error_return_has_zero_cost(self):
        with patch.object(node_module.requests, "post", side_effect=node_module.requests.exceptions.Timeout("request timed out")):
            result = asyncio.run(respond(make_node()))

        self.assertEqual(len(result), 5)
        self.assertEqual(result[4], 0.0)

    def test_chat_mode_keeps_working_with_cost(self):
        node = make_node()
        node.chat_manager = Mock()
        node.chat_manager.get_or_create_session.return_value = ("session", [{"role": "system", "content": "system"}])
        with patch.object(node_module.requests, "post", return_value=make_response("hi", cost=0.5)):
            result = asyncio.run(node.generate_response(
                api_key="test-key", system_prompt="system", user_message_box="hello", model="openai/gpt-4o",
                web_search=False, cheapest=False, fastest=False, temperature=1.0, pdf_engine="auto",
                chat_mode=True))

        self.assertEqual(result[0], "hi")
        self.assertEqual(result[4], 0.5)
        node.chat_manager.save_conversation.assert_called_once()

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


def make_tier_response(service_tier):
    response = make_response()
    response.json.return_value["service_tier"] = service_tier
    return response


class ServiceTierTests(unittest.TestCase):
    def sent_payload(self, **kwargs):
        with patch.object(node_module.requests, "post", return_value=make_response()) as post:
            asyncio.run(respond(make_node(), **kwargs))
        return post.call_args.kwargs["json"]

    def test_service_tier_is_last_required_widget(self):
        with patch.object(Node, "fetch_openrouter_models", return_value=["openai/gpt-4o"]):
            required = Node.INPUT_TYPES()["required"]
        self.assertEqual(list(required)[-1], "service_tier")
        self.assertEqual(required["service_tier"][0], ["auto", "default", "flex", "priority", "ultrafast"])
        self.assertEqual(required["service_tier"][1], {"default": "auto"})

    def test_auto_sends_no_service_tier(self):
        self.assertNotIn("service_tier", self.sent_payload())
        self.assertNotIn("service_tier", self.sent_payload(service_tier="auto"))

    def test_explicit_tier_is_sent(self):
        for tier in ("default", "flex", "priority", "ultrafast"):
            with self.subTest(tier=tier):
                self.assertEqual(self.sent_payload(service_tier=tier)["service_tier"], tier)

    def test_response_tier_shown_in_stats(self):
        with patch.object(node_module.requests, "post", return_value=make_tier_response("flex")):
            result = asyncio.run(respond(make_node(), service_tier="flex"))
        self.assertIn("Service tier: flex", result[2])

    def test_missing_or_null_response_tier_leaves_stats_unchanged(self):
        for response in (make_response(), make_tier_response(None)):
            with patch.object(node_module.requests, "post", return_value=response):
                result = asyncio.run(respond(make_node()))
            self.assertEqual(result[0], "done")
            self.assertNotIn("Service tier", result[2])

    def test_changing_service_tier_changes_cache_key(self):
        self.assertNotEqual(is_changed(service_tier="auto"), is_changed(service_tier="flex"))
        self.assertEqual(is_changed(), is_changed(service_tier="auto"))


if __name__ == "__main__":
    unittest.main()
