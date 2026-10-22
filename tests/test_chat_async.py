import asyncio
import importlib.util
import inspect
import json
import os
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from comfy_execution.graph_utils import ExecutionBlocker

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

    def test_service_tier_precedes_on_error_widget(self):
        with patch.object(Node, "fetch_openrouter_models", return_value=["openai/gpt-4o"]):
            required = Node.INPUT_TYPES()["required"]
        self.assertEqual(list(required)[-2:], ["service_tier", "on_error"])
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


class OnErrorTests(unittest.TestCase):
    def setUp(self):
        Node.failures.clear()

    def failing_post(self):
        return patch.object(node_module.requests, "post", side_effect=node_module.requests.exceptions.Timeout("request timed out"))

    def test_on_error_is_last_required_widget(self):
        with patch.object(Node, "fetch_openrouter_models", return_value=["openai/gpt-4o"]):
            inputs = Node.INPUT_TYPES()
        self.assertEqual(list(inputs["required"])[-1], "on_error")
        self.assertEqual(inputs["required"]["on_error"][0], ["block", "raise"])
        self.assertEqual(inputs["required"]["on_error"][1]["default"], "block")
        self.assertIn("unique_id", inputs["hidden"])

    def test_block_returns_blockers_and_failed_stats(self):
        with self.failing_post():
            result = asyncio.run(respond(make_node(), unique_id="1"))

        self.assertEqual(len(result), 5)
        self.assertIsInstance(result[0], ExecutionBlocker)
        self.assertIsInstance(result[1], ExecutionBlocker)
        self.assertTrue(result[2].startswith("FAILED:"))
        self.assertIn("request timed out", result[2])
        self.assertEqual(result[3], "")
        self.assertEqual(result[4], 0.0)

    def test_block_is_the_default(self):
        with self.failing_post():
            result = asyncio.run(respond(make_node()))
        self.assertIsInstance(result[0], ExecutionBlocker)

    def test_raise_propagates_the_error(self):
        with self.failing_post(), self.assertRaisesRegex(RuntimeError, "request timed out"):
            asyncio.run(respond(make_node(), on_error="raise", unique_id="1"))

    def test_failure_changes_is_changed_for_that_node_only(self):
        before = is_changed(unique_id="1")
        with self.failing_post():
            asyncio.run(respond(make_node(), unique_id="1"))
        self.assertNotEqual(before, is_changed(unique_id="1"))
        self.assertEqual(before, is_changed(unique_id="2"))

    def test_raise_does_not_count_a_failure(self):
        before = is_changed(unique_id="1")
        with self.failing_post(), self.assertRaises(RuntimeError):
            asyncio.run(respond(make_node(), on_error="raise", unique_id="1"))
        self.assertEqual(before, is_changed(unique_id="1"))

    def test_missing_api_key_goes_through_on_error(self):
        with patch.dict(os.environ, {"LLM_KEY": ""}), patch.object(node_module.shared, "get_api_key", return_value=None):
            result = asyncio.run(respond(make_node()))
            self.assertTrue(result[2].startswith("FAILED: API Key not provided"))
            with self.assertRaisesRegex(RuntimeError, "API Key not provided"):
                asyncio.run(respond(make_node(), on_error="raise"))

    def test_bad_response_shape_goes_through_on_error(self):
        bad = Mock()
        bad.json.return_value = {"choices": []}
        with patch.object(node_module.requests, "post", return_value=bad):
            result = asyncio.run(respond(make_node()))
        self.assertTrue(result[2].startswith("FAILED: Invalid response format"))

    def test_image_encoding_error_goes_through_on_error(self):
        with patch.object(Node, "image_to_base64", side_effect=ValueError("bad tensor")):
            result = asyncio.run(respond(make_node(), image_1=object()))
        self.assertTrue(result[2].startswith("FAILED: Error processing image_1: bad tensor"))

    def test_failed_chat_turn_is_not_saved(self):
        node = make_node()
        node.chat_manager = Mock()
        node.chat_manager.get_or_create_session.return_value = ("session", [{"role": "system", "content": "system"}])
        with self.failing_post():
            result = asyncio.run(node.generate_response(
                api_key="test-key", system_prompt="system", user_message_box="hello", model="openai/gpt-4o",
                web_search=False, cheapest=False, fastest=False, temperature=1.0, pdf_engine="auto",
                chat_mode=True))

        self.assertTrue(result[2].startswith("FAILED:"))
        node.chat_manager.save_conversation.assert_not_called()


SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}


class JsonSchemaTests(unittest.TestCase):
    def sent_payload(self, **kwargs):
        with patch.object(node_module.requests, "post", return_value=make_response('{"a": 1}')) as post:
            asyncio.run(respond(make_node(), **kwargs))
        return post.call_args.kwargs["json"]

    def test_json_schema_is_last_widget(self):
        with patch.object(Node, "fetch_openrouter_models", return_value=["openai/gpt-4o"]):
            inputs = Node.INPUT_TYPES()
        self.assertEqual(list(inputs["optional"])[-1], "json_schema")
        self.assertTrue(inputs["optional"]["json_schema"][1]["multiline"])

    def test_empty_schema_sends_nothing(self):
        for value in ({}, {"json_schema": ""}, {"json_schema": "  \n"}):
            payload = self.sent_payload(**value)
            self.assertNotIn("response_format", payload)
            self.assertNotIn("provider", payload)

    def test_schema_sends_response_format_and_require_parameters(self):
        payload = self.sent_payload(json_schema=json.dumps(SCHEMA))
        self.assertEqual(payload["response_format"], {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": SCHEMA},
        })
        self.assertEqual(payload["provider"], {"require_parameters": True})

    def test_unsupported_params_are_not_sent(self):
        with patch.object(Node, "model_params", {"openai/gpt-4o": ["response_format", "seed"]}):
            payload = self.sent_payload(json_schema=json.dumps(SCHEMA), reasoning_effort="high")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("reasoning", payload)
        self.assertEqual(payload["seed"], 0)

    def test_unknown_model_keeps_params(self):
        with patch.object(Node, "model_params", {}):
            payload = self.sent_payload(json_schema=json.dumps(SCHEMA))
        self.assertEqual(payload["temperature"], 1.0)

    def test_schema_combines_with_model_modifiers(self):
        with patch.object(node_module.requests, "post", return_value=make_response('{"a": 1}')) as post:
            asyncio.run(node_respond_cheapest(json.dumps(SCHEMA)))
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "openai/gpt-4o:floor")
        self.assertIn("response_format", payload)

    def test_invalid_schema_fails_before_any_request(self):
        with patch.object(node_module.requests, "post") as post:
            result = asyncio.run(respond(make_node(), json_schema="{not json"))
            post.assert_not_called()
            self.assertTrue(result[2].startswith("FAILED: json_schema is not valid JSON"))
            with self.assertRaisesRegex(RuntimeError, "json_schema is not valid JSON"):
                asyncio.run(respond(make_node(), json_schema="{not json", on_error="raise"))

    def test_non_json_reply_goes_through_on_error(self):
        with patch.object(node_module.requests, "post", return_value=make_response("Sure! here you go")):
            result = asyncio.run(respond(make_node(), json_schema=json.dumps(SCHEMA)))
            self.assertIsInstance(result[0], ExecutionBlocker)
            self.assertTrue(result[2].startswith("FAILED: Reply is not valid JSON"))
            with self.assertRaisesRegex(RuntimeError, "Reply is not valid JSON"):
                asyncio.run(respond(make_node(), json_schema=json.dumps(SCHEMA), on_error="raise"))

    def test_non_json_reply_is_fine_without_schema(self):
        with patch.object(node_module.requests, "post", return_value=make_response("plain text")):
            result = asyncio.run(respond(make_node()))
        self.assertEqual(result[0], "plain text")

    def test_valid_reply_comes_out_unchanged(self):
        raw = '{"a":   1}'
        with patch.object(node_module.requests, "post", return_value=make_response(raw)):
            result = asyncio.run(respond(make_node(), json_schema=json.dumps(SCHEMA)))
        self.assertEqual(result[0], raw)

    def test_failed_json_reply_is_not_saved_in_chat_mode(self):
        node = make_node()
        node.chat_manager = Mock()
        node.chat_manager.get_or_create_session.return_value = ("session", [{"role": "system", "content": "system"}])
        with patch.object(node_module.requests, "post", return_value=make_response("nope")):
            asyncio.run(node.generate_response(
                api_key="test-key", system_prompt="system", user_message_box="hello", model="openai/gpt-4o",
                web_search=False, cheapest=False, fastest=False, temperature=1.0, pdf_engine="auto",
                chat_mode=True, json_schema=json.dumps(SCHEMA)))
        node.chat_manager.save_conversation.assert_not_called()

    def test_schema_changes_cache_key(self):
        self.assertNotEqual(is_changed(), is_changed(json_schema=json.dumps(SCHEMA)))
        self.assertEqual(is_changed(), is_changed(json_schema=""))


def node_respond_cheapest(json_schema):
    return make_node().generate_response(
        api_key="test-key", system_prompt="system", user_message_box="hello", model="openai/gpt-4o",
        web_search=False, cheapest=True, fastest=False, temperature=1.0, pdf_engine="auto", chat_mode=False,
        json_schema=json_schema)


if __name__ == "__main__":
    unittest.main()
