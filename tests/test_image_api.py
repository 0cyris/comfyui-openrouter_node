import asyncio
import base64
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))  # ComfyUI root, for comfy.utils


def load_module():
    package_name = "comfy_openrouter_image_api_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package
    modules = []
    for name in ("openrouter_shared", "node_image_api"):
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


shared, node_module = load_module()
Node = node_module.OpenRouterImageAPINode


def png_b64(width, height):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (255, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_response(status=200, body=None, text=""):
    response = Mock()
    response.status_code = status
    response.json.return_value = body
    response.text = text
    return response


def generate(*args, **kwargs):
    return asyncio.run(Node().generate_images(*args, **kwargs))


def build(**overrides):
    args = dict(model="m", prompt="p", n=1, aspect_ratio="auto", resolution="default",
                quality="auto", background="auto", seed=0, references=[])
    args.update(overrides)
    return Node.build_payload(**args)


class PayloadTests(unittest.TestCase):
    def test_defaults_omit_optional_fields(self):
        payload = build()
        self.assertEqual(payload, {"model": "m", "prompt": "p", "seed": 0, "output_format": "png"})

    def test_non_default_values_are_sent(self):
        payload = build(n=4, aspect_ratio="2:3", resolution="2K", quality="high", background="opaque")
        self.assertEqual(payload["n"], 4)
        self.assertEqual(payload["aspect_ratio"], "2:3")
        self.assertEqual(payload["resolution"], "2K")
        self.assertEqual(payload["quality"], "high")
        self.assertEqual(payload["background"], "opaque")

    def test_size_is_never_sent(self):
        self.assertNotIn("size", build(aspect_ratio="2:3", resolution="2K"))

    def test_prompt_input_overrides_prompt(self):
        with patch.object(node_module.requests, "post", return_value=make_response(
                body={"data": [{"b64_json": png_b64(4, 4)}], "usage": {"cost": 0.01}})) as post, \
                patch.object(shared, "fetch_credits", return_value="Remaining: $1.000"):
            generate("key", "widget prompt", "m", prompt_input="linked prompt")
            self.assertEqual(post.call_args.kwargs["json"]["prompt"], "linked prompt")
            generate("key", "widget prompt", "m", prompt_input="  ")
            self.assertEqual(post.call_args.kwargs["json"]["prompt"], "widget prompt")


class CapabilityFilterTests(unittest.TestCase):
    def setUp(self):
        self.old_params = Node.model_params
        Node.model_params = {"known": {"aspect_ratio": {"type": "enum", "values": ["2:3"]},
                                       "output_format": {"type": "enum", "values": ["png"]}}}

    def tearDown(self):
        Node.model_params = self.old_params

    def test_unsupported_keys_are_dropped(self):
        payload = build(model="known", n=4, aspect_ratio="2:3", resolution="2K", seed=5)
        dropped = Node.drop_unsupported(payload)
        self.assertEqual(sorted(dropped), ["n", "resolution", "seed"])
        self.assertEqual(payload, {"model": "known", "prompt": "p", "aspect_ratio": "2:3", "output_format": "png"})

    def test_unknown_model_keeps_everything(self):
        payload = build(model="unknown", n=4, resolution="2K")
        before = dict(payload)
        self.assertEqual(Node.drop_unsupported(payload), [])
        self.assertEqual(payload, before)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.old_params = Node.model_params
        Node.model_params = {
            "single": {"n": {"type": "range", "min": 1, "max": 1}, "seed": {"type": "boolean"}},
            "three": {"n": {"type": "range", "min": 1, "max": 3}},
            "no_n": {},
        }

    def tearDown(self):
        Node.model_params = self.old_params

    def run_generate(self, model, n):
        sent = []

        def fake_post(url, **kwargs):
            sent.append(kwargs["json"])
            count = kwargs["json"].get("n", 1)
            return make_response(body={"data": [{"b64_json": png_b64(4, 4)}] * count, "usage": {"cost": 0.01}})

        with patch.object(node_module.requests, "post", side_effect=fake_post), \
                patch.object(shared, "fetch_credits", return_value=""):
            images, stats, _, cost = generate("key", "p", model, n=n, seed=10)
        return images, stats, cost, sent

    def test_max_one_splits_into_single_calls_with_distinct_seeds(self):
        images, stats, cost, sent = self.run_generate("single", 4)
        self.assertEqual(images.shape[0], 4)
        self.assertEqual(len(sent), 4)
        self.assertTrue(all("n" not in p for p in sent))
        self.assertEqual(sorted(p["seed"] for p in sent), [10, 11, 12, 13])
        self.assertIn("Cost: 0.04", stats)
        self.assertAlmostEqual(cost, 0.04)

    def test_remainder_call(self):
        images, _, _, sent = self.run_generate("three", 4)
        self.assertEqual(images.shape[0], 4)
        self.assertEqual(sorted(p.get("n", 1) for p in sent), [1, 3])

    def test_n_unsupported_means_one_per_call(self):
        images, _, _, sent = self.run_generate("no_n", 2)
        self.assertEqual(images.shape[0], 2)
        self.assertEqual(len(sent), 2)

    def test_unknown_model_sends_n_in_one_call(self):
        images, _, _, sent = self.run_generate("unknown", 4)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["n"], 4)


class ReferenceTests(unittest.TestCase):
    def test_batch_frames_are_sent_in_socket_order(self):
        single = torch.zeros((1, 8, 8, 3))
        batch = torch.stack([torch.full((8, 8, 3), v) for v in (0.25, 0.5, 0.75)])
        sent = {}

        def fake_post(url, **kwargs):
            sent.update(kwargs["json"])
            return make_response(body={"data": [{"b64_json": png_b64(4, 4)}]})

        with patch.object(node_module.requests, "post", side_effect=fake_post), \
                patch.object(shared, "fetch_credits", return_value=""):
            generate("key", "p", "m", image_2=batch, image_1=single)

        refs = sent["input_references"]
        self.assertEqual(len(refs), 4)
        expected = [shared.image_to_base64(single)] + shared.image_batch_to_base64_list(batch)
        self.assertEqual([r["image_url"]["url"] for r in refs],
                         [f"data:image/png;base64,{b}" for b in expected])
        self.assertTrue(all(r["type"] == "image_url" for r in refs))
        self.assertEqual(len(set(expected)), 4)


class ResponseTests(unittest.TestCase):
    def test_mixed_sizes_resize_to_first(self):
        images = Node.decode_images([{"b64_json": png_b64(16, 8)}, {"b64_json": png_b64(32, 24)}])
        self.assertEqual(tuple(images.shape), (2, 8, 16, 3))

    def test_empty_data_raises(self):
        for data in ([], None):
            with self.assertRaises(RuntimeError):
                Node.decode_images(data)

    def test_full_generate_returns_batch_and_stats(self):
        body = {"data": [{"b64_json": png_b64(8, 8)}, {"b64_json": png_b64(8, 8)}], "usage": {"cost": 0.04}}
        with patch.object(node_module.requests, "post", return_value=make_response(body=body)), \
                patch.object(shared, "fetch_credits", return_value="Remaining: $1.000"):
            images, stats, credits, cost = generate("key", "p", "m", n=2)
        self.assertEqual(tuple(images.shape), (2, 8, 8, 3))
        self.assertIn("Cost: 0.04", stats)
        self.assertEqual(credits, "Remaining: $1.000")
        self.assertIsInstance(cost, float)
        self.assertAlmostEqual(cost, 0.04)

    def test_missing_usage_gives_zero_cost(self):
        body = {"data": [{"b64_json": png_b64(8, 8)}]}
        with patch.object(node_module.requests, "post", return_value=make_response(body=body)), \
                patch.object(shared, "fetch_credits", return_value=""):
            _, stats, _, cost = generate("key", "p", "m")
        self.assertEqual(cost, 0.0)
        self.assertIn("Cost: None", stats)


class HttpErrorTests(unittest.TestCase):
    def run_error(self, status, message):
        response = make_response(status=status, body={"error": {"message": message}})
        with patch.object(node_module.requests, "post", return_value=response):
            with self.assertRaises(RuntimeError) as ctx:
                generate("key", "p", "m")
        return str(ctx.exception)

    def test_400_raises_with_api_message(self):
        text = self.run_error(400, 'provider: Unrecognized key: "data"')
        self.assertIn('Unrecognized key: "data"', text)
        self.assertIn("400", text)

    def test_502_raises_with_api_message(self):
        self.assertIn("generation did not complete", self.run_error(502, "generation did not complete"))

    def test_non_json_error_body_uses_text(self):
        response = make_response(status=502, text="Bad Gateway")
        response.json.side_effect = ValueError
        with patch.object(node_module.requests, "post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "Bad Gateway"):
                generate("key", "p", "m")


class RetryTests(unittest.TestCase):
    def setUp(self):
        sleep = patch.object(node_module.asyncio, "sleep", new=self.no_sleep)
        sleep.start()
        self.addCleanup(sleep.stop)

    @staticmethod
    async def no_sleep(seconds):
        pass

    @staticmethod
    def error(status, message, metadata=None):
        error = {"message": message}
        if metadata:
            error["metadata"] = metadata
        return make_response(status=status, body={"error": error})

    @staticmethod
    def success():
        return make_response(body={"data": [{"b64_json": png_b64(4, 4)}], "usage": {"cost": 0.01}})

    def run_posts(self, responses, **kwargs):
        with patch.object(node_module.requests, "post", side_effect=responses) as post, \
                patch.object(shared, "fetch_credits", return_value=""):
            try:
                result = generate("key", "p", "m", **kwargs)
            except RuntimeError as e:
                result = e
        return result, post.call_count

    def test_moderation_400_then_success(self):
        moderation = self.error(400, "Gemini blocked this request through content moderation.")
        (images, _, _, _), calls = self.run_posts([moderation, self.success()])
        self.assertEqual(images.shape[0], 1)
        self.assertEqual(calls, 2)

    def test_moderation_detected_from_metadata(self):
        flagged = self.error(400, "Provider returned error", {"flagged_input": "text"})
        (images, _, _, _), calls = self.run_posts([flagged, self.success()])
        self.assertEqual((images.shape[0], calls), (1, 2))

    def test_502_then_success(self):
        (images, _, _, _), calls = self.run_posts([self.error(502, "generation did not complete"), self.success()])
        self.assertEqual((images.shape[0], calls), (1, 2))

    def test_timeout_then_success(self):
        (images, _, _, _), calls = self.run_posts([node_module.requests.Timeout("slow"), self.success()])
        self.assertEqual((images.shape[0], calls), (1, 2))

    def test_no_retries_raises_after_one_post(self):
        moderation = self.error(400, "blocked through content moderation")
        result, calls = self.run_posts([moderation, self.success()], retries=0)
        self.assertIsInstance(result, RuntimeError)
        self.assertEqual(calls, 1)

    def test_retries_exhausted_raises_last_error(self):
        result, calls = self.run_posts([self.error(502, "first"), self.error(503, "second")], retries=1)
        self.assertIn("second", str(result))
        self.assertEqual(calls, 2)

    def test_non_transient_400_raises_immediately(self):
        result, calls = self.run_posts([self.error(400, "n: must be exactly 1"), self.success()], retries=2)
        self.assertIn("n: must be exactly 1", str(result))
        self.assertEqual(calls, 1)

    def test_error_text_has_provider_and_reasons_but_not_flagged_input(self):
        metadata = {"provider_name": "Google", "reasons": ["sexual", "minors"], "raw": "x" * 400,
                    "flagged_input": "secret user text"}
        result, _ = self.run_posts([self.error(400, "blocked", metadata)], retries=0)
        text = str(result)
        self.assertIn("(provider: Google; reasons: sexual, minors; raw: " + "x" * 300 + ")", text)
        self.assertNotIn("x" * 301, text)
        self.assertNotIn("secret user text", text)

    def test_split_chunk_retries_independently(self):
        old_params = Node.model_params
        Node.model_params = {"single": {"n": {"type": "range", "min": 1, "max": 1}, "seed": {"type": "boolean"}}}
        self.addCleanup(setattr, Node, "model_params", old_params)
        failed = []

        def fake_post(url, **kwargs):
            if kwargs["json"]["seed"] == 1 and not failed:
                failed.append(True)
                return self.error(502, "generation did not complete")
            return self.success()

        with patch.object(node_module.requests, "post", side_effect=fake_post) as post, \
                patch.object(shared, "fetch_credits", return_value=""):
            images, _, _, _ = generate("key", "p", "single", n=2)
        self.assertEqual(images.shape[0], 2)
        self.assertEqual(post.call_count, 3)


if __name__ == "__main__":
    unittest.main()
