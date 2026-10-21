import asyncio
import base64
import importlib.util
import io
import json
import os
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))


def load_module():
    package_name = "comfy_openrouter_decision_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package
    modules = []
    for name in ("openrouter_shared", "node_decision"):
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


shared, node_module = load_module()
Node = node_module.OpenRouterDecisionNode

URL = "https://openrouter.ai/api/alpha/decisions"


def make_response(status=200, body=None, text=""):
    response = Mock()
    response.status_code = status
    response.json.return_value = body
    response.text = text
    return response


def answer_body(noul, **extra):
    return {"answers": {"decision": {"type": "noul", "noul": noul}}, **extra}


def choice_body(choice, confidence, probabilities, **extra):
    return {"answers": {"decision": {"type": "choice", "choice": choice, "confidence": confidence,
                                     "probabilities": probabilities}}, **extra}


def score_body(score, legend, **extra):
    return {"answers": {"decision": {"type": "score", "score": score, "confidence": 0.9,
                                     "probabilities": {}, "legend": legend}}, **extra}


def decide(**overrides):
    args = dict(api_key="key", model="m", question_type="noul", instructions="Is it a bug?", criteria="",
                true_guidance="broken", false_guidance="fine", state_text="the text", threshold=0.5,
                request_timeout=120, retries=1)
    args.update(overrides)
    return asyncio.run(Node().decide(**args))


def run_with(body, **overrides):
    with patch.object(shared.requests, "post", return_value=make_response(body=body)) as post:
        return decide(**overrides), post


def frames(count, width, height):
    return torch.rand(count, height, width, 3)


def decode_part(part):
    prefix = "data:image/jpeg;base64,"
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith(prefix), part["type"]
    image = Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"][len(prefix):])))
    assert image.format == "JPEG"
    return image.size


def setUpModule():
    patcher = patch.dict(os.environ, {"LLM_KEY": "test-key"})
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class PayloadTests(unittest.TestCase):
    def test_sends_single_noul_question_to_alpha_endpoint(self):
        _, post = run_with(answer_body(0.9))
        self.assertEqual(post.call_args.args[0], URL)
        self.assertEqual(post.call_args.kwargs["json"], {
            "model": "m",
            "questions": {"decision": {
                "type": "noul",
                "instructions": "Is it a bug?",
                "criteria": {"true": "broken", "false": "fine"},
            }},
            "state": ["the text"],
        })

    def test_empty_state_text_sends_empty_state(self):
        _, post = run_with(answer_body(0.9), state_text="")
        self.assertEqual(post.call_args.kwargs["json"]["state"], [])


class ContextImageTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(Node, "model_input_modalities", {"m": ["text", "image"], "text-only": ["text"]})
        patcher.start()
        self.addCleanup(patcher.stop)

    def sent_state(self, **overrides):
        _, post = run_with(answer_body(0.9), **overrides)
        return post.call_args.kwargs["json"]["state"]

    def test_state_is_text_then_images_in_socket_order_with_all_frames(self):
        state = self.sent_state(image_10=frames(1, 600, 400), image_2=frames(2, 320, 300),
                                image_1=frames(1, 640, 480))
        self.assertEqual(state[0], "the text")
        self.assertEqual([decode_part(part) for part in state[1:]],
                         [(640, 480), (320, 300), (320, 300), (600, 400)])

    def test_images_without_text_are_sent_alone(self):
        state = self.sent_state(state_text="", image_1=frames(1, 640, 480))
        self.assertEqual([decode_part(part) for part in state], [(640, 480)])

    def test_unconnected_image_is_skipped(self):
        state = self.sent_state(image_1=None, image_2=frames(1, 640, 480))
        self.assertEqual(len(state), 2)

    def test_too_large_image_is_downscaled_keeping_aspect_ratio(self):
        width, height = decode_part(self.sent_state(image_1=frames(1, 3000, 2000))[1])
        self.assertEqual((width, height), (1568, 1045))
        self.assertAlmostEqual(width / height, 3000 / 2000, delta=0.01)

    def test_too_small_image_is_upscaled_keeping_aspect_ratio(self):
        width, height = decode_part(self.sent_state(image_1=frames(1, 100, 50))[1])
        self.assertEqual((width, height), (512, 256))

    def test_image_under_minimum_pixel_count_is_upscaled(self):
        width, height = decode_part(self.sent_state(image_1=frames(1, 260, 260))[1])
        self.assertGreaterEqual(width * height, 78400)
        self.assertEqual(width, height)

    def test_in_range_image_keeps_its_size(self):
        self.assertEqual(decode_part(self.sent_state(image_1=frames(1, 1568, 1568))[1]), (1568, 1568))
        self.assertEqual(decode_part(self.sent_state(image_1=frames(1, 1000, 300))[1]), (1000, 300))

    def test_text_only_model_with_image_raises_without_request(self):
        with patch.object(shared.requests, "post") as post:
            with self.assertRaisesRegex(RuntimeError, "text-only does not accept images"):
                decide(model="text-only", image_1=frames(1, 640, 480))
        post.assert_not_called()

    def test_text_only_model_without_images_sends_text_only(self):
        self.assertEqual(self.sent_state(model="text-only"), ["the text"])

    def test_model_missing_from_catalog_is_not_refused(self):
        self.assertEqual(len(self.sent_state(model="unlisted", image_1=frames(1, 640, 480))), 2)

    def test_choice_and_score_include_images(self):
        for question_type, body in (("choice", choice_body("a", 0.8, {})), ("score", score_body(1.0, {"0": "x", "1": "y"}))):
            _, post = run_with(body, question_type=question_type, criteria="a\nb", image_1=frames(1, 640, 480))
            self.assertEqual(len(post.call_args.kwargs["json"]["state"]), 2)


LEVELS = [0.1, 0.4, 0.7, 0.9]


def candidates(count=3, width=640, height=480):
    return torch.stack([torch.full((height, width, 3), LEVELS[i]) for i in range(count)])


def candidate_index(payload):
    part = payload["state"][-1]
    pixel = Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))).getpixel((10, 10))[0]
    return min(range(len(LEVELS)), key=lambda i: abs(LEVELS[i] * 255 - pixel))


def per_candidate_post(responses, barrier=None):
    """Mocked requests.post that answers each request from the queue of the candidate it carries."""
    sent = []

    def post(url, headers, json, timeout):
        index = candidate_index(json)
        sent.append(index)
        if barrier:
            barrier.wait()
        return responses[index].pop(0)

    return patch.object(shared.requests, "post", side_effect=post), sent


def ok(body):
    return [make_response(body=body)]


class CandidateTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(Node, "model_input_modalities", {"m": ["text", "image"], "text-only": ["text"]})
        patcher.start()
        self.addCleanup(patcher.stop)

        async def no_sleep(seconds):
            pass
        patcher = patch.object(shared.asyncio, "sleep", new=no_sleep)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_batch(self, noul_values, usages=None, **overrides):
        usages = usages or [None] * len(noul_values)
        responses = [ok(answer_body(v, **({"usage": {"cost": u}} if u is not None else {}))) for v, u in zip(noul_values, usages)]
        patcher, sent = per_candidate_post(responses)
        with patcher as post:
            outputs = decide(candidates=candidates(len(noul_values)), **overrides)
        return outputs, post, sent

    def test_one_request_per_candidate_with_text_context_then_candidate(self):
        _, post, sent = self.run_batch([0.2, 0.9, 0.5], image_1=frames(1, 640, 480), image_2=frames(1, 320, 300))
        self.assertEqual(post.call_count, 3)
        self.assertEqual(sorted(sent), [0, 1, 2])
        for call in post.call_args_list:
            state = call.kwargs["json"]["state"]
            self.assertEqual(state[0], "the text")
            self.assertEqual([decode_part(part) for part in state[1:]], [(640, 480), (320, 300), (640, 480)])

    def test_candidate_scaled_like_context_images(self):
        patcher, _ = per_candidate_post([ok(answer_body(0.5))])
        with patcher as post:
            decide(candidates=candidates(1, 100, 50))
        self.assertEqual(decode_part(post.call_args.kwargs["json"]["state"][-1]), (512, 256))

    def test_requests_run_concurrently(self):
        patcher, _ = per_candidate_post([ok(answer_body(v)) for v in (0.2, 0.9, 0.5)], barrier=threading.Barrier(3, timeout=5))
        with patcher:
            self.assertEqual(decide(candidates=candidates(3))[1], 0.9)

    def test_best_is_highest_value(self):
        (answer, value, passed, _, _, best_index, _), _, _ = self.run_batch([0.2, 0.9, 0.5])
        self.assertEqual((answer, value, passed, best_index), ("true", 0.9, True, 1))

    def test_tie_picks_lowest_index(self):
        outputs, _, _ = self.run_batch([0.3, 0.8, 0.8])
        self.assertEqual(outputs[5], 1)

    def test_best_image_is_original_frame(self):
        outputs, _, _ = self.run_batch([0.2, 0.9, 0.5])
        self.assertTrue(torch.equal(outputs[6], candidates(3)[1:2]))

    def test_best_image_keeps_original_size(self):
        patcher, _ = per_candidate_post([ok(answer_body(0.5))])
        with patcher:
            outputs = decide(candidates=candidates(1, 100, 50))
        self.assertEqual(outputs[6].shape, (1, 50, 100, 3))

    def test_details_is_array_in_candidate_order(self):
        outputs, _, _ = self.run_batch([0.2, 0.9, 0.5])
        self.assertEqual([d["noul"] for d in json.loads(outputs[3])], [0.2, 0.9, 0.5])

    def test_cost_is_summed_with_missing_usage_as_zero(self):
        outputs, _, _ = self.run_batch([0.2, 0.9, 0.5], usages=[0.00001, None, 0.00002])
        self.assertAlmostEqual(outputs[4], 0.00003)

    def test_failing_candidate_retries_on_its_own(self):
        responses = [ok(answer_body(0.2)),
                     [make_response(503, {"error": {"message": "busy"}}), make_response(body=answer_body(0.9))],
                     ok(answer_body(0.5))]
        patcher, sent = per_candidate_post(responses)
        with patcher:
            outputs = decide(candidates=candidates(3))
        self.assertEqual(sorted(sent), [0, 1, 1, 2])
        self.assertEqual((outputs[1], outputs[5]), (0.9, 1))

    def test_permanently_failing_candidate_raises(self):
        responses = [ok(answer_body(0.2)), [make_response(400, {"error": {"message": "bad image"}})], ok(answer_body(0.5))]
        patcher, _ = per_candidate_post(responses)
        with patcher:
            with self.assertRaisesRegex(RuntimeError, "OpenRouter Decision error 400: bad image"):
                decide(candidates=candidates(3))

    def test_stop_on_false_blocks_best_image_for_a_failed_best_candidate(self):
        outputs, _, _ = self.run_batch([0.1, 0.3], stop_on_false=True)
        self.assertIsInstance(outputs[6], node_module.ExecutionBlocker)
        self.assertEqual(outputs[6].message, "Decision check failed: answer false, value 0.30 is below threshold 0.50")
        self.assertEqual((outputs[0], outputs[2], outputs[5]), ("false", False, 1))
        self.assertAlmostEqual(outputs[1], 0.3)

    def test_stop_on_false_raises_for_a_failed_best_candidate_with_on_false_raise(self):
        with self.assertRaisesRegex(RuntimeError, r"Decision check failed: answer false, value 0\.30 is below threshold 0\.50"):
            self.run_batch([0.1, 0.3], stop_on_false=True, on_false="raise")

    def test_stop_on_false_passes_when_best_candidate_passes(self):
        outputs, _, _ = self.run_batch([0.1, 0.8], stop_on_false=True)
        self.assertEqual((outputs[2], outputs[5]), (True, 1))
        self.assertIsInstance(outputs[6], torch.Tensor)

    def test_failed_candidate_does_not_raise_by_default(self):
        outputs, _, _ = self.run_batch([0.1, 0.3])
        self.assertEqual(outputs[2], False)

    def test_no_candidates_is_a_single_request(self):
        outputs, post = run_with(answer_body(0.8, usage={"cost": 0.00002}))
        self.assertEqual(post.call_count, 1)
        self.assertEqual(outputs[:5], ("true", 0.8, True, json.dumps({"type": "noul", "noul": 0.8}), 0.00002))
        self.assertEqual(outputs[5], 0)
        self.assertEqual(outputs[6].shape, (1, 64, 64, 3))
        self.assertEqual(outputs[6].dtype, torch.float32)

    def test_text_only_model_with_candidates_raises_without_request(self):
        with patch.object(shared.requests, "post") as post:
            with self.assertRaisesRegex(RuntimeError, "text-only does not accept images"):
                decide(model="text-only", candidates=candidates(2))
        post.assert_not_called()

    def test_invalid_criteria_raises_without_request(self):
        with patch.object(shared.requests, "post") as post:
            with self.assertRaisesRegex(RuntimeError, "criteria is empty.*name: guidance"):
                decide(question_type="choice", criteria="", candidates=candidates(2))
        post.assert_not_called()

    def test_choice_candidates_pick_highest_confidence(self):
        patcher, _ = per_candidate_post([ok(choice_body("a", 0.4, {})), ok(choice_body("b", 0.95, {})), ok(choice_body("a", 0.6, {}))])
        with patcher:
            outputs = decide(question_type="choice", criteria="a\nb", candidates=candidates(3))
        self.assertEqual((outputs[0], outputs[1], outputs[5]), ("b", 0.95, 1))

    def test_score_candidates_pick_highest_score(self):
        legend = {"0": "bad", "1": "ok", "2": "good"}
        patcher, _ = per_candidate_post([ok(score_body(1.0, legend)), ok(score_body(2.0, legend)), ok(score_body(0.0, legend))])
        with patcher:
            outputs = decide(question_type="score", criteria="bad\nok\ngood", candidates=candidates(3))
        self.assertEqual((outputs[0], outputs[1], outputs[5]), ("good", 1.0, 1))


class CriteriaPayloadTests(unittest.TestCase):
    def test_choice_options_from_lines(self):
        criteria = "left: faces viewer-left\n\n  right : faces: viewer-right  \nback\n"
        _, post = run_with(choice_body("left", 0.8, {}), question_type="choice", criteria=criteria)
        self.assertEqual(post.call_args.kwargs["json"]["questions"], {"decision": {
            "type": "choice",
            "instructions": "Is it a bug?",
            "criteria": {"left": "faces viewer-left", "right": "faces: viewer-right", "back": None},
        }})

    def test_score_levels_from_lines(self):
        _, post = run_with(score_body(1.0, {"0": "bad", "1": "good: ok", "2": "great"}), question_type="score", criteria="bad\n\ngood: ok\n  great  ")
        self.assertEqual(post.call_args.kwargs["json"]["questions"], {"decision": {
            "type": "score",
            "instructions": "Is it a bug?",
            "criteria": ["bad", "good: ok", "great"],
        }})

    def test_noul_ignores_criteria(self):
        _, post = run_with(answer_body(0.9), criteria="a: b\nc")
        self.assertEqual(post.call_args.kwargs["json"]["questions"]["decision"]["criteria"],
                         {"true": "broken", "false": "fine"})


class ChoiceOutputTests(unittest.TestCase):
    def test_maps_choice_confidence_and_probabilities(self):
        body = choice_body("left", 0.7, {"left": 0.7, "right": 0.3}, usage={"cost": 0.00001})
        answer, value, passed, details, cost, *_ = run_with(body, question_type="choice", criteria="left\nright")[0]
        self.assertEqual((answer, value, passed, cost), ("left", 0.7, True, 0.00001))
        self.assertEqual(json.loads(details)["probabilities"], {"left": 0.7, "right": 0.3})

    def test_below_threshold_not_passed(self):
        body = choice_body("left", 0.4, {"left": 0.4, "right": 0.3})
        answer, value, passed, _, _, *_ = run_with(body, question_type="choice", criteria="left\nright")[0]
        self.assertEqual((answer, value, passed), ("left", 0.4, False))


    def test_confidence_falls_back_to_chosen_option_probability(self):
        body = {"answers": {"decision": {"type": "choice", "choice": "left", "probabilities": {"left": 0.8, "right": 0.2}}}}
        answer, value, _, _, _, *_ = run_with(body, question_type="choice", criteria="left\nright")[0]
        self.assertEqual((answer, value), ("left", 0.8))


class ScoreOutputTests(unittest.TestCase):
    legend = {"0": "bad", "1": "ok", "2": "good", "3": "great"}

    def run_score(self, score, **overrides):
        return run_with(score_body(score, self.legend), question_type="score", criteria="bad\nok\ngood\ngreat",
                        **overrides)[0]

    def test_normalises_score_by_level_count(self):
        answer, value, passed, details, _, *_ = self.run_score(2.0)
        self.assertEqual((answer, value, passed), ("good", 2 / 3, True))
        self.assertEqual(json.loads(details)["score"], 2.0)

    def test_answer_is_legend_text_of_nearest_level(self):
        self.assertEqual(self.run_score(1.4)[0], "ok")
        self.assertEqual(self.run_score(1.6)[0], "good")
        self.assertEqual(self.run_score(2.99)[0], "great")

    def test_answer_without_legend_uses_criteria_text(self):
        body = {"answers": {"decision": {"type": "score", "score": 1.6}}}
        answer, value, _, _, _, *_ = run_with(body, question_type="score", criteria="bad\nok\ngood\ngreat")[0]
        self.assertEqual((answer, value), ("good", 1.6 / 3))

    def test_threshold_boundary_is_inclusive(self):
        self.assertTrue(self.run_score(1.5, threshold=0.5)[2])
        self.assertFalse(self.run_score(1.4, threshold=0.5)[2])

    def test_single_level_gives_zero(self):
        body = score_body(0.0, {"0": "only"})
        answer, value, passed, _, _, *_ = run_with(body, question_type="score", criteria="only")[0]
        self.assertEqual((answer, value, passed), ("only", 0.0, False))


class InvalidCriteriaTests(unittest.TestCase):
    def assert_rejected(self, pattern, **overrides):
        with patch.object(shared.requests, "post") as post:
            with self.assertRaisesRegex(RuntimeError, pattern):
                decide(**overrides)
        post.assert_not_called()

    def test_choice_without_options(self):
        self.assert_rejected("criteria is empty.*name: guidance", question_type="choice", criteria=" \n\n")

    def test_choice_duplicate_names(self):
        self.assert_rejected("Duplicate option 'a'", question_type="choice", criteria="a: x\nb\na: y")

    def test_score_without_levels(self):
        self.assert_rejected("criteria is empty.*lowest first", question_type="score", criteria="\n \n")


class OutputTests(unittest.TestCase):
    def test_above_threshold_is_true(self):
        answer, value, passed, details, cost, *_ = run_with(answer_body(0.8))[0]
        self.assertEqual((answer, value, passed), ("true", 0.8, True))
        self.assertEqual(json.loads(details), {"type": "noul", "noul": 0.8})

    def test_below_threshold_is_false(self):
        answer, value, passed, _, _, *_ = run_with(answer_body(0.2))[0]
        self.assertEqual((answer, value, passed), ("false", 0.2, False))

    def test_threshold_boundary_is_inclusive(self):
        answer, value, passed, _, _, *_ = run_with(answer_body(0.75), threshold=0.75)[0]
        self.assertEqual((answer, passed), ("true", True))

    def test_cost_from_usage(self):
        self.assertEqual(run_with(answer_body(0.5, usage={"cost": 0.00002}))[0][4], 0.00002)

    def test_missing_usage_gives_zero_cost(self):
        self.assertEqual(run_with(answer_body(0.5))[0][4], 0.0)

    def test_missing_cost_gives_zero_cost(self):
        self.assertEqual(run_with(answer_body(0.5, usage={"input_tokens": 3}))[0][4], 0.0)


class ModelListTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("models_cache", None), ("model_input_modalities", {}), ("last_fetch_time", 0)):
            patcher = patch.object(Node, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def wait_for_refresh(self):
        deadline = time.time() + 5
        while shared._refresh_running and time.time() < deadline:
            time.sleep(0.005)
        self.assertFalse(shared._refresh_running)

    def test_lists_decision_models_with_input_modalities(self):
        body = {"data": [
            {"id": "b/decider", "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["decisions"]}},
            {"id": "a/decider-text", "architecture": {"input_modalities": ["text"], "output_modalities": ["decisions"]}},
        ]}
        release = threading.Event()

        def gated_get(*args, **kwargs):
            release.wait(5)
            return make_response(body=body)

        with patch.object(node_module.requests, "get", side_effect=gated_get) as get:
            self.assertEqual(Node.fetch_openrouter_models(), Node._fallback_models)
            release.set()
            self.wait_for_refresh()
            self.assertEqual(Node.fetch_openrouter_models(), ["a/decider-text", "b/decider"])
        self.assertEqual(get.call_count, 1)
        self.assertEqual(get.call_args.args[0], "https://openrouter.ai/api/v1/models")
        self.assertEqual(get.call_args.kwargs["params"], {"output_modalities": "decisions"})
        self.assertEqual(Node.model_input_modalities, {"b/decider": ["text", "image"], "a/decider-text": ["text"]})

    def test_fetch_failure_uses_fallback(self):
        with patch.object(node_module.requests, "get", side_effect=OSError("down")):
            Node.fetch_openrouter_models()
            self.wait_for_refresh()
            self.assertEqual(Node.fetch_openrouter_models(), Node._fallback_models)
            self.wait_for_refresh()


class ErrorTests(unittest.TestCase):
    def setUp(self):
        async def no_sleep(seconds):
            pass
        patcher = patch.object(shared.asyncio, "sleep", new=no_sleep)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_transient_errors_are_retried(self):
        for status in (429, 503):
            responses = [make_response(status, {"error": {"message": "busy"}}), make_response(body=answer_body(0.6))]
            with patch.object(shared.requests, "post", side_effect=responses) as post:
                self.assertEqual(decide()[1], 0.6)
            self.assertEqual(post.call_count, 2)

    def test_400_raises_with_message_without_retry(self):
        with patch.object(shared.requests, "post",
                          return_value=make_response(400, {"error": {"message": "bad question"}})) as post:
            with self.assertRaisesRegex(RuntimeError, "OpenRouter Decision error 400: bad question"):
                decide()
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
