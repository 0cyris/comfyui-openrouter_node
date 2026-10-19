import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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
        answer, value, passed, details, cost = run_with(body, question_type="choice", criteria="left\nright")[0]
        self.assertEqual((answer, value, passed, cost), ("left", 0.7, True, 0.00001))
        self.assertEqual(json.loads(details)["probabilities"], {"left": 0.7, "right": 0.3})

    def test_below_threshold_not_passed(self):
        body = choice_body("left", 0.4, {"left": 0.4, "right": 0.3})
        answer, value, passed, _, _ = run_with(body, question_type="choice", criteria="left\nright")[0]
        self.assertEqual((answer, value, passed), ("left", 0.4, False))


class ScoreOutputTests(unittest.TestCase):
    legend = {"0": "bad", "1": "ok", "2": "good", "3": "great"}

    def run_score(self, score, **overrides):
        return run_with(score_body(score, self.legend), question_type="score", criteria="bad\nok\ngood\ngreat",
                        **overrides)[0]

    def test_normalises_score_by_level_count(self):
        answer, value, passed, details, _ = self.run_score(2.0)
        self.assertEqual((answer, value, passed), ("good", 2 / 3, True))
        self.assertEqual(json.loads(details)["score"], 2.0)

    def test_answer_is_legend_text_of_nearest_level(self):
        self.assertEqual(self.run_score(1.4)[0], "ok")
        self.assertEqual(self.run_score(1.6)[0], "good")
        self.assertEqual(self.run_score(2.99)[0], "great")

    def test_threshold_boundary_is_inclusive(self):
        self.assertTrue(self.run_score(1.5, threshold=0.5)[2])
        self.assertFalse(self.run_score(1.4, threshold=0.5)[2])

    def test_single_level_gives_zero(self):
        body = score_body(0.0, {"0": "only"})
        answer, value, passed, _, _ = run_with(body, question_type="score", criteria="only")[0]
        self.assertEqual((answer, value, passed), ("only", 0.0, False))


class InvalidCriteriaTests(unittest.TestCase):
    def assert_rejected(self, pattern, **overrides):
        with patch.object(shared.requests, "post") as post:
            with self.assertRaisesRegex(RuntimeError, pattern):
                decide(**overrides)
        post.assert_not_called()

    def test_choice_without_options(self):
        self.assert_rejected("at least one option", question_type="choice", criteria=" \n\n")

    def test_choice_duplicate_names(self):
        self.assert_rejected("Duplicate option 'a'", question_type="choice", criteria="a: x\nb\na: y")

    def test_score_without_levels(self):
        self.assert_rejected("at least one level", question_type="score", criteria="\n \n")


class OutputTests(unittest.TestCase):
    def test_above_threshold_is_true(self):
        answer, value, passed, details, cost = run_with(answer_body(0.8))[0]
        self.assertEqual((answer, value, passed), ("true", 0.8, True))
        self.assertEqual(json.loads(details), {"type": "noul", "noul": 0.8})

    def test_below_threshold_is_false(self):
        answer, value, passed, _, _ = run_with(answer_body(0.2))[0]
        self.assertEqual((answer, value, passed), ("false", 0.2, False))

    def test_threshold_boundary_is_inclusive(self):
        answer, value, passed, _, _ = run_with(answer_body(0.75), threshold=0.75)[0]
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

    def test_lists_decision_models_with_input_modalities(self):
        body = {"data": [
            {"id": "b/decider", "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["decisions"]}},
            {"id": "a/decider-text", "architecture": {"input_modalities": ["text"], "output_modalities": ["decisions"]}},
        ]}
        with patch.object(node_module.requests, "get", return_value=make_response(body=body)) as get:
            models = Node.fetch_openrouter_models()
        self.assertEqual(models, ["a/decider-text", "b/decider"])
        self.assertEqual(get.call_args.args[0], "https://openrouter.ai/api/v1/models")
        self.assertEqual(get.call_args.kwargs["params"], {"output_modalities": "decisions"})
        self.assertEqual(Node.model_input_modalities, {"b/decider": ["text", "image"], "a/decider-text": ["text"]})

    def test_fetch_failure_uses_fallback(self):
        with patch.object(node_module.requests, "get", side_effect=OSError("down")):
            self.assertEqual(Node.fetch_openrouter_models(), Node._fallback_models)


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
