import unittest

import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.landmark_extractor import LandmarkExtractor
from backend.app.orchestration import ai_orchestrator
from backend.app.services.sign_to_text import sign_to_text_service


class LandmarkContractTests(unittest.TestCase):
    def test_kinematics_preserves_expected_feature_contract(self):
        sequence = np.arange(4 * 190, dtype=np.float32).reshape(4, 190)

        extractor = LandmarkExtractor.__new__(LandmarkExtractor)
        result = extractor.compute_kinematics(sequence)

        self.assertEqual(result.shape, (4, 570))
        np.testing.assert_array_equal(result[:, :190], sequence)
        np.testing.assert_array_equal(result[0, 190:380], np.zeros(190, dtype=np.float32))
        np.testing.assert_array_equal(result[0:2, 380:], np.zeros((2, 190), dtype=np.float32))


class ModelReadinessTests(unittest.TestCase):
    def test_committed_recognition_checkpoints_load_with_runtime_contract(self):
        readiness = sign_to_text_service.readiness()

        self.assertTrue(readiness["fingerspelling"]["available"], readiness["fingerspelling"]["error"])
        self.assertTrue(readiness["word_recognition"]["available"], readiness["word_recognition"]["error"])
        self.assertFalse(readiness["sentence_recognition"]["available"])

    def test_readiness_endpoint_matches_runtime_state(self):
        with TestClient(app) as client:
            response = client.get("/health/ready")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["models"]["word_recognition"]["available"])
        self.assertFalse(body["models"]["sentence_recognition"]["available"])

    def test_orchestrator_records_real_operation_metrics(self):
        result = ai_orchestrator.translate_sign_sequence(gloss_sequence=["hello"], target_lang="en")

        self.assertEqual(result["english_text"], "Hello")
        diagnostics = ai_orchestrator.diagnostics()
        self.assertGreaterEqual(diagnostics["operations"]["sign_sequence_translation"]["samples"], 1)

        with TestClient(app) as client:
            response = client.get("/api/diagnostics")

        self.assertEqual(response.status_code, 200)
        self.assertIn("operations", response.json())


if __name__ == "__main__":
    unittest.main()
