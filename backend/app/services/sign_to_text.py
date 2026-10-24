import logging
import json
from dataclasses import asdict, dataclass
import numpy as np
from typing import Dict, Any

from backend.app.config import FINGERSPELLING_MODEL_PATH, WORD_MODEL_PATH, WORD_LABELS_PATH, SENTENCE_MODEL_PATH

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionCandidate:
    label: str
    confidence: float


@dataclass(frozen=True)
class RecognitionPrediction:
    """A model probability, not a calibrated measure of real-world accuracy."""

    label: str
    confidence: float
    accepted: bool
    threshold: float
    alternatives: tuple[PredictionCandidate, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelUnavailableError(RuntimeError):
    """Raised when an inference request requires an unavailable model."""


class RecognitionInferenceError(RuntimeError):
    """Raised when a loaded model cannot process a valid inference request."""


class SignToTextService:
    def __init__(self):
        self.device = None
        
        # Models
        self.fingerspell_model = None
        self.word_model = None
        self.sentence_model = None
        self.fingerspell_model_load_error: str | None = None
        self.word_model_load_error: str | None = None
        self.sentence_model_load_error: str | None = None
        
        # Labels
        self.word_labels = []
        self.fingerspell_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
        self.fingerspell_confidence_threshold = 0.75
        self.word_confidence_threshold = 0.40
        
        # Load models asynchronously to avoid blocking FastApi startup (~5.4s saved)
        import threading
        threading.Thread(target=self.load_models, daemon=True).start()

    def load_models(self):
        import torch
        from backend.app.models.fingerspell_model import FingerspellCNN
        from backend.app.models.word_model import WordTransformer
        
        if self.device is None:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
            logger.info(f"SignToTextService using device: {self.device}")
        # 1. Fingerspelling model
        if FINGERSPELLING_MODEL_PATH.exists():
            try:
                self.fingerspell_model = FingerspellCNN(num_classes=29)
                self.fingerspell_model.load_state_dict(torch.load(FINGERSPELLING_MODEL_PATH, map_location=self.device))
                self.fingerspell_model.to(self.device)
                self.fingerspell_model.eval()
                logger.info("Loaded fingerspelling CNN model.")
            except Exception as e:
                self.fingerspell_model_load_error = str(e)
                logger.exception("Failed to load fingerspelling model.")
        else:
            self.fingerspell_model_load_error = f"Checkpoint not found: {FINGERSPELLING_MODEL_PATH}"
            logger.warning(self.fingerspell_model_load_error)

        # 2. Word model
        if WORD_LABELS_PATH.exists() and WORD_MODEL_PATH.exists():
            try:
                with open(WORD_LABELS_PATH, 'r', encoding='utf-8') as f:
                    self.word_labels = json.load(f)
                
                # The committed checkpoint was trained with four attention heads and a
                # 512-wide feed-forward layer. These values must remain coupled to the
                # checkpoint; the WordTransformer defaults are intentionally larger.
                self.word_model = WordTransformer(
                    num_classes=len(self.word_labels),
                    d_model=256,
                    nhead=4,
                    num_layers=4,
                    dim_feedforward=512,
                    dropout=0.3,
                )
                self.word_model.load_state_dict(torch.load(WORD_MODEL_PATH, map_location=self.device))
                self.word_model.to(self.device)
                self.word_model.eval()
                logger.info(f"Loaded word recognition model with {len(self.word_labels)} classes.")
            except Exception as e:
                self.word_model = None
                self.word_model_load_error = str(e)
                logger.exception("Failed to load word recognition model.")
        else:
            self.word_model_load_error = "Word checkpoint or label manifest is missing."
            logger.warning(self.word_model_load_error)

        # 3. Sentence model
        if SENTENCE_MODEL_PATH.exists() and WORD_LABELS_PATH.exists():
            try:
                # We reuse word_labels as the vocabulary for the sentence model
                from backend.app.models.sentence_model import SentenceTransformer
                # +1 for CTC blank token (we will put it at index 0)
                num_sentence_classes = len(self.word_labels) + 1
                
                self.sentence_model = SentenceTransformer(
                    num_classes=num_sentence_classes,
                    d_model=256,
                    nhead=4,
                    num_layers=4,
                    dim_feedforward=512,
                    dropout=0.3
                )
                self.sentence_model.load_state_dict(torch.load(SENTENCE_MODEL_PATH, map_location=self.device))
                self.sentence_model.to(self.device)
                self.sentence_model.eval()
                logger.info(f"Loaded sentence recognition model with {num_sentence_classes} classes (including CTC blank).")
            except Exception as e:
                self.sentence_model = None
                self.sentence_model_load_error = str(e)
                logger.exception("Failed to load sentence recognition model.")
        else:
            self.sentence_model_load_error = "Sentence checkpoint or vocabulary is missing."
            logger.warning(self.sentence_model_load_error)

    @staticmethod
    def _prediction_from_logits(
        logits: Any,
        labels: list[str],
        threshold: float,
    ) -> RecognitionPrediction:
        import torch
        probabilities = torch.softmax(logits, dim=1)[0]
        candidate_count = min(3, len(labels))
        top_probabilities, top_indices = torch.topk(probabilities, k=candidate_count)
        alternatives = tuple(
            PredictionCandidate(label=labels[index.item()], confidence=round(probability.item(), 6))
            for probability, index in zip(top_probabilities, top_indices, strict=True)
        )
        best = alternatives[0]
        
        is_accepted = best.confidence >= threshold
        if best.label.lower() in {"del", "nothing"}:
            is_accepted = best.confidence >= 0.85

        # For ambiguous close candidates (e.g. C vs O), require a distinct margin over #2
        if is_accepted and len(alternatives) > 1 and best.confidence < 0.65:
            margin = best.confidence - alternatives[1].confidence
            if margin < 0.05:
                is_accepted = False

        return RecognitionPrediction(
            label=best.label,
            confidence=best.confidence,
            accepted=is_accepted,
            threshold=threshold,
            alternatives=alternatives,
        )

    def predict_fingerspell(self, frame_image: np.ndarray) -> RecognitionPrediction:
        """
        Takes a raw image frame, runs CNN classifier, returns letter/symbol.
        """
        if self.fingerspell_model is None:
            raise ModelUnavailableError(
                f"Fingerspelling model is unavailable. {self.fingerspell_model_load_error or ''}".strip()
            )

        try:
            if not isinstance(frame_image, np.ndarray) or frame_image.ndim != 3 or frame_image.shape[2] != 3:
                raise ValueError("Expected a three-channel BGR image.")

            # Match the RGB ImageNet preprocessing used by train_fingerspell.py.
            import cv2
            img = cv2.cvtColor(cv2.resize(frame_image, (224, 224)), cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                [0.229, 0.224, 0.225], dtype=np.float32
            )
            import torch
            x = torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.fingerspell_model(x)
                return self._prediction_from_logits(
                    logits,
                    self.fingerspell_labels,
                    self.fingerspell_confidence_threshold,
                )
        except Exception as e:
            logger.exception("Fingerspelling prediction failed.")
            raise RecognitionInferenceError("Fingerspelling inference failed.") from e

    def predict_word(self, landmark_sequence: np.ndarray) -> RecognitionPrediction:
        """
        Takes a sequence of normalized landmark features shaped (30, 570) and
        returns the recognized word.
        """
        if self.word_model is None:
            raise ModelUnavailableError(
                f"Word recognition model is unavailable. {self.word_model_load_error or ''}".strip()
            )

        try:
            landmark_sequence = np.asarray(landmark_sequence, dtype=np.float32)
            if landmark_sequence.ndim != 2:
                raise ValueError("Expected landmarks with shape [frames, features].")
            if landmark_sequence.shape[0] < 2:
                raise ValueError("At least two landmark frames are required for word recognition.")
            if landmark_sequence.shape[1] != 570:
                raise ValueError(
                    "Word recognition expects 570 features per frame "
                    "(190 landmarks plus velocity and acceleration)."
                )
            if not np.isfinite(landmark_sequence).all():
                landmark_sequence = np.nan_to_num(landmark_sequence, nan=0.0, posinf=0.0, neginf=0.0)

            # Motion check: ensure active hand landmark motion is present across frames
            # Features 24..150 contain left (21*3=63) and right (21*3=63) hand 3D coordinates
            hand_slice = landmark_sequence[:, 24:150]
            hand_motion_range = np.max(hand_slice, axis=0) - np.min(hand_slice, axis=0)
            if np.max(hand_motion_range) < 0.008:
                # Hand is idle or stationary — reject false positive
                return RecognitionPrediction(
                    label="nothing",
                    confidence=0.0,
                    accepted=False,
                    threshold=self.word_confidence_threshold,
                    alternatives=(),
                )

            # Reshape/pad sequence if necessary to match model's expected temporal size
            if landmark_sequence.shape[0] != 30:
                source_positions = np.arange(landmark_sequence.shape[0], dtype=np.float32)
                target_positions = np.linspace(0, landmark_sequence.shape[0] - 1, 30, dtype=np.float32)
                landmark_sequence = np.stack(
                    [np.interp(target_positions, source_positions, landmark_sequence[:, i]) for i in range(570)],
                    axis=1,
                ).astype(np.float32)

            import torch
            x = torch.tensor(landmark_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.word_model(x)
                return self._prediction_from_logits(
                    logits,
                    self.word_labels,
                    self.word_confidence_threshold,
                )
        except Exception as e:
            logger.exception("Word recognition prediction failed.")
            if isinstance(e, ValueError):
                raise
            raise RecognitionInferenceError("Word recognition inference failed.") from e

    def readiness(self) -> Dict[str, Any]:
        """Return model availability without exposing internals to a prediction response."""
        return {
            "fingerspelling": {
                "available": self.fingerspell_model is not None,
                "error": self.fingerspell_model_load_error,
            },
            "word_recognition": {
                "available": self.word_model is not None,
                "error": self.word_model_load_error,
            },
            "sentence_recognition": {
                "available": False,
                "error": self.sentence_model_load_error or "No validated sentence model is configured.",
            },
        }

sign_to_text_service = SignToTextService()
