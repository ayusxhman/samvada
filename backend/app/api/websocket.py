import base64
import binascii
import cv2
import json
import logging
import numpy as np
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from collections import deque
from backend.app.models.landmark_extractor import landmark_extractor
from backend.app.services.landmark_stabilizer import landmark_stabilizer
from backend.app.services.sign_to_text import ModelUnavailableError
from backend.app.orchestration import ai_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["Real-time Recognition"])


def _build_190_feature_vector(lh_raw, rh_raw, pose_raw):
    """
    Build the exact 190-dim feature vector that matches the trained word model's
    input contract: 8 pose points (3D) + 21 left hand (3D) + 21 right hand (3D) + 20 face (2D zeros).
    
    Total: 8*3 + 21*3 + 21*3 + 20*2 = 24 + 63 + 63 + 40 = 190
    
    Applies torso-centered normalization using shoulder midpoint and width.
    """
    pose_feats = np.zeros((8, 3), dtype=np.float32)
    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)
    face_feats = np.zeros((20, 2), dtype=np.float32)

    # Pose: extract the 8 upper-body indices [11,12,13,14,15,16,23,24]
    pose_indices = [11, 12, 13, 14, 15, 16, 23, 24]
    shoulder_mid = np.zeros(3, dtype=np.float32)
    shoulder_width = 1.0

    if len(pose_raw) >= 25:
        pose_arr = np.array(pose_raw, dtype=np.float32)
        left_shoulder = pose_arr[11]
        right_shoulder = pose_arr[12]
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
        if shoulder_width < 1e-5:
            shoulder_width = 1.0
        for i, idx in enumerate(pose_indices):
            if idx < len(pose_arr):
                pose_feats[i] = pose_arr[idx]

    # Hands
    if len(lh_raw) == 21:
        left_hand = np.array(lh_raw, dtype=np.float32)
    if len(rh_raw) == 21:
        right_hand = np.array(rh_raw, dtype=np.float32)

    # Torso-centered normalization
    def normalize_3d(arr):
        mask = np.any(arr != 0, axis=1)
        arr[mask] = (arr[mask] - shoulder_mid) / shoulder_width
        return arr

    pose_feats = normalize_3d(pose_feats)
    left_hand = normalize_3d(left_hand)
    right_hand = normalize_3d(right_hand)

    # Zero out non-nose pose to match training pipeline
    pose_feats[1:] = 0.0

    features = np.concatenate([
        pose_feats.flatten(),    # 8*3 = 24
        left_hand.flatten(),     # 21*3 = 63
        right_hand.flatten(),    # 21*3 = 63
        face_feats.flatten()     # 20*2 = 40
    ])
    return features  # (190,)


@router.websocket("/recognize")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Real-time webcam WebSocket connected.")
    
    # Rolling buffer of 190-dim landmark features (before kinematics)
    raw_landmark_buffer = deque(maxlen=60)
    recognized_glosses = deque(maxlen=32)
    last_fingerspell_char = None
    fingerspell_streak = 0
    last_emitted_char = None
    last_emit_time = 0.0
    frames_since_last_prediction = 0
    last_pred_time = 0.0
    last_mode = None
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Message must be valid JSON."})
                continue
            
            msg_type = message.get("type", "frame")
            target_lang = message.get("target_lang", "en")
            mode = message.get("mode", "word")  # 'word' or 'fingerspell'
            
            # Wipe state on mode switch to prevent stale buffer/word leakage
            if mode != last_mode:
                last_mode = mode
                raw_landmark_buffer.clear()
                recognized_glosses.clear()
                last_fingerspell_char = None
                fingerspell_streak = 0
                last_emitted_char = None
                last_emit_time = 0.0
                frames_since_last_prediction = 0
                last_pred_time = 0.0
            
            logger.info(f"[WS-DEBUG] Received msg_type={msg_type}, mode={mode}, keys={list(message.keys())}")
            
            start_time = time.time()
            lh_raw = []
            rh_raw = []
            pose_raw = []
            frame = None
            
            if msg_type == "landmarks":
                # Client-side WebGL pre-extracted 3D landmarks
                lh_raw = message.get("left_hand", [])
                rh_raw = message.get("right_hand", [])
                pose_raw = message.get("pose", [])
                logger.info(f"[WS-DEBUG] Landmarks payload: lh_len={len(lh_raw)}, rh_len={len(rh_raw)}, pose_len={len(pose_raw)}")
            else:
                # Base64 JPEG frame payload
                frame_data = message.get("frame")
                if not frame_data:
                    continue
                if "," in frame_data:
                    frame_data = frame_data.split(",")[1]
                try:
                    image_bytes = base64.b64decode(frame_data, validate=True)
                    np_arr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                except (ValueError, binascii.Error):
                    await websocket.send_json({"error": "Frame must be valid base64 image data."})
                    continue
                
                if frame is None:
                    await websocket.send_json({"error": "Failed to decode frame"})
                    continue
                    
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                landmark_extractor._initialize_models()
                hand_result = landmark_extractor.hand_landmarker.detect(mp_image)
                pose_result = landmark_extractor.pose_landmarker.detect(mp_image)

                if hand_result.hand_landmarks and hand_result.handedness:
                    for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                        label = handedness[0].category_name.lower()
                        lms_3d = [[lm.x, lm.y, lm.z] for lm in hand_lms]
                        if label == "left":
                            lh_raw = lms_3d
                        else:
                            rh_raw = lms_3d

                if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                    pose_raw = [[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]]

            has_lh = len(lh_raw) == 21
            has_rh = len(rh_raw) == 21
            has_pose = len(pose_raw) > 0
            has_body_or_hand = has_lh or has_rh or has_pose

            # If server-side detection was run, return diagnostic landmarks to client
            if msg_type != "landmarks":
                await websocket.send_json({
                    "type": "landmarks",
                    "has_lh": has_lh,
                    "has_rh": has_rh,
                    "left_hand": [[p[0], p[1]] for p in lh_raw],
                    "right_hand": [[p[0], p[1]] for p in rh_raw],
                    "detector_latency_ms": round((time.time() - start_time) * 1000, 2)
                })

            # Skip recognition if no pose or hands detected at all
            if not has_body_or_hand:
                logger.debug(f"[WS-DEBUG] No body/hand detected. has_lh={has_lh}, has_rh={has_rh}, has_pose={has_pose}")
                continue
            
            # Build the 190-dim feature vector matching the trained model contract
            if frame is not None:
                features_190 = landmark_extractor.extract_landmarks_from_frame(frame)
            else:
                features_190 = _build_190_feature_vector(lh_raw, rh_raw, pose_raw)

            raw_landmark_buffer.append(features_190)
            
            # Mode Classification
            if mode == "fingerspell":
                if not has_lh and not has_rh:
                    continue
                if frame is not None:
                    # Crop the frame around the detected hand to prevent CNN from guessing on full image
                    if 'hand_result' in locals() and hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
                        h, w, _ = frame.shape
                        target_lms = hand_result.hand_landmarks[0]
                        # Prefer the right hand for fingerspelling if both are present
                        if len(hand_result.hand_landmarks) > 1:
                            for idx, handedness in enumerate(hand_result.handedness):
                                if handedness[0].category_name.lower() == "right":
                                    target_lms = hand_result.hand_landmarks[idx]
                                    break
                                    
                        px_xs = [lm.x * w for lm in target_lms]
                        px_ys = [lm.y * h for lm in target_lms]
                        
                        min_x, max_x = min(px_xs), max(px_xs)
                        min_y, max_y = min(px_ys), max(px_ys)
                        
                        center_x = (min_x + max_x) / 2
                        center_y = (min_y + max_y) / 2
                        
                        # Use the larger pixel dimension for a square crop, plus 20% padding
                        box_size = max(max_x - min_x, max_y - min_y) * 1.3
                        
                        x1 = max(0, int(center_x - box_size / 2))
                        x2 = min(w, int(center_x + box_size / 2))
                        y1 = max(0, int(center_y - box_size / 2))
                        y2 = min(h, int(center_y + box_size / 2))
                        
                        cropped_hand = frame[y1:y2, x1:x2]
                        if cropped_hand.size > 0:
                            frame = cropped_hand

                    # Image-based CNN classification (EfficientNet-B0)
                    try:
                        prediction = ai_orchestrator.recognize_fingerspell(frame)
                    except ModelUnavailableError as e:
                        logger.warning(f"[WS-DEBUG] Fingerspell model unavailable: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"[WS-DEBUG] Fingerspell CNN exception: {type(e).__name__}: {e}")
                        continue
                else:
                    logger.debug("[WS-DEBUG] Fingerspell mode but no frame — skipping")
                    continue

                if not prediction.accepted:
                    await websocket.send_json({
                        "type": "uncertain",
                        "confidence": prediction.confidence,
                        "confidence_threshold": prediction.threshold,
                        "alternatives": [candidate.__dict__ for candidate in prediction.alternatives],
                    })
                    continue

                pred_char = prediction.label.upper()
                
                # Ignore non-letter noise outputs ('NOTHING', 'DEL')
                if pred_char in {"NOTHING", "DEL"}:
                    last_fingerspell_char = None
                    fingerspell_streak = 0
                    continue

                if pred_char == last_fingerspell_char:
                    fingerspell_streak += 1
                else:
                    last_fingerspell_char = pred_char
                    fingerspell_streak = 1

                # Confirm gesture with 6 consecutive matching frames (approx 200-300ms) to filter out CNN noise
                if fingerspell_streak >= 6:
                    current_time = time.time()
                    should_emit = False
                    
                    if pred_char != last_emitted_char:
                        # Shifted to a new gesture: emit immediately!
                        should_emit = True
                    elif current_time - last_emit_time >= 1.5:
                        # Held the same gesture for >= 1.5 seconds: emit repeat!
                        should_emit = True
                        
                    if should_emit:
                        last_emitted_char = pred_char
                        last_emit_time = current_time
                        await websocket.send_json({
                            "type": "fingerspell",
                            "prediction": pred_char
                        })
            elif mode == "word":
                # Word mode: sequence-based classification
                frames_since_last_prediction += 1
                
                # Trigger inference when buffer has enough frames (~1.2s of gesture motion)
                if len(raw_landmark_buffer) >= 8 and frames_since_last_prediction >= 4:
                    frames_since_last_prediction = 0
                    
                    # Build the 570-dim kinematic features (pos + vel + accel)
                    seq = np.array(list(raw_landmark_buffer))
                    features_570 = landmark_extractor.compute_kinematics(seq)
                    
                    try:
                        prediction = ai_orchestrator.recognize_word(features_570)
                        logger.info(f"[WS-DEBUG] Word prediction: {prediction.label} (accepted: {prediction.accepted}, confidence: {prediction.confidence:.2f})")
                    except ModelUnavailableError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Word recognition model is currently training or unavailable."
                        })
                        continue
                    except Exception as e:
                        logger.error(f"Word recognition failed: {e}")
                        continue

                    if not prediction.accepted:
                        await websocket.send_json({
                            "type": "uncertain",
                            "confidence": prediction.confidence,
                            "confidence_threshold": prediction.threshold,
                            "alternatives": [candidate.__dict__ for candidate in prediction.alternatives],
                        })
                        continue
                    
                    pred_word = prediction.label
                    current_time = time.time()
                    if not recognized_glosses or pred_word != recognized_glosses[-1] or (current_time - last_pred_time) >= 1.5:
                        recognized_glosses.append(pred_word)
                        last_pred_time = current_time
                        
                        translation_info = ai_orchestrator.translate_sign_sequence(
                            gloss_sequence=list(recognized_glosses),
                            target_lang=target_lang
                        )
                        
                        await websocket.send_json({
                            "type": "word",
                            "prediction": pred_word,
                            "confidence": prediction.confidence,
                            "glosses": list(recognized_glosses),
                            "english_text": translation_info.get("english_text", ""),
                            "translated_text": translation_info.get("translated_text", ""),
                            "translation": translation_info.get("translated_text", ""),
                            "source_lang": translation_info.get("source_lang", "en"),
                            "target_lang": translation_info.get("target_lang", target_lang)
                        })
                        # Clear buffer after successful prediction so next word gesture starts fresh
                        raw_landmark_buffer.clear()
            elif mode == "sentence":
                # Sentence mode: continuous sequence-based classification
                frames_since_last_prediction += 1
                
                # For sentences, we accumulate more frames before running inference (e.g., 60 frames = 2 seconds of continuous signing)
                if len(raw_landmark_buffer) >= 60 and frames_since_last_prediction >= 20:
                    frames_since_last_prediction = 0
                    
                    seq = np.array(list(raw_landmark_buffer))
                    features = landmark_extractor.compute_kinematics(seq)
                    
                    try:
                        prediction = ai_orchestrator.recognize_sentence(features)
                        logger.info(f"[WS-DEBUG] Sentence prediction: {prediction.label}")
                    except ModelUnavailableError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Sentence recognition model is currently training or unavailable."
                        })
                        continue
                    except Exception as e:
                        logger.error(f"Sentence recognition failed: {e}")
                        continue

                    if prediction.accepted:
                        pred_sentence = prediction.label
                        # Instead of appending glosses, sentence mode predicts full sentences
                        await websocket.send_json({
                            "type": "sentence",
                            "prediction": pred_sentence,
                            "confidence": prediction.confidence,
                            "english_text": pred_sentence,
                            "translated_text": translation_service.translate(pred_sentence, target_lang, "en"),
                            "target_lang": target_lang
                        })
                        raw_landmark_buffer.clear()
    except WebSocketDisconnect:
        logger.info("Real-time webcam WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
