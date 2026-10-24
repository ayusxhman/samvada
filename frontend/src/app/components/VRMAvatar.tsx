'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, type VRM } from '@pixiv/three-vrm';
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from '@pixiv/three-vrm-animation';

import { RotateCcw } from 'lucide-react';
import type { AnimationTimeline } from './animation/TimelinePlayer';
import { AIQuaternionProvider } from './animation/AnimationProvider';
import { AvatarBoneMapper, ALL_VRM_HUMANOID_BONES } from './animation/AvatarBoneMapper';
import { AvatarExpressionController } from './animation/AvatarExpressionController';

export interface VRMAAnimationInstruction {
  word: string;
  type: 'vrma' | 'text_hold';
  path: string;
  duration?: number;
}

interface VRMAvatarProps {
  animationTimeline?: AnimationTimeline;
  currentAnimation?: VRMAAnimationInstruction;
  playbackSpeed?: number;
  targetLanguage?: string;
  onAnimationFinished?: () => void;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';
const VRM_DEBUG = process.env.NEXT_PUBLIC_VRM_DEBUG === 'true';

function resolveAnimationUrl(path: string): string {
  return new URL(path, `${API_BASE_URL}/`).toString();
}

// Neural Male TTS Speech Helper supporting Hindi & English
function speakGreeting(lang?: string, onStart?: () => void, onEnd?: () => void): void {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
    onStart?.();
    setTimeout(() => onEnd?.(), 3200);
    return;
  }

  try {
    window.speechSynthesis.cancel();
    const isHindi = lang === 'hi' || lang === 'hin' || lang === 'hindi';
    const text = isHindi
      ? "नमस्ते! आपका स्वागत है। मैं आपकी सहायता के लिए तैयार हूँ।"
      : "Hello! Welcome back. I am ready to help you.";

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.92;
    utterance.pitch = 0.96;
    utterance.volume = 1.0;

    const voices = window.speechSynthesis.getVoices();
    let selectedVoice: SpeechSynthesisVoice | undefined;

    if (isHindi) {
      selectedVoice = voices.find(v => v.lang.includes('hi') && (v.name.includes('Male') || v.name.includes('Hemant') || v.name.includes('Madhur') || v.name.includes('Google')))
        || voices.find(v => v.lang.includes('hi'));
    }

    if (!selectedVoice) {
      selectedVoice = voices.find(v => (v.name.includes('Male') || v.name.includes('Guy') || v.name.includes('David') || v.name.includes('Natural')) && v.lang.startsWith('en'))
        || voices.find(v => v.lang.startsWith('en'));
    }

    if (selectedVoice) utterance.voice = selectedVoice;

    utterance.onstart = () => onStart?.();
    utterance.onend = () => onEnd?.();
    utterance.onerror = () => onEnd?.();

    window.speechSynthesis.speak(utterance);
  } catch {
    onEnd?.();
  }
}

// ============================================================================
// PRE-ALLOCATED MATH OBJECT POOL (Zero GC allocations inside requestAnimationFrame)
// ============================================================================
const TMP_EULER = new THREE.Euler(0, 0, 0, 'XYZ');
const TMP_QUAT_BREATH = new THREE.Quaternion();
const TMP_QUAT_OVERLAY = new THREE.Quaternion();
const TMP_QUAT_COMBINED = new THREE.Quaternion();
const TMP_VEC3_LOOKAT = new THREE.Vector3();

// Zero-allocation helper to convert Euler angles into a target Quaternion
function quatFromEuler(out: THREE.Quaternion, x: number, y: number, z: number): THREE.Quaternion {
  TMP_EULER.set(x, y, z, 'XYZ');
  return out.setFromEuler(TMP_EULER);
}

// Finger & Joint bone names constants for fast iterations
const FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'] as const;
const HAND_SIDES = ['left', 'right'] as const;
const PHASES = ['Proximal', 'Intermediate', 'Distal'] as const;

export default function VRMAvatar({
  animationTimeline,
  currentAnimation,
  playbackSpeed = 1.0,
  targetLanguage = 'en',
  onAnimationFinished
}: VRMAvatarProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const vrmRef = useRef<VRM | null>(null);
  const clockRef = useRef<THREE.Clock | null>(null);
  const mixerRef = useRef<THREE.AnimationMixer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const activeActionRef = useRef<THREE.AnimationAction | null>(null);
  const activeFinishHandlerRef = useRef<((event: { action: THREE.AnimationAction }) => void) | null>(null);
  const animationRequestRef = useRef(0);

  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [avatarReady, setAvatarReady] = useState(false);

  // State flags for life engine
  const isGreetingRef = useRef(false);
  const greetProgressRef = useRef(0);
  const isSpeakingRef = useRef(false);
  const isSigningRef = useRef(false);
  const isLeftRestingRef = useRef(true);

  // Subsystem Architectures
  const boneMapperRef = useRef<AvatarBoneMapper>(new AvatarBoneMapper());
  const expressionControllerRef = useRef<AvatarExpressionController>(new AvatarExpressionController());
  const aiProviderRef = useRef<AIQuaternionProvider>(new AIQuaternionProvider(playbackSpeed));

  const speedRef = useRef(playbackSpeed);
  const vrmaLoaderRef = useRef<GLTFLoader | null>(null);
  const animationCacheRef = useRef<Record<string, THREE.AnimationClip>>({});

  const handleResetCamera = React.useCallback(() => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(0.0, 1.25, 1.65);
      controlsRef.current.target.set(0.0, 1.15, 0.0);
      controlsRef.current.update();
    }
  }, []);

  // Pre-allocated map of target Quaternions for every mapped bone
  const targetBonesRef = useRef<Record<string, THREE.Quaternion>>({});

  useEffect(() => {
    speedRef.current = playbackSpeed;
    aiProviderRef.current.setSpeed(playbackSpeed);
  }, [playbackSpeed]);

  // Pre-initialize quaternion instances for all VRM humanoid bones
  useEffect(() => {
    ALL_VRM_HUMANOID_BONES.forEach(b => {
      if (!targetBonesRef.current[b]) {
        targetBonesRef.current[b] = new THREE.Quaternion();
      }
    });
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // 1. Setup Scene
    const scene = new THREE.Scene();

    // 2. Setup Camera (Front-facing default view)
    const camera = new THREE.PerspectiveCamera(
      32.0,
      container.clientWidth / container.clientHeight,
      0.1,
      20.0
    );
    camera.position.set(0.0, 1.25, 1.65);
    cameraRef.current = camera;

    // 3. Setup Renderer (High-Performance WebGL)
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
      precision: 'highp'
    });
    renderer.setClearColor(0x000000, 0);
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // 4. Setup Controls (Full 360 degree OrbitControls)
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0.0, 1.15, 0.0);
    controls.enablePan = true;
    controls.enableZoom = true;
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 0.4;
    controls.maxDistance = 4.5;
    controls.maxPolarAngle = Math.PI - 0.05;
    controls.minPolarAngle = 0.05;
    controls.update();
    controlsRef.current = controls;

    // 5. Lighting Setup
    const dirLight = new THREE.DirectionalLight('#ffffff', 1.3);
    dirLight.position.set(2.0, 3.0, 2.0);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    dirLight.shadow.bias = -0.001;
    scene.add(dirLight);

    const hemiLight = new THREE.HemisphereLight('#ffffff', '#a8b0c4', 0.85);
    scene.add(hemiLight);

    // 6. Floor Shadow Receiver
    const planeGeo = new THREE.PlaneGeometry(10, 10);
    const shadowMat = new THREE.ShadowMaterial({ opacity: 0.08 });
    const floor = new THREE.Mesh(planeGeo, shadowMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // 7. Setup VRM LookAt Target (Positioned directly in front of camera)
    const lookAtTarget = new THREE.Object3D();
    lookAtTarget.position.set(0, 1.25, 3.0);
    scene.add(lookAtTarget);

    // 8. Load eric1.vrm Model
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    loader.register((parser) => new VRMAnimationLoaderPlugin(parser));
    vrmaLoaderRef.current = loader;

    clockRef.current = new THREE.Clock();

    let idleBehaviorTimer = 0;
    let nextIdleBehaviorTime = Math.random() * 15 + 15;
    const microSaccadeOffset = new THREE.Vector2(0, 0);
    let animationFrameCount = 0;
    let mediaTimelineFrameCount = 0;

    loader.load(
      '/eric1.vrm',
      (gltf) => {
        const vrm = gltf.userData.vrm as VRM;
        vrmRef.current = vrm;
        mixerRef.current = new THREE.AnimationMixer(vrm.scene);

        // Initialize Bone Mapper & Expression Controller
        const validation = boneMapperRef.current.initialize(vrm);
        expressionControllerRef.current.initialize(vrm);

        console.info(`[eric1.vrm] Successfully initialized. Mapped ${validation.totalMapped}/${validation.totalRequired} VRM humanoid bones.`);
        if (VRM_DEBUG) {
          console.info('[VRM debug] bone mapper validation', validation);
        }

        vrm.scene.traverse((obj: THREE.Object3D) => {
          obj.frustumCulled = false;
          if (obj instanceof THREE.Mesh) {
            obj.castShadow = true;
            obj.receiveShadow = true;
          }
        });

        // Face front towards camera (+Z)
        vrm.scene.rotation.y = 0;
        scene.add(vrm.scene);

        if (vrm.lookAt) {
          vrm.lookAt.target = lookAtTarget;
        }

        // Initialize ALL target bones with the natural idle pose BEFORE first render
        applyNaturalIdlePose(targetBonesRef.current, isLeftRestingRef.current);

        // Force bones instantly on load to eliminate T-Pose completely
        ALL_VRM_HUMANOID_BONES.forEach(boneName => {
          const boneNode = boneMapperRef.current.getBoneNode(boneName);
          if (boneNode && targetBonesRef.current[boneName]) {
            boneMapperRef.current.applyBoneQuaternion(
              boneName,
              targetBonesRef.current[boneName],
              TMP_QUAT_COMBINED
            );
          }
        });

        // Compute initial VRM matrix transform on natural idle pose
        vrm.update(0.0);
        renderer.render(scene, camera);

        // Trigger smooth greeting wave sequence on load
        isGreetingRef.current = true;
        greetProgressRef.current = 0;

        // Speak warm welcome message via Neural Male TTS
        speakGreeting(
          targetLanguage,
          () => { isSpeakingRef.current = true; },
          () => { isSpeakingRef.current = false; }
        );

        // Reveal avatar only AFTER full initialization and 1st frame render
        requestAnimationFrame(() => {
          setLoading(false);
          setAvatarReady(true);
        });
      },
      undefined,
      (error) => {
        console.error('Error loading eric1.vrm:', error);
        setLoadError('Unable to load 3D avatar eric1.vrm');
        setLoading(false);
      }
    );

    // ========================================================================
    // UNIFIED 60 FPS PRODUCTION ANIMATION RENDER LOOP
    // ========================================================================
    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      const rawDelta = clockRef.current ? clockRef.current.getDelta() : 0.0166;
      const delta = Math.min(rawDelta, 0.1) * speedRef.current;
      const elapsedTime = clockRef.current ? clockRef.current.getElapsedTime() : 0;

      if (vrmRef.current) {
        const vrm = vrmRef.current;

        // --------------------------------------------------------------------
        // PASS 1: FULL ARM ELEVATION LOGIN GREETING WAVE
        // The greeting BLENDS from the natural idle pose to the wave pose.
        // w=0 means full idle (arms at sides), w=1 means full wave.
        // This guarantees the avatar NEVER passes through T-pose.
        // --------------------------------------------------------------------
        if (isGreetingRef.current) {
          greetProgressRef.current += delta;
          const greetDuration = 3.6;
          const progress = Math.min(1.0, greetProgressRef.current / greetDuration);
          const w = Math.sin(progress * Math.PI); // 0 → 1 → 0 bell curve

          const smileVal = 0.45 * w;
          expressionControllerRef.current.setExpression('happy', smileVal);
          expressionControllerRef.current.setExpression('smile', smileVal);
          expressionControllerRef.current.setExpression('relaxed', smileVal * 0.5);

          if (progress < 1.0) {
            const waveOscillation = Math.sin(elapsedTime * 6.0) * 0.35 * w;

            // IDLE values (arms hanging at sides) → WAVE values (arm raised)
            // lerp(idle, wave, w) so at w=0 we get idle, at w=1 we get wave
            const lerp = (idle: number, wave: number) => idle + (wave - idle) * w;

            // Right shoulder: idle (-0.06 Z) → wave (-0.22 Z)
            setTargetBoneQuat('rightShoulder', lerp(0, 0.05), 0, lerp(-0.06, -0.22));
            // Right upper arm: idle (0.12, 0.08, -1.38) → wave raised (0.65, -0.20, -0.38)
            setTargetBoneQuat('rightUpperArm', lerp(0.12, 0.65), lerp(0.08, -0.20), lerp(-1.38, -0.38));
            // Right lower arm: idle (-0.25, -0.10, 0) → wave bent (-1.25, 0.35, 0)
            setTargetBoneQuat('rightLowerArm', lerp(-0.25, -1.25), lerp(-0.10, 0.35), 0);
            // Right hand: idle relaxed + wave oscillation overlay
            setTargetBoneQuat('rightHand', -0.10, 0, lerp(0.05, waveOscillation));

            // Right fingers: idle curled → greeting open
            FINGERS.forEach(finger => {
              PHASES.forEach((phase, idx) => {
                const idleCurl = finger === 'Thumb' ? 0.1 : (idx === 0 ? -0.30 : (idx === 1 ? -0.45 : -0.25));
                setTargetBoneQuat(`right${finger}${phase}`, lerp(idleCurl, -0.12), 0, 0);
              });
            });

            // Gentle head nod
            const headNod = Math.sin(elapsedTime * 3.0) * 0.06 * w;
            setTargetBoneQuat('neck', 0.02 + headNod, 0.02, 0.02);
            setTargetBoneQuat('spine', 0.03, 0, 0);
          } else {
            isGreetingRef.current = false;
            applyNaturalIdlePose(targetBonesRef.current, isLeftRestingRef.current);
            expressionControllerRef.current.resetToNeutral();
          }
        }

        // --------------------------------------------------------------------
        // PASS 2: IDLE BREATHING, SWAY & LIP-SYNC
        // --------------------------------------------------------------------
        if (!isSigningRef.current) {
          idleBehaviorTimer += delta;
          if (idleBehaviorTimer > nextIdleBehaviorTime) {
            idleBehaviorTimer = 0;
            nextIdleBehaviorTime = Math.random() * 15 + 15;
            triggerRandomIdleBehavior();
          }

          if (Math.random() < 0.01) {
            microSaccadeOffset.x = (Math.random() - 0.5) * 0.08;
            microSaccadeOffset.y = (Math.random() - 0.5) * 0.08;
          } else {
            microSaccadeOffset.x *= 0.95;
            microSaccadeOffset.y *= 0.95;
          }
        } else {
          idleBehaviorTimer = 0;
          microSaccadeOffset.x = 0;
          microSaccadeOffset.y = 0;
        }

        // Lip-sync viseme animation during speech
        expressionControllerRef.current.updateSpeechVisemes(elapsedTime, isSpeakingRef.current);

        // --------------------------------------------------------------------
        // PASS 3: BLINKING & FACIAL EXPRESSIONS UPDATE
        // --------------------------------------------------------------------
        expressionControllerRef.current.update(delta);;

        // --------------------------------------------------------------------
        // PASS 4: SINGLE UNIFIED EXPONENTIAL SLERP BONE UPDATE
        // --------------------------------------------------------------------
        const aiUpdate = aiProviderRef.current.update(delta);
        if (aiUpdate && aiUpdate.bones) {
          mediaTimelineFrameCount += 1;
          isSigningRef.current = !aiUpdate.finished;
          aiUpdate.bones.forEach((quat, boneName) => {
            if (!targetBonesRef.current[boneName]) {
              targetBonesRef.current[boneName] = new THREE.Quaternion();
            }
            targetBonesRef.current[boneName].copy(quat);
          });

          if (aiUpdate.blendshapes) {
            Object.entries(aiUpdate.blendshapes).forEach(([bsName, val]) => {
              expressionControllerRef.current.setExpression(bsName, val);
            });
          }

          if (VRM_DEBUG && mediaTimelineFrameCount % 60 === 0) {
            console.info('[VRM debug] timeline quaternion frame', {
              mediaTimelineFrameCount,
              boneCount: aiUpdate.bones.size,
              leftUpperArm: aiUpdate.bones.get('leftUpperArm')?.toArray(),
              rightUpperArm: aiUpdate.bones.get('rightUpperArm')?.toArray(),
            });
          }
        }

        if (!activeActionRef.current && boneMapperRef.current.ready()) {
          const baseRate = isGreetingRef.current ? 8.0 : 6.0;
          // Bypass double-smoothing lag during active signing: evaluate timeline quaternions directly (alpha = 1.0)
          const alpha = isSigningRef.current ? 1.0 : (1.0 - Math.exp(-baseRate * delta));

          const breathX = Math.sin(elapsedTime * 1.5) * 0.015;
          const breathZ = Math.sin(elapsedTime * 0.8) * 0.008;

          ALL_VRM_HUMANOID_BONES.forEach(boneName => {
            const boneNode = boneMapperRef.current.getBoneNode(boneName);
            if (boneNode) {
              const targetQuat = targetBonesRef.current[boneName];
              if (!targetQuat) return;

              if (!isSigningRef.current && boneName === 'chest') {
                quatFromEuler(TMP_QUAT_BREATH, breathX, 0, 0);
                TMP_QUAT_OVERLAY.multiplyQuaternions(targetQuat, TMP_QUAT_BREATH);
                boneMapperRef.current.applyBoneQuaternion(boneName, TMP_QUAT_OVERLAY, TMP_QUAT_COMBINED, alpha);
              } else if (!isSigningRef.current && boneName === 'spine') {
                quatFromEuler(TMP_QUAT_BREATH, 0, 0, breathZ);
                TMP_QUAT_OVERLAY.multiplyQuaternions(targetQuat, TMP_QUAT_BREATH);
                boneMapperRef.current.applyBoneQuaternion(boneName, TMP_QUAT_OVERLAY, TMP_QUAT_COMBINED, alpha);
              } else {
                boneMapperRef.current.applyBoneQuaternion(boneName, targetQuat, TMP_QUAT_COMBINED, alpha);
              }
            }
          });
        }

        // --------------------------------------------------------------------
        // PASS 5: SINGLE FINAL VRM SCENE UPDATE
        // Correct Order: 1. mixer.update() -> 2. vrm.update() -> 3. renderer.render()
        // --------------------------------------------------------------------
        if (mixerRef.current) mixerRef.current.update(delta);
        vrm.update(delta);

        if (VRM_DEBUG && ++animationFrameCount % 60 === 0) {
          const debugBone = (boneName: string) => {
            const bone = boneMapperRef.current.getBoneNode(boneName);
            if (!bone) return null;
            return {
              localQuaternion: bone.quaternion.toArray(),
              worldQuaternion: bone.getWorldQuaternion(new THREE.Quaternion()).toArray(),
            };
          };
          console.info('[VRM debug] render loop', {
            fps: rawDelta > 0 ? Math.round(1 / rawDelta) : 0,
            animationFrameCount,
            leftUpperArm: debugBone('leftUpperArm'),
            rightUpperArm: debugBone('rightUpperArm'),
          });
        }
      }

      renderer.render(scene, camera);
    };
    animate();

    // 9. Mouse Interaction (LookAt Target Smooth Lerp)
    let mouseX = 0, mouseY = 0;
    const handleMouseMove = (event: MouseEvent) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      mouseX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouseY = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    };
    window.addEventListener('mousemove', handleMouseMove);

    let lookAtAnimationFrame: number;
    const updateLookAt = () => {
      lookAtAnimationFrame = requestAnimationFrame(updateLookAt);
      if (!isSigningRef.current) {
        const targetX = mouseX * 2 + microSaccadeOffset.x;
        const targetY = 1.35 + mouseY * 2 + microSaccadeOffset.y;
        TMP_VEC3_LOOKAT.set(targetX, targetY, 1.0);
        lookAtTarget.position.lerp(TMP_VEC3_LOOKAT, 0.1);
      } else {
        TMP_VEC3_LOOKAT.set(0, 1.35, 1.0);
        lookAtTarget.position.lerp(TMP_VEC3_LOOKAT, 0.1);
      }
    };
    updateLookAt();

    // 10. Resize Handler
    const handleResize = () => {
      if (!containerRef.current) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationFrameId);
      cancelAnimationFrame(lookAtAnimationFrame);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      stopActiveAnimation();
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function setTargetBoneQuat(boneName: string, x: number, y: number, z: number): void {
    if (!targetBonesRef.current[boneName]) {
      targetBonesRef.current[boneName] = new THREE.Quaternion();
    }
    quatFromEuler(targetBonesRef.current[boneName], x, y, z);
  }

  // Primary AI Quaternion Timeline Playback (Single Source of Truth)
  useEffect(() => {
    if (!avatarReady || !animationTimeline) return;
    
    // Stop any legacy VRMA action to ensure ONE animation provider is active
    stopActiveAnimation();
    
    aiProviderRef.current.onFinished = () => {
      recoverToIdle();
      onAnimationFinished?.();
    };
    aiProviderRef.current.setSpeed(playbackSpeed);
    aiProviderRef.current.start(animationTimeline);
  }, [avatarReady, animationTimeline, playbackSpeed, onAnimationFinished]);

  // Legacy VRMA clip playback (ONLY used when NO quaternion timeline exists)
  useEffect(() => {
    if (!avatarReady || !currentAnimation || animationTimeline) return;
    playVRMA(currentAnimation);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [avatarReady, currentAnimation, animationTimeline]);

  function applyNaturalIdlePose(targetObj: Record<string, THREE.Quaternion>, isLeftResting: boolean): void {
    const setQuat = (boneName: string, x: number, y: number, z: number) => {
      if (!targetObj[boneName]) targetObj[boneName] = new THREE.Quaternion();
      quatFromEuler(targetObj[boneName], x, y, z);
    };

    // Relaxed shoulders & natural arm hanging posture
    setQuat('rightShoulder', 0, 0, -0.06);
    setQuat('leftShoulder', 0, 0, 0.06);
    setQuat('rightUpperArm', 0.12, 0.08, -1.38);
    setQuat('leftUpperArm', 0.12, -0.08, 1.38);
    setQuat('rightLowerArm', -0.25, -0.10, 0);
    setQuat('leftLowerArm', -0.25, 0.10, 0);
    setQuat('rightHand', -0.10, 0, 0.05);
    setQuat('leftHand', -0.10, 0, -0.05);

    // Natural human relaxed finger curl
    HAND_SIDES.forEach(side => {
      FINGERS.forEach(finger => {
        PHASES.forEach((phase, idx) => {
          const boneName = `${side}${finger}${phase}`;
          if (finger === 'Thumb') {
            setQuat(boneName, 0.1, side === 'left' ? 0.25 : -0.25, side === 'left' ? 0.2 : -0.2);
          } else {
            const curl = idx === 0 ? -0.30 : (idx === 1 ? -0.45 : -0.25);
            setQuat(boneName, curl, 0, 0);
          }
        });
      });
    });

    // Body balance & relaxed spine posture
    setQuat('spine', 0.03, 0, 0);
    setQuat('chest', 0.02, 0, 0);
    setQuat('neck', 0.02, 0, 0);
    setQuat('hips', 0, 0, isLeftResting ? 0.03 : -0.03);
    setQuat('rightUpperLeg', isLeftResting ? 0.06 : 0.02, 0, 0);
    setQuat('leftUpperLeg', isLeftResting ? 0.02 : 0.06, 0, 0);
    setQuat('rightLowerLeg', isLeftResting ? -0.06 : -0.02, 0, 0);
    setQuat('leftLowerLeg', isLeftResting ? -0.02 : -0.06, 0, 0);
  }

  function triggerRandomIdleBehavior(): void {
    isLeftRestingRef.current = !isLeftRestingRef.current;
    applyNaturalIdlePose(targetBonesRef.current, isLeftRestingRef.current);

    const setQuat = (boneName: string, x: number, y: number, z: number) => {
      if (!targetBonesRef.current[boneName]) targetBonesRef.current[boneName] = new THREE.Quaternion();
      quatFromEuler(targetBonesRef.current[boneName], x, y, z);
    };

    const randomAction = Math.floor(Math.random() * 3);
    if (randomAction === 0) {
      setQuat('neck', 0.02, (Math.random() - 0.5) * 0.1, (Math.random() - 0.5) * 0.1);
    } else if (randomAction === 1) {
      setQuat('rightShoulder', 0, 0, (Math.random() - 0.5) * 0.05);
      setQuat('leftShoulder', 0, 0, (Math.random() - 0.5) * 0.05);
    } else if (randomAction === 2) {
      setQuat('rightHand', -0.1 + Math.random() * 0.1, 0, 0);
      setQuat('leftHand', -0.1 + Math.random() * 0.1, 0, 0);
    }
  }

  function stopActiveAnimation(): void {
    if (activeActionRef.current) {
      activeActionRef.current.stop();
      activeActionRef.current = null;
    }
    if (mixerRef.current && activeFinishHandlerRef.current) {
      mixerRef.current.removeEventListener('finished', activeFinishHandlerRef.current);
      activeFinishHandlerRef.current = null;
    }
  }

  function playVRMA(instruction: VRMAAnimationInstruction): void {
    const vrm = vrmRef.current;
    const mixer = mixerRef.current;
    const loader = vrmaLoaderRef.current;
    if (!vrm || !mixer || !loader) return;

    const requestId = ++animationRequestRef.current;
    isSigningRef.current = true;

    const startClip = (clip: THREE.AnimationClip) => {
      if (requestId !== animationRequestRef.current) return;

      const newAction = mixer.clipAction(clip);
      if (activeFinishHandlerRef.current) {
        mixer.removeEventListener('finished', activeFinishHandlerRef.current);
      }

      newAction.reset();
      newAction.setLoop(THREE.LoopOnce, 1);
      newAction.clampWhenFinished = true;

      const prevAction = activeActionRef.current;
      if (prevAction && prevAction.isRunning()) {
        newAction.play();
        newAction.crossFadeFrom(prevAction, 0.25, true);
      } else {
        newAction.fadeIn(0.25);
        newAction.play();
      }

      const onFinished = (event: { action: THREE.AnimationAction }) => {
        if (event.action !== newAction) return;
        mixer.removeEventListener('finished', onFinished);
        if (activeFinishHandlerRef.current === onFinished) {
          activeFinishHandlerRef.current = null;
        }
        if (activeActionRef.current === newAction) {
          newAction.fadeOut(0.3);
          activeActionRef.current = null;
        }
        recoverToIdle();
        onAnimationFinished?.();
      };

      activeActionRef.current = newAction;
      activeFinishHandlerRef.current = onFinished;
      mixer.addEventListener('finished', onFinished);
    };

    const cachedClip = animationCacheRef.current[instruction.path];
    if (cachedClip) {
      startClip(cachedClip);
      return;
    }

    loader.load(
      resolveAnimationUrl(instruction.path),
      (gltf) => {
        if (requestId !== animationRequestRef.current) return;
        const vrmAnimation = gltf.userData.vrmAnimations?.[0];
        if (!vrmAnimation) {
          recoverToIdle();
          return;
        }
        try {
          const clip = createVRMAnimationClip(vrmAnimation, vrm);
          animationCacheRef.current[instruction.path] = clip;
          startClip(clip);
        } catch {
          recoverToIdle();
        }
      },
      undefined,
      () => {
        if (requestId !== animationRequestRef.current) return;
        recoverToIdle();
      }
    );
  }

  function recoverToIdle(): void {
    isSigningRef.current = false;
    applyNaturalIdlePose(targetBonesRef.current, isLeftRestingRef.current);
  }

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}>
      {/* Floating Reset Camera Control Button */}
      {avatarReady && (
        <button
          onClick={handleResetCamera}
          title="Reset Camera to Front View"
          style={{
            position: 'absolute',
            top: '12px',
            right: '12px',
            background: 'rgba(255, 255, 255, 0.12)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            color: '#ffffff',
            padding: '8px 14px',
            borderRadius: '10px',
            fontSize: '12px',
            fontWeight: 500,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            zIndex: 20,
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255, 255, 255, 0.22)';
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255, 255, 255, 0.12)';
          }}
        >
          <RotateCcw size={14} />
          Reset Camera
        </button>
      )}

      {loading && (
        <div style={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(10, 8, 24, 0.95)',
          backdropFilter: 'blur(24px)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          color: '#ffffff',
          fontFamily: 'var(--font-inter)',
          gap: '16px',
          zIndex: 10,
          transition: 'opacity 0.6s ease'
        }}>
          <div className="spinner" />
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '15px', fontWeight: 600, letterSpacing: '-0.2px', marginBottom: '4px' }}>
              Preparing your AI interpreter...
            </div>
            <div style={{ fontSize: '12px', opacity: 0.6, fontWeight: 400 }}>
              Configuring humanoid IK rig & facial expressions
            </div>
          </div>
        </div>
      )}
      {loadError && (
        <div role="alert" style={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(10, 8, 24, 0.85)',
          backdropFilter: 'blur(20px)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          color: '#ef4444',
          fontFamily: 'var(--font-inter)',
          gap: '12px',
          zIndex: 10,
        }}>
          <span style={{ fontSize: '14px', fontWeight: 500 }}>{loadError}</span>
        </div>
      )}
      <div
        ref={containerRef}
        style={{
          width: '100%',
          height: '100%',
          minHeight: '350px',
          opacity: avatarReady ? 1 : 0,
          transition: 'opacity 0.8s cubic-bezier(0.16, 1, 0.3, 1)'
        }}
      />
    </div>
  );
}
