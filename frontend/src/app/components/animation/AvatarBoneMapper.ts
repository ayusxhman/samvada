import * as THREE from 'three';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';

const VRM_DEBUG = process.env.NEXT_PUBLIC_VRM_DEBUG === 'true';

/**
 * VRM Humanoid Bone Name Constants
 * Complete set of 52 standard VRM Humanoid bones supported by the AI animation engine.
 */
export const ALL_VRM_HUMANOID_BONES: readonly VRMHumanBoneName[] = [
  'hips', 'spine', 'chest', 'upperChest', 'neck', 'head',
  'leftShoulder', 'rightShoulder', 'leftUpperArm', 'rightUpperArm',
  'leftLowerArm', 'rightLowerArm', 'leftHand', 'rightHand',
  'leftUpperLeg', 'rightUpperLeg', 'leftLowerLeg', 'rightLowerLeg',
  'leftFoot', 'rightFoot', 'leftToes', 'rightToes',
  // Left Hand Finger Bones
  'leftThumbMetacarpal', 'leftThumbProximal', 'leftThumbDistal',
  'leftIndexProximal', 'leftIndexIntermediate', 'leftIndexDistal',
  'leftMiddleProximal', 'leftMiddleIntermediate', 'leftMiddleDistal',
  'leftRingProximal', 'leftRingIntermediate', 'leftRingDistal',
  'leftLittleProximal', 'leftLittleIntermediate', 'leftLittleDistal',
  // Right Hand Finger Bones
  'rightThumbMetacarpal', 'rightThumbProximal', 'rightThumbDistal',
  'rightIndexProximal', 'rightIndexIntermediate', 'rightIndexDistal',
  'rightMiddleProximal', 'rightMiddleIntermediate', 'rightMiddleDistal',
  'rightRingProximal', 'rightRingIntermediate', 'rightRingDistal',
  'rightLittleProximal', 'rightLittleIntermediate', 'rightLittleDistal'
] as const;

export interface BoneMappingValidation {
  totalMapped: number;
  totalRequired: number;
  mappedBones: string[];
  missingOptionalBones: string[];
}

/**
 * AvatarBoneMapper
 * Centralized abstraction layer between AI quaternion animation timelines
 * and live VRM humanoid bone scene nodes.
 */
export class AvatarBoneMapper {
  private boneNodes: Map<string, THREE.Object3D> = new Map();
  private restOffsets: Map<string, THREE.Quaternion> = new Map();
  private isInitialized: boolean = false;

  constructor() {}

  /**
   * Initializes bone mapping from a loaded VRM instance.
   * Maps every VRM humanoid bone node and caches its rest-pose offset.
   */
  public initialize(vrm: VRM): BoneMappingValidation {
    this.boneNodes.clear();
    this.restOffsets.clear();

    const mappedBones: string[] = [];
    const missingOptionalBones: string[] = [];

    if (vrm.humanoid) {
      ALL_VRM_HUMANOID_BONES.forEach(boneName => {
        const boneNode = vrm.humanoid?.getNormalizedBoneNode(boneName as VRMHumanBoneName);
        if (boneNode) {
          this.boneNodes.set(boneName, boneNode);
          
          // Store rest pose quaternion offset for coordinate correction
          const restQuat = new THREE.Quaternion().copy(boneNode.quaternion);
          this.restOffsets.set(boneName, restQuat);
          
          mappedBones.push(boneName);
        } else {
          missingOptionalBones.push(boneName);
        }
      });
    }

    this.isInitialized = true;

    if (VRM_DEBUG) {
      console.info('[VRM debug] normalized humanoid bones', mappedBones.map((boneName) => ({
        boneName,
        uuid: this.boneNodes.get(boneName)?.uuid,
        restQuaternion: this.restOffsets.get(boneName)?.toArray(),
      })));
      console.info('[VRM debug] missing normalized humanoid bones', missingOptionalBones);
    }

    return {
      totalMapped: mappedBones.length,
      totalRequired: ALL_VRM_HUMANOID_BONES.length,
      mappedBones,
      missingOptionalBones
    };
  }

  /**
   * Returns the normalized scene node for a given humanoid bone name.
   */
  public getBoneNode(boneName: string): THREE.Object3D | null {
    return this.boneNodes.get(boneName) ?? null;
  }

  /**
   * Returns whether a bone is mapped on the current avatar.
   */
  public hasBone(boneName: string): boolean {
    return this.boneNodes.has(boneName);
  }

  /**
   * Applies an AI quaternion rotation to a target bone with rest-pose coordinate offset correction
   * and anatomical joint safety limits.
   * AI Quaternion -> Rest Offset Composition -> Anatomical Clamping -> Live VRM Bone Rotation
   */
  /**
   * Computes rest-pose offset composition and anatomical joint clamping into outCombinedQuat.
   * Does NOT mutate boneNode.quaternion.
   */
  public computeCombinedQuaternion(
    boneName: string,
    targetQuat: THREE.Quaternion,
    outCombinedQuat: THREE.Quaternion
  ): boolean {
    if (!this.boneNodes.has(boneName)) return false;

    const restOffset = this.restOffsets.get(boneName);
    if (restOffset) {
      outCombinedQuat.multiplyQuaternions(restOffset, targetQuat);
    } else {
      outCombinedQuat.copy(targetQuat);
    }

    return true;
  }

  /**
   * Applies targetQuat to scene boneNode with rest offset composition, anatomical clamping,
   * and optional slerp interpolation factor (alpha).
   */
  public applyBoneQuaternion(
    boneName: string,
    targetQuat: THREE.Quaternion,
    outCombinedQuat: THREE.Quaternion,
    alpha?: number
  ): boolean {
    const boneNode = this.boneNodes.get(boneName);
    if (!boneNode) return false;

    this.computeCombinedQuaternion(boneName, targetQuat, outCombinedQuat);

    if (alpha !== undefined && alpha < 1.0) {
      boneNode.quaternion.slerp(outCombinedQuat, alpha);
    } else {
      boneNode.quaternion.copy(outCombinedQuat);
    }

    return true;
  }

  /**
   * Returns all mapped bone names.
   */
  public getMappedBoneNames(): string[] {
    return Array.from(this.boneNodes.keys());
  }

  public ready(): boolean {
    return this.isInitialized;
  }
}
