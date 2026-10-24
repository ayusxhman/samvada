import * as THREE from 'three';

export interface AnimationFrame {
  t: number;
  bones: Record<string, [number, number, number, number]>; // [x, y, z, w]
  blendshapes?: Record<string, number>;
}

export interface AnimationTimeline {
  fps: number;
  duration: number;
  frames: AnimationFrame[];
}

export class TimelinePlayer {
  private timeline: AnimationTimeline | null = null;
  private elapsed: number = 0;
  private playing: boolean = false;
  private playbackSpeed: number = 1.0;

  // Pre-allocated static Quaternions to prevent garbage collection allocations during rendering
  private quatCache: Map<string, THREE.Quaternion> = new Map();
  private qStart: THREE.Quaternion = new THREE.Quaternion();
  private qEnd: THREE.Quaternion = new THREE.Quaternion();

  constructor() {}

  public play(timeline: AnimationTimeline): void {
    if (!timeline || !timeline.frames || timeline.frames.length === 0) return;
    this.timeline = timeline;
    this.elapsed = 0;
    this.playing = true;

    // Pre-allocate quaternion instances for all unique bones in this timeline
    const allBones = new Set<string>();
    timeline.frames.forEach(f => {
      if (f.bones) {
        Object.keys(f.bones).forEach(b => allBones.add(b));
      }
    });

    allBones.forEach(b => {
      if (!this.quatCache.has(b)) {
        this.quatCache.set(b, new THREE.Quaternion());
      }
    });
  }

  public setSpeed(speed: number): void {
    this.playbackSpeed = Math.max(0.1, Math.min(5.0, speed));
  }

  public stop(): void {
    this.playing = false;
    this.elapsed = 0;
  }

  public isPlaying(): boolean {
    return this.playing;
  }

  public getElapsed(): number {
    return this.elapsed;
  }

  public getDuration(): number {
    return this.timeline?.duration || 0;
  }

  /**
   * Updates player time and evaluates interpolated bone quaternions and blendshapes.
   * Frame-rate independent SLERP computation.
   */
  public update(delta: number): {
    bones: Map<string, THREE.Quaternion>;
    blendshapes: Record<string, number>;
    finished: boolean;
  } | null {
    if (!this.playing || !this.timeline || this.timeline.frames.length === 0) {
      return null;
    }

    this.elapsed += delta * this.playbackSpeed;

    const frames = this.timeline.frames;
    const duration = this.timeline.duration;

    if (this.elapsed >= duration) {
      this.playing = false;
      // Return last frame state
      const lastFrame = frames[frames.length - 1];
      const resultBones = new Map<string, THREE.Quaternion>();
      if (lastFrame.bones) {
        Object.entries(lastFrame.bones).forEach(([bName, [x, y, z, w]]) => {
          let q = this.quatCache.get(bName);
          if (!q) {
            q = new THREE.Quaternion();
            this.quatCache.set(bName, q);
          }
          q.set(x, y, z, w);
          resultBones.set(bName, q);
        });
      }
      return {
        bones: resultBones,
        blendshapes: lastFrame.blendshapes || {},
        finished: true
      };
    }

    // Binary search to find bracket keyframe indices
    let left = 0;
    let right = frames.length - 1;
    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      if (frames[mid].t <= this.elapsed) {
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }

    const idx0 = Math.max(0, right);
    const idx1 = Math.min(frames.length - 1, idx0 + 1);

    const f0 = frames[idx0];
    const f1 = frames[idx1];

    let factor = 0;
    if (idx0 !== idx1 && f1.t > f0.t) {
      factor = Math.max(0, Math.min(1, (this.elapsed - f0.t) / (f1.t - f0.t)));
    }

    const resultBones = new Map<string, THREE.Quaternion>();
    const resultBlendshapes: Record<string, number> = {};

    // Combine all bone names present in either keyframe
    const boneNames = new Set<string>([
      ...Object.keys(f0.bones || {}),
      ...Object.keys(f1.bones || {})
    ]);

    boneNames.forEach(bName => {
      const arr0 = f0.bones[bName];
      const arr1 = f1.bones[bName] || arr0;

      let qTarget = this.quatCache.get(bName);
      if (!qTarget) {
        qTarget = new THREE.Quaternion();
        this.quatCache.set(bName, qTarget);
      }

      if (arr0 && arr1) {
        this.qStart.set(arr0[0], arr0[1], arr0[2], arr0[3]);
        this.qEnd.set(arr1[0], arr1[1], arr1[2], arr1[3]);
        // SLERP interpolation between keyframes
        qTarget.copy(this.qStart).slerp(this.qEnd, factor);
      } else if (arr0) {
        qTarget.set(arr0[0], arr0[1], arr0[2], arr0[3]);
      }

      resultBones.set(bName, qTarget);
    });

    // Interpolate blendshapes
    if (f0.blendshapes) {
      Object.entries(f0.blendshapes).forEach(([bsName, val0]) => {
        const val1 = f1.blendshapes?.[bsName] ?? val0;
        resultBlendshapes[bsName] = val0 + (val1 - val0) * factor;
      });
    }

    return {
      bones: resultBones,
      blendshapes: resultBlendshapes,
      finished: false
    };
  }
}
