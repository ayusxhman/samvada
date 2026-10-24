import * as THREE from 'three';
import { TimelinePlayer, type AnimationTimeline } from './TimelinePlayer';

export interface AnimationUpdate {
  bones: Map<string, THREE.Quaternion>;
  blendshapes: Record<string, number>;
  finished: boolean;
}

export interface AnimationProvider {
  readonly type: 'ai_quaternion' | 'vrma' | 'procedural' | 'idle';
  
  start(timelineOrPath?: unknown): void;
  update(delta: number): AnimationUpdate | null;
  isPlaying(): boolean;
  stop(): void;
  onFinished?: () => void;
}

/**
 * Primary Animation Provider: Consumes AI-generated bone quaternion timelines
 */
export class AIQuaternionProvider implements AnimationProvider {
  readonly type = 'ai_quaternion';
  private player: TimelinePlayer = new TimelinePlayer();
  public onFinished?: () => void;

  constructor(speed: number = 1.0) {
    this.player.setSpeed(speed);
  }

  public start(timeline: AnimationTimeline): void {
    if (!timeline) return;
    this.player.play(timeline);
  }

  public update(delta: number): AnimationUpdate | null {
    const res = this.player.update(delta);
    if (res && res.finished) {
      this.onFinished?.();
    }
    return res;
  }

  public isPlaying(): boolean {
    return this.player.isPlaying();
  }

  public stop(): void {
    this.player.stop();
  }

  public setSpeed(speed: number): void {
    this.player.setSpeed(speed);
  }
}

/**
 * Idle Animation Provider: Default posture & resting state
 */
export class IdleProvider implements AnimationProvider {
  readonly type = 'idle';
  public onFinished?: () => void;

  public start(): void {}
  public update(): AnimationUpdate | null {
    return null; // Handled by standard natural idle pose slerp pass
  }
  public isPlaying(): boolean {
    return false;
  }
  public stop(): void {}
}
