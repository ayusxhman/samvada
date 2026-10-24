import type { VRM } from '@pixiv/three-vrm';

/**
 * AvatarExpressionController
 * Encapsulates facial expressions, eye blinking, lip-sync visemes, and gaze control.
 * Supports both VRM 1.0 (expressionManager) and VRM 0.x (blendShapeProxy) fallbacks.
 */
export class AvatarExpressionController {
  private vrm: VRM | null = null;
  private isBlinking: boolean = false;
  private blinkPhase: number = 0;
  private blinkTime: number = 0;
  private nextBlinkInterval: number = Math.random() * 3 + 2;

  constructor() {}

  public initialize(vrm: VRM): void {
    this.vrm = vrm;
    this.isBlinking = false;
    this.blinkTime = 0;
    this.nextBlinkInterval = Math.random() * 3 + 2;
  }

  /**
   * Sets a specific facial expression or morph target weight (0.0 to 1.0).
   */
  public setExpression(name: string, value: number): void {
    if (!this.vrm) return;
    const clampedValue = Math.max(0.0, Math.min(1.0, value));

    // 1. Try VRM 1.0 Expression Manager
    if (this.vrm.expressionManager) {
      try {
        this.vrm.expressionManager.setValue(name, clampedValue);
      } catch {
        // missing expression ignored safely
      }
    }

    // 2. Try VRM 0.x BlendShape Proxy Fallback
    // @ts-expect-error fallback for VRM 0.x blendShapeProxy
    if (this.vrm.blendShapeProxy) {
      try {
        // Upper-case conversion for VRM 0.x preset names (e.g. happy -> HAPPY, blink -> BLINK)
        const vrm0Name = name.toUpperCase();
        // @ts-expect-error fallback for VRM 0.x blendShapeProxy
        this.vrm.blendShapeProxy.setValue(vrm0Name, clampedValue);
      } catch {
        // missing blendshape ignored safely
      }
    }
  }



  /**
   * Drives real-time lip-sync mouth movement during speech synthesis.
   */
  public updateSpeechVisemes(elapsedTime: number, isSpeaking: boolean): void {
    if (!isSpeaking) {
      this.setExpression('aa', 0.0);
      this.setExpression('ih', 0.0);
      this.setExpression('ou', 0.0);
      return;
    }

    // Parametric natural mouth open/close visemes synced to vocal frequencies
    const mouthOpen = (Math.sin(elapsedTime * 14.0) * 0.4 + Math.sin(elapsedTime * 22.0) * 0.25 + 0.35);
    const clampedMouth = Math.max(0.05, Math.min(0.7, mouthOpen));
    
    const cycle = Math.floor(elapsedTime * 4.0) % 3;
    if (cycle === 0) {
      this.setExpression('aa', clampedMouth);
      this.setExpression('ih', 0.0);
      this.setExpression('ou', 0.0);
    } else if (cycle === 1) {
      this.setExpression('aa', 0.0);
      this.setExpression('ih', clampedMouth * 0.8);
      this.setExpression('ou', 0.0);
    } else {
      this.setExpression('aa', 0.0);
      this.setExpression('ih', 0.0);
      this.setExpression('ou', clampedMouth * 0.7);
    }
  }

  /**
   * Per-frame update for blinking dynamics and expression updates.
   */
  public update(delta: number): void {
    if (!this.vrm) return;

    // Automatic natural eye blinking dynamics
    this.blinkTime += delta;
    if (!this.isBlinking && this.blinkTime > this.nextBlinkInterval) {
      this.isBlinking = true;
      this.blinkPhase = 0;
      this.blinkTime = 0;
      this.nextBlinkInterval = Math.random() * 4 + 2;
    }

    if (this.isBlinking) {
      this.blinkPhase += delta * 14.0; // Smooth 140ms blink speed
      let blinkVal = 0;
      if (this.blinkPhase < 1.0) {
        blinkVal = Math.sin((this.blinkPhase * Math.PI) / 2.0);
      } else if (this.blinkPhase < 2.0) {
        blinkVal = Math.cos(((this.blinkPhase - 1.0) * Math.PI) / 2.0);
      } else {
        this.isBlinking = false;
        blinkVal = 0;
      }
      this.setExpression('blink', blinkVal);
    }

    // Trigger updates on VRM expression managers
    if (this.vrm.expressionManager) {
      this.vrm.expressionManager.update();
    }
    // @ts-expect-error fallback for VRM 0.x blendShapeProxy
    if (this.vrm.blendShapeProxy) {
      // @ts-expect-error fallback for VRM 0.x blendShapeProxy
      this.vrm.blendShapeProxy.update();
    }
  }

  /**
   * Sets gaze look-at direction weights.
   */
  public setGazeDirection(horizontal: number, vertical: number): void {
    if (horizontal > 0) {
      this.setExpression('lookRight', horizontal);
      this.setExpression('lookLeft', 0);
    } else {
      this.setExpression('lookLeft', Math.abs(horizontal));
      this.setExpression('lookRight', 0);
    }

    if (vertical > 0) {
      this.setExpression('lookUp', vertical);
      this.setExpression('lookDown', 0);
    } else {
      this.setExpression('lookDown', Math.abs(vertical));
      this.setExpression('lookUp', 0);
    }
  }

  /**
   * Resets all facial expressions to neutral resting state.
   */
  public resetToNeutral(): void {
    const expressions = [
      'happy', 'relaxed', 'smile', 'joy', 'sad', 'angry', 'surprised',
      'aa', 'ih', 'ou', 'ee', 'oh', 'lookLeft', 'lookRight', 'lookUp', 'lookDown'
    ];
    expressions.forEach(exp => this.setExpression(exp, 0.0));
    this.setExpression('neutral', 1.0);
  }
}
