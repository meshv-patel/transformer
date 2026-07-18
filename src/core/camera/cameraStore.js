import { spring } from 'svelte/motion';

// A "shot" is authored, not measured: { x, y, scale } in the stage's own
// coordinate space. We deliberately don't do live DOM-rect measurement for
// pan/zoom targets — authored shots (like a real camera operator's shot
// list) are predictable across screen sizes and projector aspect ratios,
// where measured-geometry approaches tend to break.
//
// Reusable contract for future scenes: define a SHOTS map keyed by
// sub-step index (or name), call `focusShot(SHOTS[i])` reactively when the
// sub-step changes. That's the whole integration surface.

const reducedMotion =
  typeof window !== 'undefined' &&
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

// Slightly underdamped spring = natural anticipation/follow-through (a small
// overshoot before settling) without any hand-authored easing curve.
export const camera = spring(
  { x: 0, y: 0, scale: 1 },
  reducedMotion ? { stiffness: 1, damping: 1 } : { stiffness: 0.07, damping: 0.55 }
);

export function focusShot(shot = {}, opts) {
  camera.set({ x: shot.x ?? 0, y: shot.y ?? 0, scale: shot.scale ?? 1 }, opts);
}

export function resetCamera(opts) {
  camera.set({ x: 0, y: 0, scale: 1 }, opts);
}

// Whether the camera is meaningfully zoomed in — used to drive background
// vignette/dimming, so "focus" reads as a single coherent cinematic effect
// rather than two independently-tuned systems.
export function isFocused(cameraState) {
  return cameraState.scale > 1.02;
}
