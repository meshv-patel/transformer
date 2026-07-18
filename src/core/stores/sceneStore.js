import { writable, derived } from 'svelte/store';
import { SCENES } from '../scene-registry.js';

export const sceneIndex = writable(0);
export const subStepIndex = writable(0);
export const appMode = writable('presenter'); // 'presenter' | 'explore'
export const dataMode = writable('lecture');  // 'lecture' | 'interactive'

export const currentScene = derived(sceneIndex, ($i) => SCENES[$i]);

export const currentSubSteps = derived(currentScene, ($scene) => $scene?.subSteps ?? []);

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

export function goToScene(index) {
  sceneIndex.set(clamp(index, 0, SCENES.length - 1));
  subStepIndex.set(0);
}

export function next() {
  // Explicit two-phase advance: exhaust sub-steps within a scene first,
  // then move to the next scene (this is what makes "every click advances
  // the animation" true even for multi-step scenes).
  let sIdx, subIdx;
  sceneIndex.subscribe((v) => (sIdx = v))();
  subStepIndex.subscribe((v) => (subIdx = v))();
  const subSteps = SCENES[sIdx]?.subSteps ?? [];

  if (subIdx < subSteps.length - 1) {
    subStepIndex.set(subIdx + 1);
  } else if (sIdx < SCENES.length - 1) {
    sceneIndex.set(sIdx + 1);
    subStepIndex.set(0);
  }
}

export function prev() {
  let sIdx, subIdx;
  sceneIndex.subscribe((v) => (sIdx = v))();
  subStepIndex.subscribe((v) => (subIdx = v))();

  if (subIdx > 0) {
    subStepIndex.set(subIdx - 1);
  } else if (sIdx > 0) {
    const prevSubSteps = SCENES[sIdx - 1]?.subSteps ?? [];
    sceneIndex.set(sIdx - 1);
    subStepIndex.set(Math.max(0, prevSubSteps.length - 1));
  }
}

export function replayCurrent() {
  // Scene components subscribe to this counter and re-run their tween sequence
  // whenever it changes, without altering position in the lecture.
  replayTick.update((n) => n + 1);
}

export const replayTick = writable(0);

export const deepDiveOpen = writable(false);

export function toggleDeepDive() {
  deepDiveOpen.update((v) => !v);
}
