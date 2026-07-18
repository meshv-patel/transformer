import { writable, derived } from 'svelte/store';

// Values baked into forward-pass.json — Lecture Mode is locked to these.
export const LECTURE_DEFAULTS = { dModel: 16, numHeads: 4, seqLen: 4 };

export const RANGES = {
  dModel: { min: 8, max: 64, step: 8 },
  numHeads: { min: 1, max: 8, step: 1 },
  seqLen: { min: 2, max: 8, step: 1 },
};

export const dModel = writable(LECTURE_DEFAULTS.dModel);
export const numHeads = writable(LECTURE_DEFAULTS.numHeads);
export const seqLen = writable(LECTURE_DEFAULTS.seqLen);

export const dK = derived([dModel, numHeads], ([$dModel, $numHeads]) =>
  $numHeads > 0 ? $dModel / $numHeads : $dModel
);

export const isValidHeadSplit = derived(
  [dModel, numHeads],
  ([$dModel, $numHeads]) => $numHeads > 0 && $dModel % $numHeads === 0
);

export function resetToLectureDefaults() {
  dModel.set(LECTURE_DEFAULTS.dModel);
  numHeads.set(LECTURE_DEFAULTS.numHeads);
  seqLen.set(LECTURE_DEFAULTS.seqLen);
}

export function isAtLectureDefaults(state) {
  return (
    state.dModel === LECTURE_DEFAULTS.dModel &&
    state.numHeads === LECTURE_DEFAULTS.numHeads &&
    state.seqLen === LECTURE_DEFAULTS.seqLen
  );
}
