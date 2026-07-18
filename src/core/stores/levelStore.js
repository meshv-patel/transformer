import { writable } from 'svelte/store';

export const LEVELS = [
  { id: 'beginner', label: 'Beginner' },
  { id: 'mtech', label: 'MTech' },
  { id: 'research', label: 'Research Notes' },
];

export const explanationLevel = writable('mtech'); // sensible default for the target audience
