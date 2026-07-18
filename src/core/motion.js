export const prefersReducedMotion =
  typeof window !== 'undefined' &&
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/**
 * Scales a Svelte transition duration/delay to ~0 when the user prefers
 * reduced motion. CSS `@media (prefers-reduced-motion: reduce)` cannot
 * neutralize Svelte's `fly`/`fade`/`crossfade` transitions on its own —
 * they animate via JS-computed inline styles, not CSS `transition:` rules —
 * so anywhere a scene passes a `duration`/`delay` into one of those, wrap it
 * with this instead of the raw number.
 */
export function motionMs(ms) {
  return prefersReducedMotion ? 0 : ms;
}
