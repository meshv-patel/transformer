<script>
  import { fade } from 'svelte/transition';
  import { cellColor } from '../core/heatmap-color.js';

  // Renders one vector as a strip of colored cells. Used for both the
  // embedding table's rows and the materialized vector strips — this is
  // the one piece of real (observed, not speculative) duplication worth
  // extracting at this stage of the project. See docs/reference-implementation.md
  // "External review response" for why the rest of the review's proposed
  // framework was NOT adopted.
  //
  // Its cell-color logic now also lives in core/heatmap-color.js, shared
  // with MatrixHeatmap.svelte (Self-Attention scenes) — same reasoning,
  // extended once a second real consumer appeared. This component's own
  // markup, props, and rendered output are unchanged.

  export let vector = [];
  export let size = 'normal'; // 'normal' | 'small'
  export let maxAbs = 0.6; // color-scale saturation reference
  export let getTooltip = null; // (value, index) => string | undefined
  export let staggerDelay = null; // (index) => ms — entrance stagger; omit for no animation
</script>

<div class="heatmap" class:small={size === 'small'}>
  {#each vector as v, i}
    {#if staggerDelay}
      <span
        class="cell"
        style="background:{cellColor(v, maxAbs)}"
        title={getTooltip ? getTooltip(v, i) : undefined}
        in:fade={{ duration: 120, delay: staggerDelay(i) }}
      ></span>
    {:else}
      <span
        class="cell"
        style="background:{cellColor(v, maxAbs)}"
        title={getTooltip ? getTooltip(v, i) : undefined}
      ></span>
    {/if}
  {/each}
</div>

<style>
  .heatmap { display: flex; gap: 1px; }
  .cell { display: inline-block; width: 0.9rem; height: 1.4rem; border-radius: 2px; }
  .heatmap.small .cell { width: 0.45rem; height: 0.7rem; }
</style>
