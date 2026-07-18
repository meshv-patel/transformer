<script>
  import { highlightedTermId, setHighlight, clearHighlight } from '../core/stores/highlightStore.js';

  // Phase 3/4: each scene supplies its own `terms` array via syncMap
  // (architecture Addendum §6). Phase 2 ships the wiring + a placeholder.
  export let terms = []; // [{ id, tex }]
</script>

<div class="equation-panel">
  {#if terms.length}
    {#each terms as term}
      <span
        class="term"
        role="button"
        tabindex="0"
        aria-label={`Equation term: ${term.tex} — synced with code and visualization`}
        class:active={$highlightedTermId === term.id}
        on:mouseenter={() => setHighlight(term.id)}
        on:mouseleave={clearHighlight}
        on:focus={() => setHighlight(term.id)}
        on:blur={clearHighlight}
      >
        {term.tex}
      </span>
    {/each}
  {:else}
    <span class="placeholder">Equation not authored for this scene yet.</span>
  {/if}
</div>

<style>
  .equation-panel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .term { padding: 0.1rem 0.3rem; border-radius: 0.25rem; cursor: default; }
  .term:focus-visible { outline: 2px solid var(--accent, #7aa2ff); outline-offset: 1px; }
  .term.active { background: var(--accent, #7aa2ff); color: #0b0e14; }
  .placeholder { opacity: 0.4; font-family: 'Inter', sans-serif; }
</style>
