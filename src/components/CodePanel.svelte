<script>
  import { highlightedTermId, setHighlight, clearHighlight } from '../core/stores/highlightStore.js';

  // [{ id, code }] — one entry per highlightable line, keyed to the same
  // termId space as EquationPanel and the scene's SVG elements.
  export let lines = [];
</script>

<pre class="code-panel">{#if lines.length}{#each lines as line}<span
      class="line"
      role="button"
      tabindex="0"
      aria-label={`Code line, synced with equation and visualization`}
      class:active={$highlightedTermId === line.id}
      on:mouseenter={() => setHighlight(line.id)}
      on:mouseleave={clearHighlight}
      on:focus={() => setHighlight(line.id)}
      on:blur={clearHighlight}
    >{line.code}</span>{'\n'}{/each}{:else}<span class="placeholder">// code mapping not authored for this scene yet</span>{/if}</pre>

<style>
  .code-panel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    background: rgba(0,0,0,0.25);
    border-radius: 0.5rem;
    padding: 0.75rem;
    overflow-x: auto;
    white-space: pre;
  }
  .line { display: inline-block; padding: 0 0.2rem; border-radius: 0.2rem; }
  .line:focus-visible { outline: 2px solid var(--accent, #7aa2ff); outline-offset: 1px; }
  .line.active { background: var(--accent, #7aa2ff); color: #0b0e14; }
  .placeholder { opacity: 0.4; font-family: 'Inter', sans-serif; }
</style>
