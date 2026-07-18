<script>
  import { dModel, numHeads, seqLen, dK, RANGES, isValidHeadSplit } from '../core/stores/configStore.js';
  import { dataMode } from '../core/stores/sceneStore.js';
</script>

{#if $dataMode === 'interactive'}
  <div class="config-panel">
    <h4>Model shape</h4>

    <label>
      d_model: <strong>{$dModel}</strong>
      <input type="range" min={RANGES.dModel.min} max={RANGES.dModel.max} step={RANGES.dModel.step} bind:value={$dModel} />
    </label>

    <label>
      heads: <strong>{$numHeads}</strong>
      <input type="range" min={RANGES.numHeads.min} max={RANGES.numHeads.max} step={RANGES.numHeads.step} bind:value={$numHeads} />
    </label>

    <label>
      seq_len: <strong>{$seqLen}</strong>
      <input type="range" min={RANGES.seqLen.min} max={RANGES.seqLen.max} step={RANGES.seqLen.step} bind:value={$seqLen} />
    </label>

    {#if $isValidHeadSplit}
      <p class="derived">d_k = d_model / heads = <strong>{$dK}</strong></p>
    {:else}
      <p class="warning">heads must evenly divide d_model — pick a different combination.</p>
    {/if}
  </div>
{/if}

<style>
  .config-panel {
    background: rgba(20, 23, 34, 0.9);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.75rem;
    padding: 0.75rem 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    width: 14rem;
  }
  h4 { margin: 0 0 0.5rem; font-size: 0.75rem; text-transform: uppercase; opacity: 0.6; letter-spacing: 0.05em; }
  label { display: block; margin-bottom: 0.5rem; }
  input[type='range'] { width: 100%; }
  .derived { margin: 0.25rem 0 0; opacity: 0.8; }
  .warning { margin: 0.25rem 0 0; color: #ff8a8a; }
</style>
