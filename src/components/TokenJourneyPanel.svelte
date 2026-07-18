<script>
  import { selectedTokenPos, tokenTrace } from '../core/stores/tokenJourneyStore.js';
  import { forwardPassData } from '../core/data-loader.js';
</script>

{#if $selectedTokenPos !== null}
  <div class="journey-panel">
    <div class="journey-header">
      <span>Token Journey — position {$selectedTokenPos}</span>
      <button class="close-btn" on:click={() => selectedTokenPos.set(null)} aria-label="Close">×</button>
    </div>

    {#if $tokenTrace}
      <div class="journey-strip">
        {#each $tokenTrace as stage}
          <div class="stage-cell">
            <span class="stage-label">{stage.stageLabel}</span>
            <!-- Phase 3/4: render an actual heatmap/sparkline of stage.vector here -->
            <div class="stage-placeholder">
              {stage.vector ? `[${stage.vector.length}]` : '—'}
            </div>
          </div>
        {/each}
      </div>
    {:else}
      <p class="empty">No forward-pass data loaded yet.</p>
    {/if}
  </div>
{:else}
  <div class="journey-hint">
    Click any token in the visualization to track its representation across every stage.
    {#if !$forwardPassData}<span class="note"> (data not loaded yet — Phase 2 stub)</span>{/if}
  </div>
{/if}

<style>
  .journey-panel {
    background: rgba(20, 23, 34, 0.95);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.75rem;
    padding: 0.75rem 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    max-width: 40rem;
  }

  @media (max-width: 900px) {
    .journey-panel, .journey-hint { max-width: 92vw; font-size: 0.7rem; }
  }

  .journey-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    font-weight: 600;
  }
  .close-btn { background: none; border: none; color: inherit; cursor: pointer; font-size: 1rem; }
  .journey-strip { display: flex; gap: 0.5rem; overflow-x: auto; }
  .stage-cell { display: flex; flex-direction: column; align-items: center; min-width: 3.5rem; }
  .stage-label { font-size: 0.6rem; opacity: 0.6; text-align: center; margin-bottom: 0.25rem; }
  .stage-placeholder {
    width: 100%;
    height: 2rem;
    border: 1px dashed var(--border-subtle, #333a4d);
    border-radius: 0.35rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    opacity: 0.5;
  }
  .journey-hint, .empty {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    opacity: 0.5;
    padding: 0.5rem 0;
  }
  .note { opacity: 0.4; }
</style>
