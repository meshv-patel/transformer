<script>
  import { highlightedTermId, setHighlight, clearHighlight } from '../core/stores/highlightStore.js';

  // steps: [{ label: string, shape: number[], highlightId?: string }]
  export let steps = [];
  export let activeIndex = 0;
</script>

<div class="shape-trace">
  {#each steps as step, i}
    <div
      class="step"
      role="note"
      class:active={i === activeIndex}
      class:past={i < activeIndex}
      class:synced={step.highlightId && $highlightedTermId === step.highlightId}
      on:mouseenter={() => step.highlightId && setHighlight(step.highlightId)}
      on:mouseleave={() => step.highlightId && clearHighlight()}
    >
      <span class="step-label">{step.label}</span>
      <span class="step-shape">[{step.shape.join(', ')}]</span>
    </div>
    {#if i < steps.length - 1}
      <span class="arrow" class:active={i < activeIndex}>→</span>
    {/if}
  {/each}
</div>

<style>
  .shape-trace {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    flex-wrap: wrap;
    justify-content: center;
  }
  .step {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0.35rem 0.6rem;
    border-radius: 0.5rem;
    border: 1px solid var(--border-subtle, #232838);
    opacity: 0.45;
    transition: opacity 0.25s, border-color 0.25s, transform 0.25s;
  }
  .step.past { opacity: 0.7; }
  .step.active {
    opacity: 1;
    border-color: var(--accent, #7aa2ff);
    transform: translateY(-1px);
  }
  .step.synced { border-color: var(--accent, #7aa2ff); box-shadow: 0 0 8px rgba(122, 162, 255, 0.35); }
  .step-label { font-family: 'Inter', sans-serif; font-size: 0.62rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.04em; }
  .step-shape { font-weight: 600; }
  .arrow { opacity: 0.3; }
  .arrow.active { opacity: 0.8; color: var(--accent, #7aa2ff); }
</style>
