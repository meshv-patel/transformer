<script>
  export let before; // { label, shape: number[] }
  export let after;  // { label, shape: number[] }
  export let whatChanged = '';
  export let structured = null; // { entered, happened, changed, leaves } — preferred, more scannable
</script>

<div class="before-after">
  <div class="card">
    <span class="card-eyebrow">Before</span>
    <span class="card-label">{before.label}</span>
    <span class="card-shape">[{before.shape.join(', ')}]</span>
  </div>

  <div class="arrow-col">
    <span class="big-arrow">→</span>
  </div>

  <div class="card after">
    <span class="card-eyebrow">After</span>
    <span class="card-label">{after.label}</span>
    <span class="card-shape">[{after.shape.join(', ')}]</span>
  </div>

  {#if structured}
    <div class="recap">
      <div class="recap-row"><span class="recap-label">Entered</span><span>{structured.entered}</span></div>
      <div class="recap-row"><span class="recap-label">Happened</span><span>{structured.happened}</span></div>
      <div class="recap-row"><span class="recap-label">Changed</span><span>{structured.changed}</span></div>
      <div class="recap-row"><span class="recap-label">Leaves</span><span>{structured.leaves}</span></div>
    </div>
  {:else if whatChanged}
    <div class="what-changed">
      <span class="wc-label">What changed?</span>
      <p>{whatChanged}</p>
    </div>
  {/if}
</div>

<style>
  .before-after {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 1rem;
    align-items: center;
    max-width: 40rem;
    width: 100%;
  }
  .card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.3rem;
    background: var(--bg-elevated, #141722);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.85rem;
    padding: 1.25rem 1rem;
  }
  .card.after { border-color: var(--accent, #7aa2ff); box-shadow: 0 0 20px rgba(122, 162, 255, 0.15); }
  .card-eyebrow { font-family: 'Inter', sans-serif; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.55; }
  .card-label { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem; }
  .card-shape { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; opacity: 0.8; }
  .arrow-col { display: flex; justify-content: center; }
  .big-arrow { font-size: 1.6rem; opacity: 0.5; }
  .what-changed {
    grid-column: 1 / -1;
    text-align: center;
    margin-top: 0.5rem;
    font-family: 'Inter', sans-serif;
  }
  .wc-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.55; }
  .what-changed p { margin: 0.25rem 0 0; line-height: 1.5; font-size: 0.9rem; }

  .recap {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
    gap: 0.4rem;
    margin-top: 0.5rem;
  }
  .recap-row {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.5rem;
    padding: 0.4rem 0.6rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    text-align: left;
  }
  .recap-label { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.55; }
</style>
