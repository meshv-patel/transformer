<script>
  export let items = []; // [{ title, body }]
  // { left, right: number[], caption, leftLabel, rightLabel, leftNote, rightNote,
  //   leftStyle, rightStyle: 'binary' | 'continuous' }
  // 'binary' renders on/off cells (Embedding's one-hot column); 'continuous'
  // renders signed intensity (Embedding's dense column, and both of
  // Positional Encoding's fast-vs-slow-frequency columns). Defaults match
  // Embedding's original one-hot-vs-dense framing exactly.
  export let example = null;
  let open = false;
</script>

<div class="why-panel">
  <button class="why-toggle" on:click={() => (open = !open)} aria-expanded={open}>
    {open ? '▾' : '▸'} Why?
  </button>

  {#if open}
    <div class="why-body">
      {#each items as item}
        <div class="why-item">
          <strong>{item.title}</strong>
          <p>{item.body}</p>
        </div>
      {/each}

      {#if example}
        <div class="why-example">
          <p class="example-caption">{example.caption}</p>
          <div class="example-row">
            <div class="example-col">
              <span class="example-label">{example.leftLabel ?? `One-hot [${example.left.length}]`}</span>
              <div class="mini-strip">
                {#each example.left as v}
                  {#if (example.leftStyle ?? 'binary') === 'binary'}
                    <span class="mini-cell" style="opacity: {v === 1 ? 1 : 0.12}"></span>
                  {:else}
                    <span
                      class="mini-cell"
                      style="background: {v < 0 ? 'rgba(122,162,255,0.8)' : 'rgba(255,184,107,0.8)'}; opacity: {Math.min(1, Math.abs(v) / 0.6) * 0.85 + 0.15}"
                    ></span>
                  {/if}
                {/each}
              </div>
              <span class="example-note">{example.leftNote ?? 'mostly zeros, no notion of "similar"'}</span>
            </div>
            <div class="example-col">
              <span class="example-label">{example.rightLabel ?? `Dense [${example.right.length}]`}</span>
              <div class="mini-strip">
                {#each example.right as v}
                  <span
                    class="mini-cell"
                    style="background: {v < 0 ? 'rgba(122,162,255,0.8)' : 'rgba(255,184,107,0.8)'}; opacity: {Math.min(1, Math.abs(v) / 0.6) * 0.85 + 0.15}"
                  ></span>
                {/each}
              </div>
              <span class="example-note">{example.rightNote ?? 'every dimension carries information'}</span>
            </div>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .why-toggle {
    background: none; border: none; color: var(--text-secondary, #9aa1b5);
    font-family: 'Inter', sans-serif; font-size: 0.78rem; cursor: pointer; padding: 0.25rem 0;
  }
  .why-body {
    border-left: 2px solid var(--accent-2, #ffb86b);
    padding-left: 0.75rem;
    margin-top: 0.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .why-item strong { font-size: 0.82rem; font-family: 'Inter', sans-serif; }
  .why-item p { margin: 0.15rem 0 0; font-size: 0.8rem; line-height: 1.5; opacity: 0.85; }

  .why-example { margin-top: 0.2rem; }
  .example-caption { font-size: 0.78rem; opacity: 0.7; margin: 0 0 0.5rem; }
  .example-row { display: flex; gap: 1.5rem; flex-wrap: wrap; }
  .example-col { display: flex; flex-direction: column; gap: 0.25rem; }
  .example-label { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; opacity: 0.7; }
  .mini-strip { display: flex; gap: 1px; }
  .mini-cell { width: 0.6rem; height: 1rem; border-radius: 1px; background: var(--accent, #7aa2ff); }
  .example-note { font-size: 0.65rem; opacity: 0.5; font-style: italic; }
</style>
