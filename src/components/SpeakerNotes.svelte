<script>
  import { subStepIndex } from '../core/stores/sceneStore.js';

  // narration: array indexed by sub-step —
  // { duration, objective, script, audienceQuestion, expectedAnswer, misconception, transition }
  export let narration = [];

  let open = false;
  $: current = narration[$subStepIndex] ?? null;
</script>

<div class="narration-panel" class:open>
  <button class="toggle-btn" on:click={() => (open = !open)}>
    {open ? 'Hide' : 'Show'} narration {#if current}· {current.duration}{/if}
  </button>

  {#if open}
    {#if current}
      <div class="body">
        <div class="meta-row">
          {#if current.duration}<span class="badge">⏱ {current.duration}</span>{/if}
          {#if current.objective}<span class="badge objective">{current.objective}</span>{/if}
        </div>

        {#if current.script}
          <p class="script">{current.script}</p>
        {/if}

        {#if current.audienceQuestion}
          <section>
            <h5>Ask the room</h5>
            <p class="qa-q">"{current.audienceQuestion}"</p>
            {#if current.expectedAnswer}
              <p class="qa-a">Expected: {current.expectedAnswer}</p>
            {/if}
          </section>
        {/if}

        {#if current.misconception}
          <section>
            <h5>Common misconception</h5>
            <p>{current.misconception}</p>
          </section>
        {/if}

        {#if current.transition}
          <section class="transition-section">
            <h5>Transition</h5>
            <p class="transition-line">"{current.transition}"</p>
          </section>
        {/if}
      </div>
    {:else}
      <p class="empty">No narration authored for this sub-step yet.</p>
    {/if}
  {/if}
</div>

<style>
  .narration-panel {
    position: fixed;
    right: 1rem;
    bottom: 1rem;
    max-width: 22rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
  }

  @media (max-width: 900px) {
    .narration-panel { right: 0.5rem; bottom: 4.5rem; max-width: 16rem; }
    .body { max-height: 40vh; font-size: 0.7rem; }
  }

  .toggle-btn {
    background: rgba(20, 23, 34, 0.9);
    border: 1px solid var(--border-subtle, #232838);
    color: var(--text-primary, #e6e8ef);
    border-radius: 999px;
    padding: 0.4rem 0.9rem;
    cursor: pointer;
    white-space: nowrap;
  }
  .body {
    background: rgba(20, 23, 34, 0.96);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.75rem;
    padding: 0.85rem;
    margin-top: 0.5rem;
    max-height: 60vh;
    overflow-y: auto;
  }
  .meta-row { display: flex; gap: 0.4rem; margin-bottom: 0.5rem; flex-wrap: wrap; }
  .badge {
    background: rgba(122, 162, 255, 0.12);
    border: 1px solid var(--accent-dim, #4a5578);
    border-radius: 999px;
    padding: 0.15rem 0.55rem;
    font-size: 0.68rem;
  }
  .badge.objective { background: rgba(255, 184, 107, 0.1); border-color: rgba(255, 184, 107, 0.4); }
  .script { line-height: 1.55; opacity: 0.95; margin: 0 0 0.6rem; white-space: pre-line; }
  h5 { margin: 0.5rem 0 0.25rem; font-size: 0.68rem; text-transform: uppercase; opacity: 0.55; letter-spacing: 0.05em; }
  .qa-q { font-style: italic; opacity: 0.9; margin: 0; }
  .qa-a { opacity: 0.6; margin: 0.2rem 0 0; }
  .transition-section { border-top: 1px dashed var(--border-subtle, #232838); padding-top: 0.4rem; margin-top: 0.5rem; }
  .transition-line { opacity: 0.75; font-style: italic; margin: 0; }
  .empty { opacity: 0.5; padding: 0.5rem 0; }
</style>
