<script>
  export let isFullscreen = false;
</script>

<div class="layout" class:fullscreen={isFullscreen}>
  <div class="viz-pane">
    <slot name="viz" />
  </div>
  <div class="explain-pane">
    <slot name="explain" />
  </div>
  <div class="chrome-pane">
    <slot name="chrome" />
  </div>
</div>

<style>
  .layout {
    display: grid;
    grid-template-rows: 70% 30%;
    grid-template-areas: 'viz' 'explain';
    height: 100vh;
    width: 100vw;
    overflow: hidden;
    background: var(--bg-primary, #0b0e14);
    color: var(--text-primary, #e6e8ef);
  }
  .viz-pane {
    grid-area: viz;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .explain-pane {
    grid-area: explain;
    position: relative;
    overflow-y: auto;
    border-top: 1px solid var(--border-subtle, #232838);
    padding: 1rem 2rem;
  }
  .chrome-pane {
    position: absolute;
    inset: 0;
    pointer-events: none;
  }
  .chrome-pane :global(> *) {
    pointer-events: auto;
  }
</style>
