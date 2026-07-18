<script>
  import { next, prev, replayCurrent, sceneIndex, subStepIndex } from '../core/stores/sceneStore.js';
  import { SCENES } from '../core/scene-registry.js';

  export let isFullscreen = false;
  export let onToggleFullscreen = () => {};
  export let onToggleDeepDive = () => {};

  function handleKeydown(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    switch (e.key) {
      case 'ArrowRight':
      case ' ':
        e.preventDefault();
        next();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        prev();
        break;
      case 'f':
      case 'F':
        onToggleFullscreen();
        break;
      case 'd':
      case 'D':
        onToggleDeepDive();
        break;
    }
  }
</script>

<svelte:window on:keydown={handleKeydown} />

<div class="presenter-controls">
  <button class="nav-btn" on:click={prev} disabled={$sceneIndex === 0 && $subStepIndex === 0} aria-label="Previous">
    ←
  </button>

  <div class="scene-label">
    <span class="scene-title">{SCENES[$sceneIndex]?.title}</span>
    <span class="scene-count">{$sceneIndex + 1} / {SCENES.length}</span>
  </div>

  <button class="nav-btn" on:click={next} aria-label="Next">→</button>

  <button class="icon-btn" on:click={replayCurrent} aria-label="Replay" title="Replay (this scene)">⟲</button>
  <button class="icon-btn" on:click={onToggleFullscreen} aria-label="Fullscreen" title="Fullscreen (F)">
    {isFullscreen ? '⤡' : '⤢'}
  </button>
</div>

<style>
  .presenter-controls {
    position: fixed;
    bottom: 1rem;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: rgba(20, 23, 34, 0.9);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 999px;
    padding: 0.5rem 1rem;
    backdrop-filter: blur(8px);
  }
  .nav-btn, .icon-btn {
    background: transparent;
    border: none;
    color: var(--text-primary, #e6e8ef);
    font-size: 1.1rem;
    cursor: pointer;
    padding: 0.4rem 0.6rem;
    border-radius: 999px;
  }
  .nav-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .nav-btn:hover:not(:disabled), .icon-btn:hover { background: rgba(255,255,255,0.08); }
  .scene-label {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 10rem;
    font-family: 'Inter', sans-serif;
  }
  .scene-title { font-size: 0.9rem; font-weight: 600; }
  .scene-count { font-size: 0.7rem; opacity: 0.6; }
</style>
