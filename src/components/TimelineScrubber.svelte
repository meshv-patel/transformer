<script>
  import { SCENES } from '../core/scene-registry.js';
  import { sceneIndex, subStepIndex, goToScene, replayCurrent } from '../core/stores/sceneStore.js';
  import { IMPLEMENTED_SCENE_IDS } from '../scenes/viz/index.js';

  function jumpTo(i) {
    goToScene(i);
  }
</script>

<div class="scrubber">
  {#each SCENES as scene, i}
    <button
      class="marker"
      class:active={i === $sceneIndex}
      class:done={i < $sceneIndex}
      class:unimplemented={!IMPLEMENTED_SCENE_IDS.has(scene.id)}
      title={scene.title}
      aria-label={`Go to scene: ${scene.title}${i === $sceneIndex ? ' (current)' : ''}`}
      aria-current={i === $sceneIndex ? 'step' : undefined}
      on:click={() => jumpTo(i)}
    >
      {#if i === $sceneIndex}
        <span class="sub-ticks">
          {#each scene.subSteps as _, si}
            <span class="sub-tick" class:active={si === $subStepIndex}></span>
          {/each}
        </span>
      {/if}
    </button>
  {/each}
  <button class="replay-btn" on:click={replayCurrent} title="Replay current scene" aria-label="Replay current scene">⟲</button>
</div>

<style>
  .scrubber {
    position: fixed;
    top: 1rem;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    align-items: center;
    gap: 0.35rem;
    max-width: 90vw;
    overflow-x: auto;
    padding: 0.4rem 0.75rem;
    background: rgba(20, 23, 34, 0.85);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 999px;
    backdrop-filter: blur(8px);
  }
  .marker {
    width: 0.55rem;
    height: 0.55rem;
    border-radius: 50%;
    background: var(--border-subtle, #333a4d);
    border: none;
    cursor: pointer;
    flex: none;
    position: relative;
    padding: 0.6rem; /* larger click target than visible dot */
    background-clip: content-box;
  }
  .marker.done { background: var(--accent-dim, #4a5578); background-clip: content-box; }
  .marker.active { background: var(--accent, #7aa2ff); background-clip: content-box; transform: scale(1.3); }
  .marker.unimplemented { opacity: 0.4; }
  .sub-ticks {
    position: absolute;
    top: -0.5rem;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 2px;
  }
  .sub-tick {
    width: 3px;
    height: 3px;
    border-radius: 50%;
    background: var(--border-subtle, #333a4d);
  }
  .sub-tick.active { background: var(--accent, #7aa2ff); }
  .replay-btn {
    margin-left: 0.5rem;
    background: transparent;
    border: none;
    color: var(--text-primary, #e6e8ef);
    cursor: pointer;
    font-size: 0.9rem;
  }
</style>
