<script>
  import { SCENES } from '../core/scene-registry.js';
  import { goToScene, sceneIndex } from '../core/stores/sceneStore.js';
  import { IMPLEMENTED_SCENE_IDS } from '../scenes/viz/index.js';

  $: groups = SCENES.reduce((acc, scene, i) => {
    (acc[scene.group] ??= []).push({ ...scene, index: i });
    return acc;
  }, {});
</script>

<nav class="explore-nav">
  {#each Object.entries(groups) as [groupName, scenes]}
    <div class="group">
      <h6>{groupName}</h6>
      {#each scenes as scene}
        <button
          class:active={scene.index === $sceneIndex}
          class:unimplemented={!IMPLEMENTED_SCENE_IDS.has(scene.id)}
          on:click={() => goToScene(scene.index)}
        >
          {scene.title}
        </button>
      {/each}
    </div>
  {/each}
</nav>

<style>
  .explore-nav {
    position: fixed;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    background: rgba(20, 23, 34, 0.9);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.75rem;
    padding: 0.75rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    max-height: 80vh;
    overflow-y: auto;
  }

  .group { margin-bottom: 0.75rem; }

  /* A full-height left sidebar would otherwise cover a large fraction of a
     narrow screen's viz area — convert to a horizontally-scrolling bottom
     bar instead, the standard mobile-nav pattern. */
  @media (max-width: 900px) {
    .explore-nav {
      left: 0.5rem;
      right: 0.5rem;
      top: auto;
      bottom: 0.5rem;
      transform: none;
      max-height: 26vh;
      display: flex;
      flex-direction: row;
      gap: 0.75rem;
      overflow-x: auto;
      overflow-y: hidden;
    }
    .group { margin-bottom: 0; flex-shrink: 0; min-width: 7rem; }
  }

  h6 { margin: 0 0 0.35rem; font-size: 0.65rem; text-transform: uppercase; opacity: 0.5; letter-spacing: 0.05em; }
  button {
    display: block;
    width: 100%;
    text-align: left;
    background: none;
    border: none;
    color: var(--text-secondary, #9aa1b5);
    padding: 0.3rem 0.4rem;
    border-radius: 0.35rem;
    cursor: pointer;
  }
  button:hover { background: rgba(255,255,255,0.06); }
  button.active { background: var(--accent, #7aa2ff); color: #0b0e14; font-weight: 600; }
  button.unimplemented { opacity: 0.45; }
</style>
