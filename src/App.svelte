<script>
  import { onMount } from 'svelte';
  import Layout from './components/Layout.svelte';
  import PresenterControls from './components/PresenterControls.svelte';
  import TimelineScrubber from './components/TimelineScrubber.svelte';
  import ModeToggle from './components/ModeToggle.svelte';
  import LevelToggle from './components/LevelToggle.svelte';
  import ConfigPanel from './components/ConfigPanel.svelte';
  import TokenJourneyPanel from './components/TokenJourneyPanel.svelte';
  import ExploreNav from './components/ExploreNav.svelte';
  import SceneShell from './scenes/SceneShell.svelte';
  import { appMode, toggleDeepDive } from './core/stores/sceneStore.js';
  import { loadLectureData } from './core/data-loader.js';

  let isFullscreen = false;

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      isFullscreen = true;
    } else {
      document.exitFullscreen();
      isFullscreen = false;
    }
  }

  onMount(async () => {
    try {
      await loadLectureData();
    } catch (e) {
      console.warn('Lecture data not available yet:', e.message);
    }
    document.addEventListener('fullscreenchange', () => {
      isFullscreen = !!document.fullscreenElement;
    });
  });
</script>

<Layout {isFullscreen}>
  <div slot="viz" style="width:100%; height:100%;">
    <SceneShell />
  </div>

  <div slot="explain"></div>
  <!-- SceneShell renders both viz + explain content directly (see its own
       markup) — Layout's two slots exist so a real per-scene component can
       later split cleanly across them without touching Layout itself. -->

  <div slot="chrome">
    <TimelineScrubber />

    {#if $appMode === 'presenter'}
      <PresenterControls
        {isFullscreen}
        onToggleFullscreen={toggleFullscreen}
        onToggleDeepDive={toggleDeepDive}
      />
    {:else}
      <ExploreNav />
    {/if}

    <div class="top-right-controls">
      <button class="mode-switch" on:click={() => appMode.update((m) => (m === 'presenter' ? 'explore' : 'presenter'))}>
        {$appMode === 'presenter' ? 'Switch to Explore' : 'Switch to Presenter'}
      </button>
      <ModeToggle />
      <LevelToggle />
    </div>

    <div class="left-bottom-controls">
      <ConfigPanel />
    </div>

    <div class="bottom-center-controls">
      <TokenJourneyPanel />
    </div>
  </div>
</Layout>

<style>
  .top-right-controls {
    position: fixed;
    top: 1rem;
    right: 1rem;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.5rem;
  }
  .mode-switch {
    background: rgba(20, 23, 34, 0.9);
    border: 1px solid var(--border-subtle, #232838);
    color: var(--text-primary, #e6e8ef);
    border-radius: 999px;
    padding: 0.35rem 0.8rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    cursor: pointer;
  }
  .left-bottom-controls {
    position: fixed;
    left: 1rem;
    bottom: 1rem;
  }
  .bottom-center-controls {
    position: fixed;
    left: 50%;
    bottom: 4.5rem;
    transform: translateX(-50%);
  }

  /* Below ~900px, several fixed panels (config, presenter controls, speaker
     notes, token journey) risk overlapping since they're all anchored near
     screen edges independently. Pull the config panel up to stack above
     presenter controls instead of sitting left of them. */
  @media (max-width: 900px) {
    .top-right-controls { top: 0.5rem; right: 0.5rem; gap: 0.35rem; }
    .left-bottom-controls {
      left: 50%;
      bottom: 4.5rem;
      transform: translateX(-50%);
    }
    .bottom-center-controls { bottom: 7.5rem; max-width: 92vw; }
  }
</style>
