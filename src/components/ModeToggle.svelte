<script>
  import { dataMode } from '../core/stores/sceneStore.js';
  import { resetToLectureDefaults } from '../core/stores/configStore.js';

  function setMode(mode) {
    dataMode.set(mode);
    if (mode === 'lecture') resetToLectureDefaults();
  }
</script>

<div class="toggle">
  <button class:active={$dataMode === 'lecture'} on:click={() => setMode('lecture')}>
    Lecture
  </button>
  <button class:active={$dataMode === 'interactive'} on:click={() => setMode('interactive')}>
    Interactive
  </button>
</div>

<style>
  .toggle {
    display: inline-flex;
    background: rgba(20, 23, 34, 0.85);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 999px;
    padding: 0.2rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
  }
  button {
    background: transparent;
    border: none;
    color: var(--text-secondary, #9aa1b5);
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    cursor: pointer;
  }
  button.active {
    background: var(--accent, #7aa2ff);
    color: #0b0e14;
    font-weight: 600;
  }
</style>
