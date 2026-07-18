<script>
  import { LEVELS, explanationLevel } from '../core/stores/levelStore.js';
</script>

<div class="toggle">
  {#each LEVELS as level}
    <button class:active={$explanationLevel === level.id} on:click={() => explanationLevel.set(level.id)}>
      {level.label}
    </button>
  {/each}
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
    white-space: nowrap;
  }
  button.active {
    background: var(--accent-2, #ffb86b);
    color: #0b0e14;
    font-weight: 600;
  }
</style>
