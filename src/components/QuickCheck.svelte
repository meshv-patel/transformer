<script>
  export let question = '';
  export let choices = [];
  export let correctIndex = 0;
  export let explanation = '';
  export let transition = '';
  export let distractorNotes = {}; // { [choiceIndex]: rebuttal text } — shown instead of/before the generic explanation

  let selected = null;
  let revealed = false;

  function pick(i) {
    if (revealed) return;
    selected = i;
    // A brief think-time before revealing, per "give the audience time to think."
    setTimeout(() => (revealed = true), 550);
  }

  function reset() {
    selected = null;
    revealed = false;
  }
</script>

<div class="quick-check">
  <span class="qc-eyebrow">Quick Check</span>
  <h3>{question}</h3>

  <div class="choices">
    {#each choices as choice, i}
      <button
        class="choice"
        class:selected={selected === i}
        class:correct={revealed && i === correctIndex}
        class:incorrect={revealed && selected === i && i !== correctIndex}
        disabled={revealed}
        on:click={() => pick(i)}
      >
        {choice}
      </button>
    {/each}
  </div>

  {#if revealed}
    <div class="reveal" aria-live="polite">
      {#if selected !== correctIndex && distractorNotes[selected]}
        <p class="explanation misconception">{distractorNotes[selected]}</p>
      {/if}
      <p class="explanation">{explanation}</p>
      {#if transition}
        <p class="transition">{transition}</p>
      {/if}
      <button class="try-again" on:click={reset}>Try again</button>
    </div>
  {/if}
</div>

<style>
  .quick-check {
    max-width: 34rem;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    text-align: center;
  }
  .qc-eyebrow { font-family: 'Inter', sans-serif; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.55; }
  h3 { font-family: 'Space Grotesk', sans-serif; margin: 0; }
  .choices { display: flex; flex-direction: column; gap: 0.5rem; width: 100%; }
  .choice {
    background: var(--bg-elevated, #141722);
    border: 1px solid var(--border-subtle, #232838);
    color: var(--text-primary, #e6e8ef);
    border-radius: 0.6rem;
    padding: 0.6rem 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
  }
  .choice:hover:not(:disabled) { border-color: var(--accent-dim, #4a5578); }
  .choice.selected { border-color: var(--accent, #7aa2ff); }
  .choice.correct { border-color: var(--accent-3, #7ee787); background: rgba(126, 231, 135, 0.1); }
  .choice.incorrect { border-color: var(--accent-4, #ff8a8a); background: rgba(255, 138, 138, 0.1); }
  .reveal { font-family: 'Inter', sans-serif; font-size: 0.85rem; line-height: 1.5; }
  .explanation { opacity: 0.9; }
  .explanation.misconception { opacity: 1; color: var(--accent-2, #ffb86b); }
  .transition { opacity: 0.6; font-style: italic; margin-top: 0.5rem; }
  .try-again { background: none; border: none; color: var(--text-secondary, #9aa1b5); font-size: 0.75rem; cursor: pointer; margin-top: 0.5rem; text-decoration: underline; }
</style>
