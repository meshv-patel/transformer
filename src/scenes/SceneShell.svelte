<script>
  import { currentScene, subStepIndex, replayTick, dataMode, appMode, deepDiveOpen, toggleDeepDive } from '../core/stores/sceneStore.js';
  import { explanationLevel } from '../core/stores/levelStore.js';
  import { getSceneCopy } from '../data/scene-copy.js';
  import { SCENE_VIZ } from './viz/index.js';
  import EquationPanel from '../components/EquationPanel.svelte';
  import CodePanel from '../components/CodePanel.svelte';
  import SpeakerNotes from '../components/SpeakerNotes.svelte';
  import WhyPanel from '../components/WhyPanel.svelte';

  $: scene = $currentScene;
  $: copy = getSceneCopy(scene?.id);
  $: bodyText = copy.body?.[$explanationLevel] || copy.body?.mtech || '';
  $: VizComponent = SCENE_VIZ[scene?.id];
  $: replayCount = $replayTick; // placeholder scenes surface this so replay is visibly wired even pre-Phase-4

  // Supports three forms, in priority order: an explicit `codeLines` array
  // (scaffolded for scenes with code panels authored independently of
  // `pytorch`), the generalized `pytorch` array form (multiple synced code
  // lines — needed once a scene has more than one, e.g. Positional
  // Encoding's formula + addition), or a single legacy string (auto-wrapped
  // with a generated id). Either way, a scene's own syncMap ids must match
  // whatever ids its viz component's hover handlers set on highlightStore
  // for the sync to actually connect.
  $: codeLines = copy.codeLines?.length
    ? copy.codeLines
    : Array.isArray(copy.pytorch)
      ? copy.pytorch
      : copy.pytorch
        ? [{ id: 'code-0', code: copy.pytorch }]
        : [];
  $: equationTerms = copy.equationTerms?.length
    ? copy.equationTerms
    : copy.deepDive?.math
      ? [{ id: 'eq-0', tex: copy.deepDive.math }]
      : [];
</script>

<div class="scene-shell">
  <div class="viz-area">
    {#if VizComponent}
      <svelte:component this={VizComponent} />
    {:else}
      <div class="not-yet">
        <span class="scene-id">{scene?.id}</span>
        <span class="not-yet-label">Animation not yet implemented</span>
        <span class="substep">sub-step {$subStepIndex + 1} / {scene?.subSteps?.length ?? 1} · replay #{replayCount}</span>
        {#if $dataMode === 'interactive'}
          <span class="mode-note">Interactive Mode — will recompute live once wired in Phase 3/4</span>
        {/if}
      </div>
    {/if}
  </div>

  <div class="explain-content">
    <div class="explain-header">
      <span class="eyebrow">{copy.eyebrow}</span>
      <h2 aria-live="polite">{copy.title || scene?.title}</h2>
    </div>

    <p class="body-text">{bodyText || 'Explanation not yet authored for this scene.'}</p>

    {#if copy.fourQuestions}
      <div class="four-questions">
        <div class="fq"><span class="fq-label">What</span><p>{copy.fourQuestions.whatIsHappening}</p></div>
        <div class="fq"><span class="fq-label">Why</span><p>{copy.fourQuestions.why}</p></div>
        <div class="fq"><span class="fq-label">Changed</span><p>{copy.fourQuestions.whatChanged}</p></div>
        <div class="fq"><span class="fq-label">Observe</span><p>{copy.fourQuestions.whatToObserve}</p></div>
      </div>
    {/if}

    {#if codeLines.length || equationTerms.length}
      <div class="sync-panels">
        {#if codeLines.length}
          <CodePanel lines={codeLines} />
        {/if}
        {#if equationTerms.length}
          <EquationPanel terms={equationTerms} />
        {/if}
      </div>
    {/if}

    {#if copy.whyPanel}
      <WhyPanel items={copy.whyPanel.items} example={copy.whyPanel.example} />
    {/if}

    {#if copy.deepDive}
      <div class="deep-dive">
        <button class="deep-dive-toggle" on:click={toggleDeepDive}>
          {$deepDiveOpen ? '▾' : '▸'} Deep Dive
        </button>
        {#if $deepDiveOpen}
          <div class="deep-dive-body">
            {#if copy.deepDive.complexity}
              <p><strong>Complexity:</strong> {copy.deepDive.complexity}</p>
            {/if}
            {#if copy.deepDive.matrixEquivalence}
              <p><strong>As a matrix multiply:</strong> {copy.deepDive.matrixEquivalence}</p>
            {/if}
            {#if copy.deepDive.misconceptions?.length}
              <p><strong>Common misconceptions</strong></p>
              <ul>{#each copy.deepDive.misconceptions as m}<li>{m}</li>{/each}</ul>
            {/if}
            {#if copy.deepDive.notes}
              <p><strong>Notes:</strong> {copy.deepDive.notes}</p>
            {/if}
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>

{#if $appMode === 'presenter'}
  <SpeakerNotes narration={copy.narration} />
{/if}

<style>
  .scene-shell { display: contents; }
  .viz-area { width: 100%; height: 100%; }

  .not-yet {
    width: 100%; height: 100%;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 0.4rem; opacity: 0.5; font-family: 'Inter', sans-serif;
  }
  .scene-id { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; }
  .not-yet-label { font-size: 0.85rem; }
  .substep { font-size: 0.7rem; }
  .mode-note { font-size: 0.7rem; color: var(--accent-2, #ffb86b); }

  .explain-content { max-width: 60rem; margin: 0 auto; }
  .eyebrow { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.6; }
  h2 { margin: 0.15rem 0 0.5rem; font-family: 'Space Grotesk', sans-serif; }
  .body-text { line-height: 1.5; margin-bottom: 0.75rem; }

  .four-questions {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
    gap: 0.5rem;
    margin-bottom: 0.85rem;
  }
  .fq {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.5rem;
    padding: 0.5rem 0.65rem;
  }
  .fq-label { font-family: 'Inter', sans-serif; font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.55; }
  .fq p { margin: 0.2rem 0 0; font-size: 0.78rem; line-height: 1.4; opacity: 0.9; }

  .sync-panels { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: flex-start; margin-bottom: 0.6rem; }

  .deep-dive-toggle {
    background: none; border: none; color: var(--text-secondary, #9aa1b5);
    font-family: 'Inter', sans-serif; font-size: 0.78rem; cursor: pointer; padding: 0.25rem 0;
  }
  .deep-dive-body {
    font-size: 0.8rem; line-height: 1.5; opacity: 0.85;
    border-left: 2px solid var(--border-subtle, #232838);
    padding-left: 0.75rem; margin-top: 0.4rem;
  }
  .deep-dive-body ul { margin: 0.2rem 0; padding-left: 1.1rem; }
</style>
