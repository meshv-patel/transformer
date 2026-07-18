<script>
  import { fade, fly } from 'svelte/transition';
  import { highlightedTermId, setHighlight, clearHighlight } from '../core/stores/highlightStore.js';
  import { subStepIndex, replayTick, dataMode, currentScene } from '../core/stores/sceneStore.js';
  import { dModel as configDModel, numHeads as configNumHeads, dK as configDK } from '../core/stores/configStore.js';
  import { forwardPassData } from '../core/data-loader.js';
  import { splitHeads, computeAttentionPipeline } from '../core/tensor-ops.js';
  import { getSceneCopy } from '../data/scene-copy.js';
  import { focusShot } from '../core/camera/cameraStore.js';
  import { motionMs, prefersReducedMotion } from '../core/motion.js';
  import { VOCAB_WORDS } from '../data/vocab.js';
  import TensorShapeTrace from './TensorShapeTrace.svelte';
  import CameraStage from './CameraStage.svelte';
  import VectorHeatmap from './VectorHeatmap.svelte';
  import BeforeAfterSummary from './BeforeAfterSummary.svelte';
  import QuickCheck from './QuickCheck.svelte';

  const STEP = { RESHAPE: 0, TRANSPOSE: 1, SUMMARY: 2, QUIZ: 3 };

  export let mode = undefined;

  // Resolve scene config
  $: sceneId = $currentScene?.id;
  $: copy = getSceneCopy(sceneId);
  $: activeMode = mode || (sceneId === 'concat' ? 'concat' : 'split');
  $: isConcatScene = activeMode === 'concat';

  // Interactive sentence builder state
  const DEFAULT_INTERACTIVE_SENTENCE = ['cat', 'chased', 'dog'];
  const MAX_INTERACTIVE_TOKENS = 6;
  let interactiveSentence = [...DEFAULT_INTERACTIVE_SENTENCE];

  function addWord(word) {
    if (interactiveSentence.length >= MAX_INTERACTIVE_TOKENS) return;
    interactiveSentence = [...interactiveSentence, word];
  }
  function removeWord(index) {
    interactiveSentence = interactiveSentence.filter((_, i) => i !== index);
  }
  function resetSentence() {
    interactiveSentence = [...DEFAULT_INTERACTIVE_SENTENCE];
  }

  $: activeSentence = $dataMode === 'lecture' 
    ? ($forwardPassData?.meta?.sentence ?? []) 
    : interactiveSentence;

  $: resolvedLabels = activeSentence;
  $: seqLen = resolvedLabels.length;
  $: currentDModel = $dataMode === 'lecture' ? ($forwardPassData?.meta?.dModel ?? 16) : $configDModel;
  $: numHeadsVal = $dataMode === 'lecture' ? ($forwardPassData?.meta?.numHeads ?? 4) : $configNumHeads;
  $: dKVal = $dataMode === 'lecture' ? (currentDModel / numHeadsVal) : $configDK;

  // Weights loading
  $: lectureWeights = $forwardPassData?.weights ?? {};

  // Input matrix representation: Load Query projection (proj-q)
  $: projQStage = $forwardPassData?.stages?.find((s) => s.id === 'proj-q') ?? null;
  $: precomputedQ = projQStage?.tokenVectors ?? [];

  // Live calculations for Interactive Mode
  $: interactiveData = computeAttentionPipeline(
    activeSentence,
    currentDModel,
    numHeadsVal,
    dKVal,
    lectureWeights
  );

  // Resolve input matrix: Q for split-heads, Concatenated Weighted Sum for concat
  $: precomputedWS = $forwardPassData?.stages?.find((s) => s.id === 'weighted-sum')?.tokenVectors ?? [];
  $: inputMatrix = isConcatScene
    ? ($dataMode === 'lecture' ? precomputedWS : (interactiveData?.concatenatedOutput ?? []))
    : ($dataMode === 'lecture' ? precomputedQ : (interactiveData?.Q ?? []));

  // Reactively compute the reshaped matrix [seq_len, num_heads, d_k]
  $: reshapedMatrix = Array.from({ length: seqLen }, (_, i) => {
    if (!inputMatrix[i]) return [];
    return Array.from({ length: numHeadsVal }, (_, h) => {
      return inputMatrix[i].slice(h * dKVal, (h + 1) * dKVal);
    });
  });

  // Reactively compute the split/transposed matrix [num_heads, seq_len, d_k]
  $: transposedMatrix = inputMatrix.length ? splitHeads(inputMatrix, numHeadsVal) : [];

  // Active Head colors mapping
  const headColors = [
    'var(--accent, #7aa2ff)',
    'var(--accent-2, #ffb86b)',
    'var(--accent-3, #7ee787)',
    'var(--accent-4, #ff8a8a)'
  ];

  let hoveredTokenIdx = null;
  let hoveredHeadIdx = null;

  $: activeHighlightId = isConcatScene ? 'eq-concat' : 'eq-split-heads';

  function handleCellEnter(tIdx, hIdx) {
    hoveredTokenIdx = tIdx;
    hoveredHeadIdx = hIdx;
    setHighlight(activeHighlightId);
  }

  function handleCellLeave() {
    hoveredTokenIdx = null;
    hoveredHeadIdx = null;
    clearHighlight();
  }

  function handleKeyDown(e, tIdx, hIdx) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleCellEnter(tIdx, hIdx);
    }
  }

  // --- Camera Transitions ---
  const WIDE = { x: 0, y: 0, scale: 1 };
  const SHOTS = [
    { x: 0, y: 4, scale: 1.05 },  // 0 RESHAPE / STITCH: Zoom slightly
    { x: 0, y: 4, scale: 1.05 },  // 1 TRANSPOSE / SUMMARY: Keep in focus
    WIDE,                         // 2 SUMMARY / QUIZ
    WIDE,                         // 3 QUIZ (split-heads only)
  ];

  let cameraBuildupTimer;
  $: {
    $replayTick;
    clearTimeout(cameraBuildupTimer);
    if ($subStepIndex === 0 || (!isConcatScene && $subStepIndex === 1)) {
      focusShot(WIDE);
      cameraBuildupTimer = setTimeout(() => focusShot(SHOTS[$subStepIndex]), 500);
    } else {
      focusShot(SHOTS[$subStepIndex] ?? WIDE);
    }
  }

  // Shape Trace Setup
  $: traceSteps = isConcatScene ? [
    { label: 'Heads O_h', shape: [numHeadsVal, seqLen, dKVal], highlightId: activeHighlightId },
    { label: 'Stitched O', shape: [seqLen, numHeadsVal, dKVal], highlightId: activeHighlightId },
    { label: 'Concatenated O', shape: [seqLen, currentDModel], highlightId: activeHighlightId }
  ] : [
    { label: 'Input Q', shape: [seqLen, currentDModel], highlightId: activeHighlightId },
    { label: 'Reshaped Q', shape: [seqLen, numHeadsVal, dKVal], highlightId: activeHighlightId },
    { label: 'Transposed Q', shape: [numHeadsVal, seqLen, dKVal], highlightId: activeHighlightId }
  ];

  $: activeTraceIndex = isConcatScene
    ? 2
    : ($subStepIndex === STEP.TRANSPOSE ? 2 : ($subStepIndex === STEP.RESHAPE ? 1 : 2));

  $: sentenceKey = [$dataMode, activeSentence.join('|'), $replayTick].join('::');
  $: stageKey = [$dataMode, activeSentence.join('|'), $subStepIndex, $replayTick].join('::');
</script>

<div class="heads-split-scene-wrap">
  {#if $dataMode === 'interactive' && ($subStepIndex === 0 || (!isConcatScene && $subStepIndex === 1))}
    <div class="word-picker" transition:fade={{ duration: motionMs(150) }}>
      <span class="hint">Build a sentence:</span>
      {#each VOCAB_WORDS as w}
        <button class="chip" on:click={() => addWord(w)} disabled={interactiveSentence.length >= MAX_INTERACTIVE_TOKENS}>
          + {w}
        </button>
      {/each}
      <button class="chip reset" on:click={resetSentence}>Reset</button>
    </div>
  {/if}

  {#key sentenceKey}
    <!-- Context words indicators -->
    {#if ($subStepIndex === 0 || (!isConcatScene && $subStepIndex === 1)) && resolvedLabels.length > 0}
      <div class="token-context-row" in:fade={{ duration: motionMs(250) }}>
        {#each resolvedLabels as word, i}
          <div 
            class="token-tag" 
            class:active-token={hoveredTokenIdx === i}
            role="button"
            tabindex="0"
            on:mouseenter={() => hoveredTokenIdx = i}
            on:mouseleave={() => hoveredTokenIdx = null}
          >
            <span class="word">{word}</span>
            <span class="pos">pos {i}</span>
            {#if $dataMode === 'interactive'}
              <!-- svelte-ignore a11y-no-static-element-interactions -->
              <span
                class="remove-x"
                role="button"
                tabindex="0"
                on:click|stopPropagation={() => removeWord(i)}
                on:keydown|stopPropagation={(e) => e.key === 'Enter' && removeWord(i)}
              >×</span>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  {/key}

  {#key stageKey}
    <div class="stage-view">
      {#if !isConcatScene && $subStepIndex === STEP.RESHAPE}
        <!-- Step 0 (split-heads): RESHAPE -> [seq_len, num_heads, d_k] -->
        <div class="reshape-card" in:fly={{ y: 16, duration: motionMs(400), delay: motionMs(150) }}>
          <div class="matrix-header">
            <div class="title-block">
              <h3>Channel Reshaping</h3>
              <p class="subtitle">Splitting d_model = {currentDModel} into {numHeadsVal} heads of size d_k = {dKVal} (shape: [seq_len, num_heads, d_k])</p>
            </div>
          </div>

          <div class="token-vectors-split-container">
            {#each resolvedLabels as word, tIdx}
              <div 
                class="split-row" 
                class:active-row={hoveredTokenIdx === tIdx}
                in:fly={{ x: -10, duration: motionMs(300), delay: motionMs(tIdx * 80) }}
              >
                <div class="row-label-block">
                  <span class="word font-semibold">{word}</span>
                  <span class="pos-lbl">pos {tIdx}</span>
                </div>

                <div class="segments-container">
                  {#each Array(numHeadsVal) as _, hIdx}
                    {@const segment = reshapedMatrix[tIdx]?.[hIdx] ?? []}
                    <div 
                      class="head-segment-block"
                      style="border-color: {headColors[hIdx]};"
                      class:hovered={hoveredTokenIdx === tIdx && hoveredHeadIdx === hIdx}
                      class:faded={hoveredHeadIdx !== null && hoveredHeadIdx !== hIdx}
                      role="button"
                      tabindex="0"
                      on:mouseenter={() => handleCellEnter(tIdx, hIdx)}
                      on:mouseleave={handleCellLeave}
                      on:focus={() => handleCellEnter(tIdx, hIdx)}
                      on:blur={handleCellLeave}
                      on:keydown={(e) => handleKeyDown(e, tIdx, hIdx)}
                    >
                      <VectorHeatmap vector={segment} size="small" />
                      <span class="block-lbl" style="color: {headColors[hIdx]};">H{hIdx}</span>
                    </div>
                  {/each}
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      {#if !isConcatScene && $subStepIndex === STEP.TRANSPOSE}
        <!-- Step 1 (split-heads): TRANSPOSE -> [num_heads, seq_len, d_k] -->
        <div class="reshape-card" in:fade={{ duration: motionMs(250) }}>
          <div class="matrix-header">
            <div class="title-block">
              <h3>Dimension Transposition</h3>
              <p class="subtitle">Grouping token segments by Head (shape: [num_heads, seq_len, d_k]) so attention can run in parallel</p>
            </div>
          </div>

          <div class="heads-grid-container">
            {#each Array(numHeadsVal) as _, hIdx}
              <div 
                class="head-card-split" 
                style="border-color: {headColors[hIdx]}; background: rgba(0,0,0,0.1);"
                class:active-head-card={hoveredHeadIdx === hIdx}
                class:faded={hoveredHeadIdx !== null && hoveredHeadIdx !== hIdx}
                in:fly={{ y: 15, duration: motionMs(350), delay: motionMs(hIdx * 80) }}
              >
                <div class="head-card-header" style="color: {headColors[hIdx]};">
                  Head {hIdx}
                </div>

                <div class="head-vectors-stack">
                  {#each resolvedLabels as word, tIdx}
                    {@const segment = transposedMatrix[hIdx]?.[tIdx] ?? []}
                    <div 
                      class="head-vector-strip"
                      class:hovered={hoveredTokenIdx === tIdx && hoveredHeadIdx === hIdx}
                      class:active-token-strip={hoveredTokenIdx === tIdx}
                      role="button"
                      tabindex="0"
                      on:mouseenter={() => handleCellEnter(tIdx, hIdx)}
                      on:mouseleave={handleCellLeave}
                      on:focus={() => handleCellEnter(tIdx, hIdx)}
                      on:blur={handleCellLeave}
                      on:keydown={(e) => handleKeyDown(e, tIdx, hIdx)}
                    >
                      <span class="strip-label">{word}</span>
                      <VectorHeatmap vector={segment} size="small" />
                    </div>
                  {/each}
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      {#if isConcatScene && $subStepIndex === 0}
        <!-- Step 0 (concat): STITCH -> Side-by-side showing inputs and concatenated outputs -->
        <div class="reshape-card stitch-card" in:fly={{ y: 16, duration: motionMs(400), delay: motionMs(150) }}>
          <div class="matrix-header">
            <div class="title-block">
              <h3>Stitching Attention Heads</h3>
              <p class="subtitle">Concatenating the {numHeadsVal} head matrices of shape [{numHeadsVal}, seq_len, d_k = {dKVal}] back into a single representation of shape [seq_len, d_model = {currentDModel}]</p>
            </div>
          </div>

          <div class="concat-layout-columns">
            <!-- Left: Split Head outputs -->
            <div class="concat-left-col">
              <div class="col-section-header">Attention Head Outputs <span class="shape-note">[{numHeadsVal} × {seqLen} × {dKVal}]</span></div>
              <div class="heads-grid-container concat-heads">
                {#each Array(numHeadsVal) as _, hIdx}
                  <div 
                    class="head-card-split small" 
                    style="border-color: {headColors[hIdx]}; background: rgba(0,0,0,0.1);"
                    class:active-head-card={hoveredHeadIdx === hIdx}
                    class:faded={hoveredHeadIdx !== null && hoveredHeadIdx !== hIdx}
                  >
                    <div class="head-card-header text-center" style="color: {headColors[hIdx]}; font-size: 0.7rem; padding-bottom: 0.15rem;">
                      Head {hIdx}
                    </div>
                    <div class="head-vectors-stack">
                      {#each resolvedLabels as word, tIdx}
                        {@const segment = transposedMatrix[hIdx]?.[tIdx] ?? []}
                        <div 
                          class="head-vector-strip"
                          class:hovered={hoveredTokenIdx === tIdx && hoveredHeadIdx === hIdx}
                          class:active-token-strip={hoveredTokenIdx === tIdx}
                          role="button"
                          tabindex="0"
                          on:mouseenter={() => handleCellEnter(tIdx, hIdx)}
                          on:mouseleave={handleCellLeave}
                          on:focus={() => handleCellEnter(tIdx, hIdx)}
                          on:blur={handleCellLeave}
                          on:keydown={(e) => handleKeyDown(e, tIdx, hIdx)}
                        >
                          <span class="strip-label" style="width: 2.2rem; font-size: 0.62rem;">{word}</span>
                          <VectorHeatmap vector={segment} size="small" />
                        </div>
                      {/each}
                    </div>
                  </div>
                {/each}
              </div>
            </div>

            <!-- Concat operator arrow glyph -->
            <div class="concat-operator-arrow">
              <span class="arrow-glyph">➔</span>
              <span class="operator-lbl font-sans">Concat</span>
            </div>

            <!-- Right: Concatenated Output matrix -->
            <div class="concat-right-col">
              <div class="col-section-header">Concatenated Representation <span class="shape-note">[{seqLen} × {currentDModel}]</span></div>
              <div class="concatenated-vectors-split">
                {#each resolvedLabels as word, tIdx}
                  <div 
                    class="split-row" 
                    class:active-row={hoveredTokenIdx === tIdx}
                    style="padding: 0.3rem 0.6rem; gap: 0.6rem;"
                  >
                    <div class="row-label-block" style="width: 3rem;">
                      <span class="word" style="font-size: 0.72rem;">{word}</span>
                      <span class="pos-lbl" style="font-size: 0.55rem;">pos {tIdx}</span>
                    </div>

                    <div class="segments-container" style="gap: 0.3rem;">
                      {#each Array(numHeadsVal) as _, hIdx}
                        {@const segment = reshapedMatrix[tIdx]?.[hIdx] ?? []}
                        <div 
                          class="head-segment-block small"
                          style="border-color: {headColors[hIdx]}; padding: 0.15rem 0.25rem; background: rgba(0,0,0,0.15);"
                          class:hovered={hoveredTokenIdx === tIdx && hoveredHeadIdx === hIdx}
                          class:faded={hoveredHeadIdx !== null && hoveredHeadIdx !== hIdx}
                          role="button"
                          tabindex="0"
                          on:mouseenter={() => handleCellEnter(tIdx, hIdx)}
                          on:mouseleave={handleCellLeave}
                          on:focus={() => handleCellEnter(tIdx, hIdx)}
                          on:blur={handleCellLeave}
                          on:keydown={(e) => handleKeyDown(e, tIdx, hIdx)}
                        >
                          <VectorHeatmap vector={segment} size="small" />
                        </div>
                      {/each}
                    </div>
                  </div>
                {/each}
              </div>
            </div>
          </div>
        </div>
      {/if}

      {#if $subStepIndex === (isConcatScene ? 1 : STEP.SUMMARY)}
        <div class="beat" in:fade={{ duration: motionMs(300) }}>
          {#if copy.beforeAfter}
            <BeforeAfterSummary
              before={{ label: copy.beforeAfter.before.label, shape: isConcatScene ? [numHeadsVal, seqLen, dKVal] : [seqLen, currentDModel] }}
              after={{ label: copy.beforeAfter.after.label, shape: isConcatScene ? [seqLen, currentDModel] : [numHeadsVal, seqLen, dKVal] }}
              whatChanged={copy.beforeAfter.whatChanged}
              structured={copy.beforeAfter.structured}
            />
          {/if}
        </div>
      {/if}

      {#if $subStepIndex === (isConcatScene ? 2 : STEP.QUIZ)}
        <div class="beat" in:fade={{ duration: motionMs(300) }}>
          {#if copy.quickCheck}
            <QuickCheck
              question={copy.quickCheck.question}
              choices={copy.quickCheck.choices}
              correctIndex={copy.quickCheck.correctIndex}
              explanation={copy.quickCheck.explanation}
              transition={copy.quickCheck.transition}
              distractorNotes={copy.quickCheck.distractorNotes}
            />
          {/if}
        </div>
      {/if}

      <!-- Shape trace indicator -->
      <div class="shape-trace-wrap">
        <TensorShapeTrace
          steps={traceSteps}
          activeIndex={activeTraceIndex}
        />
      </div>
    </div>
  {/key}
</div>

<style>
  .heads-split-scene-wrap {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    padding: 1rem;
  }

  .word-picker {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex-wrap: wrap;
    justify-content: center;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    max-width: 44rem;
  }
  .hint { opacity: 0.6; margin-right: 0.25rem; }
  .chip {
    background: rgba(122, 162, 255, 0.12);
    border: 1px solid var(--accent-dim, #4a5578);
    color: var(--text-primary, #e6e8ef);
    border-radius: 999px;
    padding: 0.25rem 0.65rem;
    cursor: pointer;
    font-size: 0.75rem;
  }
  .chip:disabled { opacity: 0.3; cursor: not-allowed; }
  .chip.reset { background: transparent; border-color: var(--border-subtle, #232838); opacity: 0.7; }

  .token-context-row {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    min-height: 2.2rem;
  }
  .token-tag {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    background: var(--bg-elevated, #141722);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.4rem;
    padding: 0.3rem 0.6rem;
    font-size: 0.75rem;
    cursor: pointer;
    transition: border-color 0.2s;
  }
  .token-tag.active-token {
    border-color: var(--accent, #7aa2ff);
  }
  .token-tag .word { font-weight: 600; font-family: 'Space Grotesk', sans-serif; }
  .token-tag .pos { font-size: 0.6rem; opacity: 0.6; font-family: 'JetBrains Mono', monospace; }

  .remove-x {
    position: absolute;
    top: -0.3rem;
    right: -0.3rem;
    width: 0.85rem;
    height: 0.85rem;
    border-radius: 50%;
    background: var(--accent-4, #ff8a8a);
    color: #0b0e14;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.55rem;
    cursor: pointer;
  }

  .stage-view { display: flex; flex-direction: column; align-items: center; gap: 0.75rem; width: 100%; }
  .beat { display: flex; justify-content: center; width: 100%; padding: 1rem 0; }

  .reshape-card {
    background: var(--bg-elevated, #141722);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.75rem;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    width: min(56rem, 95vw);
    max-width: 100%;
  }

  .matrix-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    border-bottom: 1px solid var(--border-subtle, #232838);
    padding-bottom: 0.5rem;
  }
  .title-block h3 {
    margin: 0;
    font-size: 0.85rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
  }
  .subtitle {
    margin: 0.15rem 0 0;
    font-size: 0.7rem;
    opacity: 0.6;
  }

  .token-vectors-split-container {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    padding: 0.5rem 0;
  }

  .split-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(0, 0, 0, 0.12);
    padding: 0.4rem 0.8rem;
    border-radius: 0.5rem;
    border: 1px solid transparent;
    transition: background-color 0.2s, border-color 0.2s;
  }
  .split-row.active-row {
    background: rgba(255,255,255,0.02);
    border-color: var(--border-subtle, #232838);
  }

  .row-label-block {
    display: flex;
    flex-direction: column;
    width: 4.5rem;
    font-family: 'Space Grotesk', sans-serif;
  }
  .row-label-block .pos-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    opacity: 0.5;
  }

  .segments-container {
    display: flex;
    gap: 0.75rem;
    flex: 1;
  }

  .head-segment-block {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.2rem;
    padding: 0.3rem 0.5rem;
    border-radius: 0.4rem;
    border: 1px solid rgba(255,255,255,0.03);
    cursor: pointer;
    background: rgba(0,0,0,0.2);
    transition: border-color 0.2s, transform 0.2s, opacity 0.2s;
  }
  .head-segment-block:hover, .head-segment-block.hovered {
    transform: translateY(-2px);
    box-shadow: 0 0 4px rgba(255,255,255,0.08);
  }
  .head-segment-block.faded {
    opacity: 0.35;
  }
  .block-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
  }

  .heads-grid-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(11.5rem, 1fr));
    gap: 0.8rem;
    padding: 0.5rem 0;
  }

  .head-card-split {
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.6rem;
    padding: 0.6rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    transition: border-color 0.2s, box-shadow 0.2s, opacity 0.2s;
  }
  .head-card-split.active-head-card {
    box-shadow: 0 0 6px rgba(255, 255, 255, 0.05);
  }
  .head-card-split.faded {
    opacity: 0.35;
  }
  .head-card-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    padding-bottom: 0.3rem;
  }

  .head-vectors-stack {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .head-vector-strip {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.2rem 0.4rem;
    border-radius: 0.3rem;
    border: 1px solid transparent;
    cursor: pointer;
    background: rgba(255, 255, 255, 0.01);
    transition: background-color 0.15s, border-color 0.15s;
  }
  .head-vector-strip.hovered {
    background: rgba(255, 255, 255, 0.03);
    border-color: rgba(255,255,255,0.08);
  }
  .head-vector-strip.active-token-strip {
    background: rgba(122, 162, 255, 0.04);
  }

  .strip-label {
    width: 3rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    text-align: right;
    opacity: 0.8;
  }

  /* Concat specific styles */
  .concat-layout-columns {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    width: 100%;
  }

  .concat-left-col {
    flex: 1.2;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .concat-right-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .col-section-header {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    opacity: 0.65;
    text-align: center;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    padding-bottom: 0.25rem;
  }

  .concat-heads {
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
  }

  .head-card-split.small {
    padding: 0.4rem;
    gap: 0.3rem;
  }

  .concat-operator-arrow {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.2rem;
    opacity: 0.55;
    user-select: none;
  }
  .arrow-glyph {
    font-size: 1.6rem;
  }
  .operator-lbl {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .concatenated-vectors-split {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .head-segment-block.small {
    border-width: 1px;
    border-radius: 0.3rem;
  }

  .head-segment-block:focus-visible,
  .head-vector-strip:focus-visible {
    outline: 2px solid var(--accent, #7aa2ff);
    outline-offset: 2px;
  }

  .shape-trace-wrap { margin-top: 0.25rem; }
</style>
