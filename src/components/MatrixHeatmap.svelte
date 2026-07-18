<script>
  import { fade, fly } from 'svelte/transition';
  import { cellColor } from '../core/heatmap-color.js';
  import { highlightedTermId, setHighlight, clearHighlight } from '../core/stores/highlightStore.js';
  import { subStepIndex, replayTick, dataMode, currentScene } from '../core/stores/sceneStore.js';
  import { dModel as configDModel, numHeads as configNumHeads, dK as configDK } from '../core/stores/configStore.js';
  import { forwardPassData } from '../core/data-loader.js';
  import { generateEmbeddingVector } from '../core/embedding-utils.js';
  import { sinusoidalPETable, addVectors } from '../core/positional-encoding.js';
  import { matmul, splitHeads, scale, transpose } from '../core/tensor-ops.js';
  import { getSceneCopy } from '../data/scene-copy.js';
  import { focusShot } from '../core/camera/cameraStore.js';
  import { motionMs, prefersReducedMotion } from '../core/motion.js';
  import { VOCAB_WORDS } from '../data/vocab.js';
  import TensorShapeTrace from './TensorShapeTrace.svelte';
  import CameraStage from './CameraStage.svelte';
  import BeforeAfterSummary from './BeforeAfterSummary.svelte';
  import QuickCheck from './QuickCheck.svelte';

  export let matrix = []; // 2D [seq, seq] or 3D [heads, seq, seq]
  export let labels = []; // Token words for row/column labels
  export let title = '';
  export let subtitle = '';
  export let showAllHeads = false; // If true, renders all heads side-by-side
  export let activeHeadIdx = 0; // The selected head index (for 3D matrix)
  export let onCellHover = null; // Callback (r, c, val)
  export let onCellUnhover = null; // Callback ()

  // Resolve config from stores or props
  $: sceneId = $currentScene?.id;
  $: copy = getSceneCopy(sceneId);

  // Sub-step mapping based on the active scene
  $: isSoftmaxScene = sceneId === 'scale-softmax';
  $: stepMap = isSoftmaxScene ? {
    ATTENTION_GRID: [0, 1], // divide-sqrt-dk and softmax-curve show the grid
    SUMMARY: 2,
    QUIZ: 3
  } : {
    ATTENTION_GRID: [0],    // qk-matmul and others only show grid on step 0
    SUMMARY: 1,
    QUIZ: 2
  };

  $: showGrid = Array.isArray(stepMap.ATTENTION_GRID)
    ? stepMap.ATTENTION_GRID.includes($subStepIndex)
    : $subStepIndex === stepMap.ATTENTION_GRID;

  // Interactive sentence management
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
    : (labels.length ? labels : interactiveSentence);

  $: resolvedLabels = labels.length ? labels : activeSentence;
  $: seqLen = resolvedLabels.length;
  $: currentDModel = $dataMode === 'lecture' ? ($forwardPassData?.meta?.dModel ?? 16) : $configDModel;

  // Resolve weights from store for interactive calculations
  $: lectureWeights = $forwardPassData?.weights ?? {};

  // Resolve the matrix data reactively
  $: resolvedMatrix = matrix.length ? matrix : (
    $dataMode === 'lecture'
      ? fetchLectureMatrix(sceneId, $subStepIndex, $forwardPassData)
      : computeInteractiveMatrix(sceneId, $subStepIndex, activeSentence, currentDModel, $configNumHeads, $configDK)
  );

  function fetchLectureMatrix(id, step, passData) {
    if (!passData) return [];
    if (id === 'qk-matmul') {
      const stage = passData.stages?.find((s) => s.id === 'qk-matmul');
      return stage?.attentionScores ?? stage?.extra?.attentionScores ?? [];
    }
    if (id === 'scale-softmax') {
      if (step === 0) {
        // Step 0: Division by sqrt(dk) -> retrieve precomputed scaled attention scores
        const stage = passData.stages?.find((s) => s.id === 'qk-matmul');
        return stage?.attentionScores ?? stage?.extra?.attentionScores ?? [];
      } else {
        // Step 1: Softmax weights -> retrieve softmax attention weights
        const stage = passData.stages?.find((s) => s.id === 'scale-softmax');
        return stage?.attention ?? stage?.extra?.attention ?? [];
      }
    }
    if (id === 'heads-compare' || id === 'weighted-sum') {
      const stage = passData.stages?.find((s) => s.id === 'scale-softmax');
      return stage?.attention ?? stage?.extra?.attention ?? [];
    }
    return [];
  }

  function computeInteractiveMatrix(id, step, sentence, dModelVal, numHeadsVal, dKVal) {
    if (!sentence || !sentence.length) return [];
    
    // 1. Get positional encoded vectors X_pe
    const embeds = sentence.map((w) => generateEmbeddingVector(w, dModelVal));
    const pes = sinusoidalPETable(sentence.length, dModelVal);
    const xPe = embeds.map((e, i) => addVectors(e, pes[i]));

    // 2. Extract weights
    const Wq = lectureWeights.Wq;
    const bq = lectureWeights.bq || new Array(dModelVal).fill(0);
    const Wk = lectureWeights.Wk;
    const bk = lectureWeights.bk || new Array(dModelVal).fill(0);

    if (!Wq || !Wk) return [];

    // 3. Compute projections
    const Q = matmul(xPe, Wq).map((row) => row.map((v, c) => v + bq[c]));
    const K = matmul(xPe, Wk).map((row) => row.map((v, c) => v + bk[c]));

    // 4. Split and compute raw scores
    const Qh = splitHeads(Q, numHeadsVal);
    const Kh = splitHeads(K, numHeadsVal);

    const outScores = [];
    for (let h = 0; h < numHeadsVal; h++) {
      const qHead = Qh[h];
      const kHead = Kh[h];
      const headScores = scale(matmul(qHead, transpose(kHead)), 1 / Math.sqrt(dKVal));
      
      if (id === 'qk-matmul' || (id === 'scale-softmax' && step === 0)) {
        outScores.push(headScores);
      } else {
        // Apply row-wise softmax
        const softmaxRows = (mat) => mat.map((row) => {
          const max = Math.max(...row);
          const exps = row.map((v) => Math.exp(v - max));
          const sum = exps.reduce((a, b) => a + b, 0);
          return exps.map((v) => v / sum);
        });
        outScores.push(softmaxRows(headScores));
      }
    }
    return outScores;
  }

  // Configuration strings
  $: resolvedTitle = title || (
    sceneId === 'qk-matmul' ? 'Q × Kᵀ (Scaled Attention Scores)' :
    sceneId === 'scale-softmax' 
      ? ($subStepIndex === 0 ? 'Scaling by 1/√d_k' : 'Softmax Attention Weights')
      : 'Attention Grid'
  );

  $: resolvedSubtitle = subtitle || (
    sceneId === 'qk-matmul' ? 'Raw score grid representing dot products between Queries and Keys' :
    sceneId === 'scale-softmax'
      ? ($subStepIndex === 0 
          ? 'Divide each dot product by the square root of d_k (d_k = 4) to prevent vanishing gradients' 
          : 'Normalized relevance weights where rows sum to exactly 1.0')
      : 'Color opacity represents value magnitude'
  );

  $: resolvedShowAllHeads = showAllHeads || (sceneId === 'heads-compare');

  // Multi-head tracking
  $: is3D = Array.isArray(resolvedMatrix[0]?.[0]);
  $: numHeads = is3D ? resolvedMatrix.length : 1;
  $: activeMatrix = is3D 
    ? (resolvedMatrix[activeHeadIdx] ?? []) 
    : resolvedMatrix;

  let hoveredRow = null;
  let hoveredCol = null;

  $: activeHighlightId = sceneId === 'scale-softmax' 
    ? 'eq-scale-softmax' 
    : (sceneId === 'qk-matmul' ? 'eq-qk-matmul' : 'eq-proj-q');

  function handleCellEnter(r, c, val) {
    hoveredRow = r;
    hoveredCol = c;
    setHighlight(activeHighlightId);
    if (onCellHover) {
      onCellHover(r, c, val);
    }
  }

  function handleCellLeave() {
    hoveredRow = null;
    hoveredCol = null;
    clearHighlight();
    if (onCellUnhover) {
      onCellUnhover();
    }
  }

  function handleKeyDown(e, r, c, val) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleCellEnter(r, c, val);
    }
  }

  // --- Camera director integration ---
  const WIDE = { x: 0, y: 0, scale: 1 };
  $: shotsList = isSoftmaxScene ? [
    { x: 0, y: 8, scale: 1.06 }, // 0 divide-sqrt-dk
    { x: 0, y: 8, scale: 1.06 }, // 1 softmax-curve
    WIDE,                        // 2 SUMMARY
    WIDE,                        // 3 QUIZ
  ] : [
    { x: 0, y: 8, scale: 1.06 }, // 0 ATTENTION_GRID
    WIDE,                        // 1 SUMMARY
    WIDE,                        // 2 QUIZ
  ];

  let cameraBuildupTimer;
  $: {
    $replayTick;
    clearTimeout(cameraBuildupTimer);
    const currentStep = $subStepIndex;
    if (currentStep === 0 || (isSoftmaxScene && currentStep === 1)) {
      focusShot(WIDE);
      cameraBuildupTimer = setTimeout(() => focusShot(shotsList[currentStep] || WIDE), 500);
    } else {
      focusShot(shotsList[currentStep] || WIDE);
    }
  }

  // Tensor shape steps
  $: traceSteps = isSoftmaxScene ? [
    { label: 'Raw Scores S', shape: [seqLen, seqLen], highlightId: activeHighlightId },
    { label: 'Scale Factor (1/√d_k)', shape: [], highlightId: activeHighlightId },
    { label: 'Attention Weights A', shape: [seqLen, seqLen], highlightId: activeHighlightId }
  ] : [
    { label: 'Q [seq, d_model]', shape: [seqLen, currentDModel], highlightId: activeHighlightId },
    { label: 'K [seq, d_model]', shape: [seqLen, currentDModel], highlightId: activeHighlightId },
    { label: 'Scores [seq, seq]', shape: [seqLen, seqLen], highlightId: activeHighlightId }
  ];

  $: activeTraceIndex = isSoftmaxScene 
    ? ($subStepIndex === 0 ? 1 : 2)
    : 2;

  $: sentenceKey = [$dataMode, activeSentence.join('|'), $replayTick].join('::');
  $: stageKey = [$dataMode, activeSentence.join('|'), $subStepIndex, $replayTick].join('::');
</script>

<CameraStage>
  <div class="matrix-scene">
    {#if $dataMode === 'interactive' && showGrid}
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
      <!-- Token header context -->
      {#if showGrid && resolvedLabels.length > 0}
        <div class="token-context-row" in:fade={{ duration: motionMs(250) }}>
          {#each resolvedLabels as word, i}
            <div class="token-tag" class:active-query={hoveredRow === i} class:active-key={hoveredCol === i}>
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
        {#if showGrid}
          <div class="matrix-heatmap-card" in:fly={{ y: 16, duration: motionMs(400), delay: motionMs(150) }}>
            <div class="matrix-header">
              <div class="title-block">
                <h3>{resolvedTitle}</h3>
                <p class="subtitle">{resolvedSubtitle}</p>
              </div>

              <!-- Head Selector Tabs (if 3D and not showing all heads) -->
              {#if is3D && !resolvedShowAllHeads}
                <div class="head-selector">
                  {#each Array(numHeads) as _, h}
                    <button 
                      class="head-tab" 
                      class:active={activeHeadIdx === h}
                      on:click={() => activeHeadIdx = h}
                    >
                      Head {h}
                    </button>
                  {/each}
                </div>
              {/if}
            </div>

            <div class="heatmap-wrapper" class:multi-grid={resolvedShowAllHeads && is3D}>
              {#if resolvedShowAllHeads && is3D}
                <!-- Render all heads side-by-side -->
                {#each resolvedMatrix as headMatrix, h}
                  <div class="single-head-grid">
                    <span class="head-label">Head {h}</span>
                    <div class="grid-layout">
                      <!-- Column headers (Key tokens) -->
                      <div class="col-headers">
                        <div class="corner-space"></div>
                        {#each resolvedLabels as label, c}
                          <span class="col-header-label" class:active-col={hoveredCol === c}>{label}</span>
                        {/each}
                      </div>

                      <!-- Rows -->
                      {#each headMatrix as row, r}
                        <div class="grid-row">
                          <span class="row-header-label" class:active-row={hoveredRow === r}>{resolvedLabels[r]}</span>
                          {#each row as val, c}
                            <!-- svelte-ignore a11y-interactive-supports-focus -->
                            <div
                              class="cell"
                              style="background: {cellColor(val)};"
                              role="button"
                              tabindex="0"
                              aria-label={`Head ${h}, row ${resolvedLabels[r]} Query, col ${resolvedLabels[c]} Key: value ${val.toFixed(4)}`}
                              title={`${resolvedLabels[r]} ↔ ${resolvedLabels[c]}: ${val.toFixed(4)}`}
                              in:fade={{ duration: 120, delay: prefersReducedMotion ? 0 : (r * seqLen + c) * 35 }}
                            ></div>
                          {/each}
                        </div>
                      {/each}
                    </div>
                  </div>
                {/each}
              {:else}
                <!-- Render single 2D grid -->
                <div class="grid-layout single">
                  <!-- Column headers (Key tokens) -->
                  <div class="col-headers">
                    <div class="corner-space"></div>
                    {#each resolvedLabels as label, c}
                      <span class="col-header-label" class:active-col={hoveredCol === c}>{label}</span>
                    {/each}
                  </div>

                  <!-- Rows -->
                  {#each activeMatrix as row, r}
                    <div class="grid-row">
                      <span class="row-header-label" class:active-row={hoveredRow === r}>{resolvedLabels[r]}</span>
                      {#each row as val, c}
                        <!-- svelte-ignore a11y-interactive-supports-focus -->
                        <div
                          class="cell"
                          class:hovered={hoveredRow === r && hoveredCol === c}
                          class:highlight-row={hoveredRow === r}
                          class:highlight-col={hoveredCol === c}
                          style="background: {cellColor(val)};"
                          role="button"
                          tabindex="0"
                          aria-label={`row ${resolvedLabels[r]} Query, col ${resolvedLabels[c]} Key: value ${val.toFixed(4)}`}
                          on:mouseenter={() => handleCellEnter(r, c, val)}
                          on:mouseleave={handleCellLeave}
                          on:focus={() => handleCellEnter(r, c, val)}
                          on:blur={handleCellLeave}
                          on:keydown={(e) => handleKeyDown(e, r, c, val)}
                          in:fade={{ duration: 120, delay: prefersReducedMotion ? 0 : (r * seqLen + c) * 35 }}
                        >
                          <span class="cell-val" class:visible={hoveredRow === r && hoveredCol === c}>
                            {val.toFixed(2)}
                          </span>
                        </div>
                      {/each}
                    </div>
                  {/each}
                </div>
              {/if}
            </div>

            <!-- Color Legend -->
            <div class="legend">
              <span class="legend-neg">− Negative</span>
              <div class="legend-bar"></div>
              <span class="legend-pos">+ Positive</span>
            </div>
          </div>
        {/if}

        {#if $subStepIndex === stepMap.SUMMARY}
          <div class="beat" in:fade={{ duration: motionMs(300) }}>
            {#if copy.beforeAfter}
              <BeforeAfterSummary
                before={{ label: copy.beforeAfter.before.label, shape: [seqLen, seqLen] }}
                after={{ label: copy.beforeAfter.after.label, shape: [seqLen, seqLen] }}
                whatChanged={copy.beforeAfter.whatChanged}
                structured={copy.beforeAfter.structured}
              />
            {/if}
          </div>
        {/if}

        {#if $subStepIndex === stepMap.QUIZ}
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

        <!-- Shape trace tracker -->
        <div class="shape-trace-wrap">
          <TensorShapeTrace
            steps={traceSteps}
            activeIndex={activeTraceIndex}
          />
        </div>
      </div>
    {/key}
  </div>
</CameraStage>

<style>
  .matrix-scene {
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
    transition: border-color 0.2s;
  }
  .token-tag.active-query {
    border-color: var(--accent, #7aa2ff);
  }
  .token-tag.active-key {
    border-color: var(--accent-2, #ffb86b);
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

  .matrix-heatmap-card {
    background: var(--bg-elevated, #141722);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.75rem;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    width: min(34rem, 90vw);
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

  .head-selector {
    display: flex;
    gap: 0.25rem;
    background: rgba(0, 0, 0, 0.2);
    padding: 0.15rem;
    border-radius: 0.35rem;
  }
  .head-tab {
    background: transparent;
    border: none;
    color: var(--text-secondary, #9aa1b5);
    font-size: 0.65rem;
    padding: 0.2rem 0.5rem;
    border-radius: 0.25rem;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }
  .head-tab.active {
    background: var(--border-subtle, #232838);
    color: var(--text-primary, #e6e8ef);
  }

  .heatmap-wrapper {
    display: flex;
    justify-content: center;
    overflow-x: auto;
    width: 100%;
    padding: 0.5rem 0;
  }
  .heatmap-wrapper.multi-grid {
    gap: 1.5rem;
    justify-content: flex-start;
  }

  .single-head-grid {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
  }
  .head-label {
    font-size: 0.7rem;
    font-weight: 500;
    opacity: 0.8;
  }

  .grid-layout {
    display: table;
    border-spacing: 2px;
  }

  .col-headers {
    display: table-row;
  }
  .corner-space {
    display: table-cell;
    width: 4rem;
  }
  .col-header-label {
    display: table-cell;
    width: 2.2rem;
    text-align: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    padding-bottom: 0.25rem;
    opacity: 0.65;
    transition: opacity 0.2s, color 0.2s;
  }
  .col-header-label.active-col {
    opacity: 1;
    color: var(--accent-2, #ffb86b);
  }

  .grid-row {
    display: table-row;
  }
  .row-header-label {
    display: table-cell;
    vertical-align: middle;
    text-align: right;
    padding-right: 0.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    width: 4rem;
    opacity: 0.65;
    transition: opacity 0.2s, color 0.2s;
  }
  .row-header-label.active-row {
    opacity: 1;
    color: var(--accent, #7aa2ff);
  }

  .cell {
    display: table-cell;
    width: 2.2rem;
    height: 2.2rem;
    border-radius: 3px;
    cursor: pointer;
    position: relative;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.03);
    transition: transform 0.15s, box-shadow 0.15s;
  }
  .cell:hover, .cell.hovered {
    transform: scale(1.08);
    box-shadow: 0 0 6px rgba(255, 255, 255, 0.25);
    z-index: 10;
  }
  .cell.highlight-row {
    box-shadow: inset 0 0 0 2px var(--accent, #7aa2ff);
  }
  .cell.highlight-col {
    box-shadow: inset 0 0 0 2px var(--accent-2, #ffb86b);
  }

  .cell-val {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    color: #0b0e14;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s;
  }
  .cell-val.visible {
    opacity: 0.95;
  }

  .legend {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    font-size: 0.68rem;
    opacity: 0.7;
    margin-top: 0.25rem;
  }
  .legend-bar {
    width: 6rem;
    height: 0.4rem;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(122, 162, 255, 0.9), rgba(20, 23, 34, 0.9), rgba(255, 184, 107, 0.9));
  }
  .legend-neg {
    color: rgba(122, 162, 255, 0.95);
  }
  .legend-pos {
    color: rgba(255, 184, 107, 0.95);
  }

  .cell:focus-visible {
    outline: 2px solid var(--text-primary, #e6e8ef);
    outline-offset: 1px;
    transform: scale(1.05);
    z-index: 11;
  }

  .shape-trace-wrap { margin-top: 0.25rem; }
</style>
