<script>
  import { fade, fly } from 'svelte/transition';
  import { cellColor } from '../core/heatmap-color.js';
  import { highlightedTermId, setHighlight, clearHighlight } from '../core/stores/highlightStore.js';
  import { subStepIndex, replayTick, dataMode, currentScene } from '../core/stores/sceneStore.js';
  import { dModel as configDModel, numHeads as configNumHeads, dK as configDK } from '../core/stores/configStore.js';
  import { forwardPassData } from '../core/data-loader.js';
  import { computeAttentionPipeline } from '../core/tensor-ops.js';
  import { getSceneCopy } from '../data/scene-copy.js';
  import { focusShot } from '../core/camera/cameraStore.js';
  import { motionMs, prefersReducedMotion } from '../core/motion.js';
  import { VOCAB_WORDS } from '../data/vocab.js';
  import TensorShapeTrace from './TensorShapeTrace.svelte';
  import CameraStage from './CameraStage.svelte';
  import VectorHeatmap from './VectorHeatmap.svelte';
  import BeforeAfterSummary from './BeforeAfterSummary.svelte';
  import QuickCheck from './QuickCheck.svelte';

  // Sub-step index definitions
  const STEP = { WEIGHT_VALUES: 0, SUM: 1, SUMMARY: 2, QUIZ: 3 };

  export let activeHeadIdx = 0;

  // Resolve config from stores
  $: sceneId = $currentScene?.id;
  $: copy = getSceneCopy(sceneId);

  const DEFAULT_INTERACTIVE_SENTENCE = ['cat', 'chased', 'dog'];
  const DEFAULT_TARGET_SENTENCE = ['the', 'dog', 'ran'];
  const MAX_INTERACTIVE_TOKENS = 6;

  let interactiveSentence = [...DEFAULT_INTERACTIVE_SENTENCE];
  let interactiveTargetSentence = [...DEFAULT_TARGET_SENTENCE];

  function addWord(word) {
    if (isDecoderStream) {
      if (interactiveTargetSentence.length >= MAX_INTERACTIVE_TOKENS) return;
      interactiveTargetSentence = [...interactiveTargetSentence, word];
    } else {
      if (interactiveSentence.length >= MAX_INTERACTIVE_TOKENS) return;
      interactiveSentence = [...interactiveSentence, word];
    }
  }

  function removeWord(index) {
    if (isDecoderStream) {
      interactiveTargetSentence = interactiveTargetSentence.filter((_, i) => i !== index);
    } else {
      interactiveSentence = interactiveSentence.filter((_, i) => i !== index);
    }
  }

  function resetSentence() {
    if (isDecoderStream) {
      interactiveTargetSentence = [...DEFAULT_TARGET_SENTENCE];
    } else {
      interactiveSentence = [...DEFAULT_INTERACTIVE_SENTENCE];
    }
  }

  // --- Data selection: Lecture vs Interactive ---
  $: lectureMeta = $forwardPassData?.meta ?? null;
  $: activeSentence = $dataMode === 'lecture'
    ? (isDecoderStream ? ['the', 'dog', 'ran', 'slowly'] : (lectureMeta?.sentence ?? []))
    : (isDecoderStream ? interactiveTargetSentence : interactiveSentence);

  $: resolvedLabels = activeSentence;
  $: seqLen = resolvedLabels.length;
  $: currentDModel = $dataMode === 'lecture' ? ($forwardPassData?.meta?.dModel ?? 16) : $configDModel;
  $: configNumHeadsVal = $dataMode === 'lecture' ? ($forwardPassData?.meta?.numHeads ?? 4) : $configNumHeads;
  $: configDKVal = $dataMode === 'lecture' ? (currentDModel / configNumHeadsVal) : $configDK;

  // Resolve weights from store for interactive calculations
  $: lectureWeights = $forwardPassData?.weights ?? {};

  // Lecture Mode Data Extraction
  $: lectureSoftmaxStage = $forwardPassData?.stages?.find((s) => s.id === 'scale-softmax') ?? null;
  $: lectureProjVStage = $forwardPassData?.stages?.find((s) => s.id === 'proj-v') ?? null;
  $: lectureWeightedSumStage = $forwardPassData?.stages?.find((s) => s.id === 'weighted-sum') ?? null;

  // Split precomputed concatenated Value & Output matrices into individual heads
  $: precomputedAttentionWeights = lectureSoftmaxStage?.attention ?? lectureSoftmaxStage?.extra?.attention ?? [];
  $: precomputedValueHeads = lectureProjVStage?.tokenVectors 
    ? splitHeads(lectureProjVStage.tokenVectors, configNumHeadsVal)
    : [];
  $: precomputedOutputHeads = lectureWeightedSumStage?.tokenVectors 
    ? splitHeads(lectureWeightedSumStage.tokenVectors, configNumHeadsVal)
    : [];
  $: precomputedConcatenatedOutput = lectureWeightedSumStage?.tokenVectors ?? [];

  // Live Interactive Calculations
  $: interactiveData = computeAttentionPipeline({
    encoderSentence: isDecoderStream ? ['cat', 'chased', 'dog'] : activeSentence,
    decoderSentence: isDecoderStream ? activeSentence : ['the', 'dog', 'ran'],
    dModel: currentDModel,
    numHeads: configNumHeadsVal,
    lectureWeights
  });

  $: stream = $currentScene?.config?.stream ?? 'encoder';
  $: attentionType = $currentScene?.config?.attentionType ?? 'self';
  $: isDecoderStream = stream === 'decoder';
  $: isCrossAttention = attentionType === 'cross';

  $: activePipeline = isCrossAttention
    ? interactiveData?.decoder?.crossAttention
    : (isDecoderStream ? interactiveData?.decoder : interactiveData);

  // Bind active matrices/vectors based on dataMode
  $: activeAttentionWeights = $dataMode === 'lecture' && !isDecoderStream
    ? (precomputedAttentionWeights[activeHeadIdx] ?? [])
    : (activePipeline?.weights?.[activeHeadIdx] ?? []);

  $: activeValueVectors = $dataMode === 'lecture' && !isDecoderStream
    ? (precomputedValueHeads[activeHeadIdx] ?? [])
    : (activePipeline?.Vh_trans?.[activeHeadIdx] ?? []);

  $: activeOutputVectors = $dataMode === 'lecture' && !isDecoderStream
    ? (precomputedOutputHeads[activeHeadIdx] ?? [])
    : (activePipeline?.outputHeads?.[activeHeadIdx] ?? []);

  $: concatenatedOutput = $dataMode === 'lecture'
    ? precomputedConcatenatedOutput
    : (interactiveData?.concatenatedOutput ?? []);

  let hoveredRow = null;
  let hoveredCol = null;

  function handleCellEnter(r, c) {
    hoveredRow = r;
    hoveredCol = c;
    setHighlight('eq-weighted-sum');
  }

  function handleCellLeave() {
    hoveredRow = null;
    hoveredCol = null;
    clearHighlight();
  }

  function handleKeyDown(e, r, c) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleCellEnter(r, c);
    }
  }

  // --- Camera director integration ---
  const WIDE = { x: 0, y: 0, scale: 1 };
  const SHOTS = [
    { x: 0, y: 8, scale: 1.05 }, // 0 WEIGHT_VALUES: Push slightly closer to matmul columns
    WIDE,                        // 1 SUM
    WIDE,                        // 2 SUMMARY
    WIDE,                        // 3 QUIZ
  ];

  let cameraBuildupTimer;
  $: {
    $replayTick;
    clearTimeout(cameraBuildupTimer);
    if ($subStepIndex === STEP.WEIGHT_VALUES) {
      focusShot(WIDE);
      cameraBuildupTimer = setTimeout(() => focusShot(SHOTS[0]), 500);
    } else {
      focusShot(SHOTS[$subStepIndex] ?? WIDE);
    }
  }

  // Tensor shape trace steps
  $: traceSteps = [
    { label: 'Weights A', shape: [seqLen, seqLen], highlightId: 'eq-weighted-sum' },
    { label: 'Values V', shape: [seqLen, configDKVal], highlightId: 'eq-weighted-sum' },
    { label: 'Concatenated Output', shape: [seqLen, currentDModel], highlightId: 'eq-weighted-sum' }
  ];

  $: activeTraceIndex = $subStepIndex === STEP.SUM ? 2 : ($subStepIndex === STEP.WEIGHT_VALUES ? 1 : 2);

  $: sentenceKey = [$dataMode, activeSentence.join('|'), $replayTick].join('::');
  $: stageKey = [$dataMode, activeSentence.join('|'), $subStepIndex, $replayTick].join('::');
</script>

<div class="weighted-sum-scene-wrap">
  {#if $dataMode === 'interactive' && ($subStepIndex === STEP.WEIGHT_VALUES || $subStepIndex === STEP.SUM)}
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
    {#if ($subStepIndex === STEP.WEIGHT_VALUES || $subStepIndex === STEP.SUM) && resolvedLabels.length > 0}
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
      {#if $subStepIndex === STEP.WEIGHT_VALUES}
        <div class="matmul-view-card" in:fly={{ y: 16, duration: motionMs(400), delay: motionMs(150) }}>
          <div class="matrix-header">
            <div class="title-block">
              <h3>Blending Values using Weights</h3>
              <p class="subtitle">Pairwise multiplication of Attention Weights (A) with Value vectors (V) for Head {activeHeadIdx}</p>
            </div>

            <!-- Head Selector Tabs -->
            <div class="head-selector">
              {#each Array(configNumHeadsVal) as _, h}
                <button 
                  class="head-tab" 
                  class:active={activeHeadIdx === h}
                  on:click={() => activeHeadIdx = h}
                >
                  Head {h}
                </button>
              {/each}
            </div>
          </div>

          <div class="matmul-columns">
            <!-- Column 1: Attention Weights Grid -->
            <div class="matmul-col font-sans">
              <div class="col-header">Weights <span class="mono">A</span> <span class="shape-note">[{seqLen} × {seqLen}]</span></div>
              <div class="grid-layout weights-grid">
                <div class="col-headers">
                  <div class="corner-space"></div>
                  {#each resolvedLabels as label, c}
                    <span class="col-header-label" class:active-col={hoveredCol === c}>{label}</span>
                  {/each}
                </div>

                {#each activeAttentionWeights as row, r}
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
                        aria-label={`row ${resolvedLabels[r]} Query, col ${resolvedLabels[c]} Key: weight ${val.toFixed(4)}`}
                        on:mouseenter={() => handleCellEnter(r, c)}
                        on:mouseleave={handleCellLeave}
                        on:focus={() => handleCellEnter(r, c)}
                        on:blur={handleCellLeave}
                        on:keydown={(e) => handleKeyDown(e, r, c)}
                      >
                        <span class="cell-val" class:visible={hoveredRow === r && hoveredCol === c}>
                          {val.toFixed(2)}
                        </span>
                      </div>
                    {/each}
                  </div>
                {/each}
              </div>
            </div>

            <!-- Multiplier Symbol -->
            <div class="operator-symbol">×</div>

            <!-- Column 2: Value Vectors -->
            <div class="matmul-col font-sans">
              <div class="col-header">Values <span class="mono">V</span> <span class="shape-note">[{seqLen} × {configDKVal}]</span></div>
              <div class="vector-stack">
                {#each activeValueVectors as vec, i (i)}
                  <div 
                    class="vector-strip" 
                    class:active={hoveredCol === i}
                    class:synced={$highlightedTermId === 'eq-weighted-sum'}
                    role="button"
                    tabindex="0"
                    on:mouseenter={() => { hoveredCol = i; }}
                    on:mouseleave={() => { hoveredCol = null; }}
                    on:focus={() => { hoveredCol = i; }}
                    on:blur={() => { hoveredCol = null; }}
                  >
                    <span class="row-word">{resolvedLabels[i]}</span>
                    <VectorHeatmap vector={vec} size="small" />
                  </div>
                {/each}
              </div>
            </div>

            <!-- Equals Symbol -->
            <div class="operator-symbol">=</div>

            <!-- Column 3: Output Vectors -->
            <div class="matmul-col font-sans">
              <div class="col-header">Output <span class="mono">O</span> <span class="shape-note">[{seqLen} × {configDKVal}]</span></div>
              <div class="vector-stack">
                {#each activeOutputVectors as vec, i (i)}
                  <div 
                    class="vector-strip" 
                    class:active={hoveredRow === i}
                    class:synced={$highlightedTermId === 'eq-weighted-sum'}
                    role="button"
                    tabindex="0"
                    on:mouseenter={() => { hoveredRow = i; }}
                    on:mouseleave={() => { hoveredRow = null; }}
                    on:focus={() => { hoveredRow = i; }}
                    on:blur={() => { hoveredRow = null; }}
                  >
                    <span class="row-word">{resolvedLabels[i]}</span>
                    <VectorHeatmap vector={vec} size="small" />
                  </div>
                {/each}
              </div>
            </div>
          </div>

          <!-- Color Legend -->
          <div class="legend">
            <span class="legend-neg">− Negative</span>
            <div class="legend-bar"></div>
            <span class="legend-pos">+ Positive</span>
          </div>
        </div>
      {/if}

      {#if $subStepIndex === STEP.SUM}
        <div class="sum-view-card" in:fade={{ duration: motionMs(250) }}>
          <div class="matrix-header">
            <div class="title-block">
              <h3>Head Concatenation</h3>
              <p class="subtitle">Output representations of all 4 heads concatenated side-by-side (shape: [seq_len, d_model = 16])</p>
            </div>
          </div>

          <div class="concatenated-vectors">
            {#each concatenatedOutput as vec, i (i)}
              <div class="vector-row-wrap" in:fly={{ x: -15, duration: motionMs(300), delay: motionMs(i * 80) }}>
                <span class="row-word font-semibold">{resolvedLabels[i]}</span>
                <div class="heads-row-container">
                  {#each Array(configNumHeadsVal) as _, h}
                    <div class="head-segment" class:active-segment={activeHeadIdx === h}>
                      <VectorHeatmap vector={vec.slice(h * configDKVal, (h + 1) * configDKVal)} size="small" />
                      <span class="segment-label">H{h}</span>
                    </div>
                  {/each}
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      {#if $subStepIndex === STEP.SUMMARY}
        <div class="beat" in:fade={{ duration: motionMs(300) }}>
          {#if copy.beforeAfter}
            <BeforeAfterSummary
              before={{ label: copy.beforeAfter.before.label, shape: [seqLen, seqLen] }}
              after={{ label: copy.beforeAfter.after.label, shape: [seqLen, currentDModel] }}
              whatChanged={copy.beforeAfter.whatChanged}
              structured={copy.beforeAfter.structured}
            />
          {/if}
        </div>
      {/if}

      {#if $subStepIndex === STEP.QUIZ}
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
  .weighted-sum-scene-wrap {
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

  .matmul-view-card,
  .sum-view-card {
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

  .matmul-columns {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.2rem;
    width: 100%;
    overflow-x: auto;
  }
  .matmul-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    background: rgba(0, 0, 0, 0.1);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.75rem;
    padding: 0.75rem;
  }
  .col-header { font-family: 'Inter', sans-serif; font-size: 0.68rem; opacity: 0.6; }

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

  .vector-stack {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .vector-strip {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.5rem;
    border: 1px solid transparent;
    border-radius: 0.4rem;
    background: rgba(255, 255, 255, 0.01);
    transition: border-color 0.2s, background-color 0.2s;
  }
  .vector-strip.active {
    background: rgba(255, 255, 255, 0.03);
    border-color: var(--accent-3, #7ee787);
  }
  .vector-strip.synced {
    background: rgba(126, 231, 135, 0.05);
  }
  .row-word {
    width: 3.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    text-align: right;
    font-weight: 500;
  }

  .operator-symbol {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    opacity: 0.5;
    user-select: none;
  }

  .concatenated-vectors {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.5rem 0;
  }
  .vector-row-wrap {
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .heads-row-container {
    display: flex;
    gap: 0.5rem;
    background: rgba(0, 0, 0, 0.15);
    padding: 0.3rem;
    border-radius: 0.5rem;
    border: 1px solid var(--border-subtle, #232838);
  }
  .head-segment {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.2rem;
    padding: 0.2rem 0.4rem;
    border-radius: 0.3rem;
    border: 1px solid transparent;
  }
  .head-segment.active-segment {
    border-color: var(--accent, #7aa2ff);
    background: rgba(122, 162, 255, 0.04);
  }
  .segment-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    opacity: 0.55;
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
  .legend-neg { color: rgba(122, 162, 255, 0.95); }
  .legend-pos { color: rgba(255, 184, 107, 0.95); }

  .cell:focus-visible,
  .vector-strip:focus-visible {
    outline: 2px solid var(--accent, #7aa2ff);
    outline-offset: 2px;
  }

  .shape-trace-wrap { margin-top: 0.25rem; }
</style>
