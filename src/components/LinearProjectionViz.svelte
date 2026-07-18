<script>
  import { fly, fade } from 'svelte/transition';
  import { cubicOut, quintOut } from 'svelte/easing';
  import { subStepIndex, replayTick, dataMode, currentScene } from '../core/stores/sceneStore.js';
  import { dModel as configDModel } from '../core/stores/configStore.js';
  import { forwardPassData } from '../core/data-loader.js';
  import { selectedTokenPos } from '../core/stores/tokenJourneyStore.js';
  import { highlightedTermId, setHighlight, clearHighlight } from '../core/stores/highlightStore.js';
  import { generateEmbeddingVector } from '../core/embedding-utils.js';
  import { sinusoidalPETable, addVectors } from '../core/positional-encoding.js';
  import { matmul, splitHeads, scale, transpose } from '../core/tensor-ops.js';
  import { VOCAB_WORDS } from '../data/vocab.js';
  import { getSceneCopy } from '../data/scene-copy.js';
  import { focusShot } from '../core/camera/cameraStore.js';
  import { motionMs, prefersReducedMotion } from '../core/motion.js';
  import TensorShapeTrace from './TensorShapeTrace.svelte';
  import CameraStage from './CameraStage.svelte';
  import VectorHeatmap from './VectorHeatmap.svelte';
  import BeforeAfterSummary from './BeforeAfterSummary.svelte';
  import QuickCheck from './QuickCheck.svelte';

  // Scene copy config
  $: copy = getSceneCopy($currentScene?.id);

  // Expose variables that can be overwritten by props
  export let role = undefined;
  export let weightKey = undefined;
  export let biasKey = undefined;
  export let stageId = undefined;
  export let inputStageId = undefined;

  // Reactively resolve parameters based on currentScene or props
  $: sceneId = $currentScene?.id;

  $: activeRole = role || (
    sceneId === 'proj-q' ? 'q' :
    sceneId === 'proj-k' ? 'k' :
    sceneId === 'proj-v' ? 'v' :
    sceneId === 'output-proj' ? 'o' : 'q'
  );

  $: activeWeightKey = weightKey || (
    sceneId === 'proj-q' ? 'Wq' :
    sceneId === 'proj-k' ? 'Wk' :
    sceneId === 'proj-v' ? 'Wv' :
    sceneId === 'output-proj' ? 'Wo' : 'Wq'
  );

  $: activeBiasKey = biasKey || (
    sceneId === 'proj-q' ? 'bq' :
    sceneId === 'proj-k' ? 'bk' :
    sceneId === 'proj-v' ? 'bv' :
    sceneId === 'output-proj' ? 'bo' : 'bq'
  );

  $: activeStageId = stageId || sceneId || 'proj-q';
  $: activeInputStageId = inputStageId || (
    sceneId === 'output-proj' ? 'weighted-sum' : 'positional-enc'
  );

  // Sub-step index definitions
  const STEP = { WEIGHTS: 0, MATMUL: 1, SUMMARY: 2, QUIZ: 3 };

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

  // --- Data selection: Lecture (precomputed) vs Interactive (live) ---
  $: lectureMeta = $forwardPassData?.meta ?? null;
  $: lectureInputStage = $forwardPassData?.stages?.find((s) => s.id === activeInputStageId) ?? null;
  $: lectureOutputStage = $forwardPassData?.stages?.find((s) => s.id === activeStageId) ?? null;
  $: lectureWeights = $forwardPassData?.weights ?? {};

  $: activeSentence = $dataMode === 'lecture' ? (lectureMeta?.sentence ?? []) : interactiveSentence;
  $: currentDModel = $dataMode === 'lecture' ? (lectureMeta?.dModel ?? 16) : $configDModel;
  $: seqLen = activeSentence.length;

  // Compute or select input vectors
  $: activeInputVectors = $dataMode === 'lecture'
    ? (lectureInputStage?.tokenVectors ?? [])
    : computeInteractiveInput(interactiveSentence, currentDModel);

  // Load weights
  $: activeWeightMatrix = $dataMode === 'lecture'
    ? (lectureWeights[activeWeightKey] ?? [])
    : (lectureWeights[activeWeightKey] ?? buildInteractiveWeights(currentDModel));

  $: activeBias = $dataMode === 'lecture'
    ? (lectureWeights[activeBiasKey] ?? [])
    : (lectureWeights[activeBiasKey] ?? new Array(currentDModel).fill(0));

  // Compute or select output vectors
  $: activeOutputVectors = $dataMode === 'lecture'
    ? (lectureOutputStage?.tokenVectors ?? [])
    : computeProjection(activeInputVectors, activeWeightMatrix, activeBias);

  function computeInteractiveInput(sentence, dModelVal) {
    if (sceneId === 'output-proj') {
      const numHeadsVal = 4;
      const dKVal = dModelVal / numHeadsVal;
      
      const embeds = sentence.map((w) => generateEmbeddingVector(w, dModelVal));
      const pes = sinusoidalPETable(sentence.length, dModelVal);
      const xPe = embeds.map((e, i) => addVectors(e, pes[i]));

      const Wq = lectureWeights.Wq;
      const bq = lectureWeights.bq || new Array(dModelVal).fill(0);
      const Wk = lectureWeights.Wk;
      const bk = lectureWeights.bk || new Array(dModelVal).fill(0);
      const Wv = lectureWeights.Wv;
      const bv = lectureWeights.bv || new Array(dModelVal).fill(0);

      if (!Wq || !Wk || !Wv) return Array.from({ length: sentence.length }, () => new Array(dModelVal).fill(0));

      const Q = matmul(xPe, Wq).map((row) => row.map((v, c) => v + bq[c]));
      const K = matmul(xPe, Wk).map((row) => row.map((v, c) => v + bk[c]));
      const V = matmul(xPe, Wv).map((row) => row.map((v, c) => v + bv[c]));

      const Qh = splitHeads(Q, numHeadsVal);
      const Kh = splitHeads(K, numHeadsVal);
      const Vh = splitHeads(V, numHeadsVal);

      const outConcatenated = Array.from({ length: sentence.length }, () => new Array(dModelVal).fill(0));

      for (let h = 0; h < numHeadsVal; h++) {
        const qHead = Qh[h];
        const kHead = Kh[h];
        const vHead = Vh[h];

        const headScores = scale(matmul(qHead, transpose(kHead)), 1 / Math.sqrt(dKVal));
        const softmaxRows = (mat) => mat.map((row) => {
          const max = Math.max(...row);
          const exps = row.map((v) => Math.exp(v - max));
          const sum = exps.reduce((a, b) => a + b, 0);
          return exps.map((v) => v / sum);
        });
        const headWeights = softmaxRows(headScores);

        const headOutput = matmul(headWeights, vHead);

        for (let i = 0; i < sentence.length; i++) {
          for (let d = 0; d < dKVal; d++) {
            outConcatenated[i][h * dKVal + d] = headOutput[i][d];
          }
        }
      }
      return outConcatenated;
    }
    const embeds = sentence.map((w) => generateEmbeddingVector(w, dModelVal));
    const pes = sinusoidalPETable(sentence.length, dModelVal);
    return embeds.map((e, i) => addVectors(e, pes[i]));
  }

  function buildInteractiveWeights(dModelVal) {
    const baseW = lectureWeights[activeWeightKey];
    if (baseW && baseW[0] && baseW[0].length === dModelVal) {
      return baseW.map(row => row.slice(0, dModelVal));
    }
    // Simple placeholder identity matrix of size dModelVal
    return Array.from({ length: dModelVal }, (_, i) => 
      Array.from({ length: dModelVal }, (_, j) => (i === j ? 1 : 0))
    );
  }

  function computeProjection(inputs, w, b) {
    if (!inputs.length || !w.length) return [];
    const proj = matmul(inputs, w);
    return proj.map((row) => row.map((val, colIdx) => val + (b[colIdx] ?? 0)));
  }

  // --- Dynamic Styling based on Projection Role ---
  $: accentColor = activeRole === 'q' 
    ? 'var(--accent, #7aa2ff)' 
    : activeRole === 'k' 
      ? 'var(--accent-2, #ffb86b)' 
      : activeRole === 'v' 
        ? 'var(--accent-3, #7ee787)' 
        : 'var(--accent-4, #ff8a8a)';
  $: accentDim = activeRole === 'q' 
    ? 'var(--accent-dim, #4a5578)' 
    : 'rgba(255, 255, 255, 0.15)';

  $: activeHighlightId = activeRole === 'k' 
    ? 'eq-proj-k' 
    : (activeRole === 'v' ? 'eq-proj-v' : (activeRole === 'o' ? 'eq-output-proj' : 'eq-proj-q'));

  // --- Camera Director: authored shots per sub-step ---
  const WIDE = { x: 0, y: 0, scale: 1 };
  const SHOTS = [
    { x: 0, y: 14, scale: 1.05 },  // 0 WEIGHTS: push toward the weight matrix
    WIDE,                          // 1 MATMUL: show whole operation wide
    WIDE,                          // 2 SUMMARY
    WIDE,                          // 3 QUIZ
  ];
  let cameraBuildupTimer;
  $: {
    $replayTick; // re-run on replay tick
    clearTimeout(cameraBuildupTimer);
    if ($subStepIndex === STEP.WEIGHTS) {
      focusShot(WIDE);
      cameraBuildupTimer = setTimeout(() => focusShot(SHOTS[0]), 500);
    } else {
      focusShot(SHOTS[$subStepIndex] ?? WIDE);
    }
  }

  $: sentenceKey = [$dataMode, activeSentence.join('|'), $replayTick].join('::');
  $: stageKey = [$dataMode, activeSentence.join('|'), $subStepIndex, $replayTick].join('::');

  let activeHoverIdx = null;

  function selectToken(i) {
    selectedTokenPos.set($selectedTokenPos === i ? null : i);
  }

  function handleHover(i) {
    activeHoverIdx = i;
    setHighlight(activeHighlightId);
  }

  function handleUnhover() {
    activeHoverIdx = null;
    clearHighlight();
  }
  $: inputLabel = sceneId === 'output-proj' ? 'Concatenated Heads' : 'Input X_pe';
  $: weightLabel = sceneId === 'output-proj' ? 'Weights Wo' : `Weights W_${activeRole}`;
  $: outputLabel = sceneId === 'output-proj' ? 'Output Y' : `Output ${activeRole.toUpperCase()}`;
</script>

<CameraStage>
  <div class="projection-scene">
    {#if $dataMode === 'interactive'}
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
      <div class="token-row" class:dimmed={$subStepIndex >= STEP.SUMMARY} in:fade={{ duration: motionMs(250) }}>
        {#each activeSentence as word, i (word + '-' + i)}
          <button
            class="token-card"
            class:selected={$selectedTokenPos === i}
            on:click={() => selectToken(i)}
            on:mouseenter={() => handleHover(i)}
            on:mouseleave={handleUnhover}
            on:focus={() => handleHover(i)}
            on:blur={handleUnhover}
            in:fly={{ y: -10, duration: motionMs(350), delay: motionMs(i * 90), easing: quintOut }}
          >
            <span class="token-word">{word}</span>
            <span class="token-pos">pos {i}</span>
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
          </button>
        {/each}
        {#if activeSentence.length === 0}
          <p class="empty-hint">Add a word above to see it projected.</p>
        {/if}
      </div>
    {/key}

    {#key stageKey}
      <div class="stage">
        {#if $subStepIndex === STEP.WEIGHTS}
          <div class="weight-view" in:fly={{ y: 16, duration: motionMs(400), delay: motionMs(150) }}>
            <div class="matrix-title">
              Weight Matrix <span class="mono">W_{activeRole}</span>
              <span class="shape-note">[{currentDModel} × {currentDModel}]</span>
            </div>
            <div class="matrix-grid" style="border-color: {accentDim};">
              {#each activeWeightMatrix as row, r (r)}
                <div class="matrix-row">
                  <VectorHeatmap vector={row.slice(0, currentDModel)} size="small" />
                </div>
              {/each}
            </div>
            {#if activeBias && activeBias.length > 0}
              <div class="bias-view">
                <div class="matrix-title">Bias Vector <span class="mono">b_{activeRole}</span> <span class="shape-note">[{currentDModel}]</span></div>
                <div class="vector-strip">
                  <VectorHeatmap vector={activeBias.slice(0, currentDModel)} size="small" />
                </div>
              </div>
            {/if}
          </div>
        {/if}

        {#if $subStepIndex === STEP.MATMUL && seqLen > 0}
          <div class="matmul-view" in:fade={{ duration: motionMs(250) }}>
            <!-- Column 1: Input Vectors (X_pe) -->
            <div class="matmul-col">
              <div class="col-header">Input <span class="mono">X_pe</span> <span class="shape-note">[{seqLen} × {currentDModel}]</span></div>
              <div class="vector-stack">
                {#each activeInputVectors as vec, i (i)}
                  <div 
                    class="vector-strip" 
                    class:active={activeHoverIdx === i || $selectedTokenPos === i}
                    class:synced={$highlightedTermId === activeHighlightId}
                    on:mouseenter={() => handleHover(i)}
                    on:mouseleave={handleUnhover}
                    on:focus={() => handleHover(i)}
                    on:blur={handleUnhover}
                    role="button"
                    tabindex="0"
                    on:click={() => selectToken(i)}
                    on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectToken(i); } }}
                    in:fly={{ x: -10, duration: motionMs(300), delay: motionMs(i * 80) }}
                  >
                    <span class="row-word">{activeSentence[i]}</span>
                    <VectorHeatmap vector={vec} size="small" />
                  </div>
                {/each}
              </div>
            </div>

            <!-- Operator Symbol -->
            <div class="operator-symbol">×</div>

            <!-- Column 2: Weight Matrix -->
            <div class="matmul-col">
              <div class="col-header">Weights <span class="mono">W_{activeRole}</span> <span class="shape-note">[{currentDModel} × {currentDModel}]</span></div>
              <div class="matrix-grid mini" style="border-color: {accentDim};">
                {#each activeWeightMatrix as row, r (r)}
                  <div class="matrix-row">
                    <VectorHeatmap vector={row.slice(0, currentDModel)} size="small" />
                  </div>
                {/each}
              </div>
            </div>

            <!-- Operator Symbol -->
            <div class="operator-symbol">=</div>

            <!-- Column 3: Output Vectors (Q/K/V) -->
            <div class="matmul-col">
              <div class="col-header">Output <span class="mono">{activeRole.toUpperCase()}</span> <span class="shape-note">[{seqLen} × {currentDModel}]</span></div>
              <div class="vector-stack">
                {#each activeOutputVectors as vec, i (i)}
                  <div 
                    class="vector-strip" 
                    class:active={activeHoverIdx === i || $selectedTokenPos === i}
                    class:synced={$highlightedTermId === activeHighlightId}
                    style="border-color: {(activeHoverIdx === i || $selectedTokenPos === i) ? accentColor : 'transparent'};"
                    on:mouseenter={() => handleHover(i)}
                    on:mouseleave={handleUnhover}
                    on:focus={() => handleHover(i)}
                    on:blur={handleUnhover}
                    role="button"
                    tabindex="0"
                    on:click={() => selectToken(i)}
                    on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectToken(i); } }}
                    in:fly={{ x: 10, duration: motionMs(300), delay: motionMs(i * 80) }}
                  >
                    <span class="row-word">{activeSentence[i]}</span>
                    <VectorHeatmap vector={vec} size="small" />
                  </div>
                {/each}
              </div>
            </div>
          </div>

          <div class="legend" in:fade={{ duration: motionMs(300), delay: motionMs(400) }}>
            <span class="legend-neg">−</span>
            <span class="legend-bar"></span>
            <span class="legend-pos">+</span>
            <span class="legend-label">values projected at each dimension</span>
          </div>
        {/if}

        {#if $subStepIndex === STEP.SUMMARY}
          <div class="beat" in:fade={{ duration: motionMs(300) }}>
            {#if copy.beforeAfter}
              <BeforeAfterSummary
                before={{ label: copy.beforeAfter.before.label, shape: [activeSentence.length || 0, currentDModel] }}
                after={{ label: copy.beforeAfter.after.label, shape: [activeSentence.length || 0, currentDModel] }}
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

        <div class="shape-trace-wrap">
          <TensorShapeTrace
            steps={[
              { label: inputLabel, shape: [seqLen, currentDModel], highlightId: activeHighlightId },
              { label: weightLabel, shape: [currentDModel, currentDModel], highlightId: activeHighlightId },
              { label: outputLabel, shape: [seqLen, currentDModel], highlightId: activeHighlightId }
            ]}
            activeIndex={$subStepIndex === STEP.WEIGHTS ? 1 : 2}
          />
        </div>
      </div>
    {/key}
  </div>
</CameraStage>

<style>
  .projection-scene {
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

  .token-row {
    display: flex; gap: 0.6rem; flex-wrap: wrap; justify-content: center; min-height: 3.5rem;
    transition: opacity 0.4s ease, transform 0.4s ease;
  }
  .token-row.dimmed { opacity: 0.3; transform: scale(0.92); }

  .token-card {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    background: var(--bg-elevated, #141722);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.6rem;
    padding: 0.5rem 0.9rem;
    cursor: pointer;
    color: var(--text-primary, #e6e8ef);
    transition: border-color 0.2s, transform 0.2s;
  }
  .token-card:hover { border-color: var(--accent-dim, #4a5578); transform: translateY(-2px); }
  .token-card.selected { border-color: var(--accent, #7aa2ff); box-shadow: 0 0 0 1px var(--accent, #7aa2ff); }
  .token-word { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.95rem; }
  .token-pos { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; opacity: 0.6; }
  .remove-x {
    position: absolute; top: -0.4rem; right: -0.4rem;
    width: 1rem; height: 1rem; border-radius: 50%;
    background: var(--accent-4, #ff8a8a); color: #0b0e14;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; cursor: pointer;
  }
  .empty-hint { opacity: 0.55; font-family: 'Inter', sans-serif; font-size: 0.8rem; }

  .stage { display: flex; flex-direction: column; align-items: center; gap: 0.75rem; width: 100%; }
  .beat { display: flex; justify-content: center; width: 100%; padding: 1rem 0; }

  .weight-view {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    background: var(--bg-elevated, #141722);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.75rem;
    padding: 0.75rem 1rem;
    max-width: 90vw;
  }
  .bias-view {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.25rem;
    margin-top: 0.4rem;
    border-top: 1px solid var(--border-subtle, #232838);
    padding-top: 0.5rem;
    width: 100%;
  }

  .matrix-title { font-family: 'Inter', sans-serif; font-size: 0.72rem; opacity: 0.7; }
  .mono { font-family: 'JetBrains Mono', monospace; }
  .shape-note { opacity: 0.5; margin-left: 0.3rem; }

  .matrix-grid {
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 0.25rem;
    border: 1px solid transparent;
    border-radius: 0.4rem;
  }
  .matrix-grid.mini {
    padding: 0.15rem;
  }
  .matrix-row { display: flex; gap: 1px; }

  .matmul-view {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.2rem;
    width: 100%;
    max-width: 58rem;
  }
  .matmul-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    background: var(--bg-elevated, #141722);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.75rem;
    padding: 0.75rem;
  }
  .col-header { font-family: 'Inter', sans-serif; font-size: 0.68rem; opacity: 0.6; }
  
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
  }
  .vector-strip.synced {
    background: rgba(122, 162, 255, 0.05);
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

  .legend { display: flex; align-items: center; gap: 0.4rem; font-family: 'Inter', sans-serif; font-size: 0.7rem; opacity: 0.65; }
  .legend-bar {
    width: 5rem; height: 0.5rem; border-radius: 999px;
    background: linear-gradient(90deg, rgba(122,162,255,0.9), rgba(20,23,34,0.9), rgba(255,184,107,0.9));
  }

  .shape-trace-wrap { margin-top: 0.25rem; }

  .token-card:focus-visible,
  .vector-strip:focus-visible {
    outline: 2px solid var(--accent, #7aa2ff);
    outline-offset: 2px;
  }

  @media (prefers-reduced-motion: reduce) {
    .matrix-grid { transition: none; }
  }
</style>
