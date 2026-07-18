<script>
  import { fly, fade } from 'svelte/transition';
  import { crossfade } from 'svelte/transition';
  import { cubicOut, quintOut } from 'svelte/easing';
  import { subStepIndex, replayTick, dataMode } from '../../core/stores/sceneStore.js';
  import { dModel as configDModel } from '../../core/stores/configStore.js';
  import { forwardPassData } from '../../core/data-loader.js';
  import { selectedTokenPos } from '../../core/stores/tokenJourneyStore.js';
  import { highlightedTermId, setHighlight, clearHighlight } from '../../core/stores/highlightStore.js';
  import { generateEmbeddingVector } from '../../core/embedding-utils.js';
  import { sinusoidalPositionEncoding, sinusoidalPETable, addVectors } from '../../core/positional-encoding.js';
  import { VOCAB_WORDS } from '../../data/vocab.js';
  import { SCENE_COPY } from '../../data/scene-copy.js';
  import { focusShot } from '../../core/camera/cameraStore.js';
  import { motionMs, prefersReducedMotion } from '../../core/motion.js';
  import TensorShapeTrace from '../../components/TensorShapeTrace.svelte';
  import CameraStage from '../../components/CameraStage.svelte';
  import BeforeAfterSummary from '../../components/BeforeAfterSummary.svelte';
  import QuickCheck from '../../components/QuickCheck.svelte';
  import VectorHeatmap from '../../components/VectorHeatmap.svelte';

  const copy = SCENE_COPY['positional-enc'];

  // Named sub-step constants — see docs/reference-implementation.md
  // "External review response" §1 for why (readability, zero behavior change).
  const STEP = { PROBLEM: 0, TABLE: 1, COMBINE: 2, OUTPUT: 3, SUMMARY: 4, QUIZ: 5 };

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

  // --- Data selection: Lecture (real precomputed) vs Interactive (live) ---
  // Positional encoding has no "not trained" disclosure, unlike Embedding's
  // Interactive Mode — it's a fixed formula in both modes, verified to
  // match the offline-generated Lecture data exactly (see core/positional-encoding.js).
  $: lectureMeta = $forwardPassData?.meta ?? null;
  $: lectureEmbeddingStage = $forwardPassData?.stages?.find((s) => s.id === 'embedding') ?? null;
  $: lecturePEStage = $forwardPassData?.stages?.find((s) => s.id === 'positional-enc') ?? null;

  $: activeSentence = $dataMode === 'lecture' ? (lectureMeta?.sentence ?? []) : interactiveSentence;
  $: currentDModel = $dataMode === 'lecture' ? (lectureMeta?.dModel ?? 16) : $configDModel;
  $: seqLen = activeSentence.length;

  $: embeddingVectors = $dataMode === 'lecture'
    ? (lectureEmbeddingStage?.tokenVectors ?? [])
    : interactiveSentence.map((w) => generateEmbeddingVector(w, $configDModel));

  $: positionVectors = $dataMode === 'lecture'
    ? (lecturePEStage?.pe ?? [])
    : sinusoidalPETable(seqLen, currentDModel);

  $: summedVectors = $dataMode === 'lecture'
    ? (lecturePEStage?.tokenVectors ?? [])
    : embeddingVectors.map((e, i) => addVectors(e, positionVectors[i]));

  // A couple of "beyond this sentence" position rows for the table step —
  // real, computable values, illustrating that sinusoidal PE doesn't stop
  // existing just because the current sentence is shorter (see Deep Dive:
  // "generalizes past training length").
  $: extraPositionRows = [seqLen, seqLen + 1].map((pos) => ({
    pos,
    vector: sinusoidalPositionEncoding(pos, currentDModel),
  }));

  function wordVector(word) {
    const idx = activeSentence.indexOf(word);
    return idx >= 0 ? embeddingVectors[idx] : null;
  }
  $: swappedSentence = seqLen >= 2 ? [activeSentence[1], activeSentence[0], ...activeSentence.slice(2)] : activeSentence;
  $: swappedVectors = swappedSentence.map((w) => wordVector(w));

  let swapped = false;

  // --- Camera Director: authored shots per sub-step (reusing the exact
  // same system EmbeddingViz.svelte uses — no new camera code here). ---
  const WIDE = { x: 0, y: 0, scale: 1 };
  const SHOTS = [
    WIDE,                          // PROBLEM — wide; need both rows (original + swapped) visible at once
    { x: 0, y: 20, scale: 1.04 },  // TABLE — push toward the table
    { x: 0, y: -14, scale: 1.07 }, // COMBINE — pan into the addition
    { x: 0, y: -6, scale: 1.02 },  // OUTPUT — settle toward the final tensor
    WIDE,                          // SUMMARY — camera gets out of the way
    WIDE,                          // QUIZ — camera gets out of the way
  ];
  let cameraBuildupTimer;
  $: {
    $replayTick;
    clearTimeout(cameraBuildupTimer);
    if ($subStepIndex === STEP.COMBINE) {
      focusShot(WIDE); // intro: both operands visible
      cameraBuildupTimer = setTimeout(() => focusShot(SHOTS[STEP.COMBINE]), 500); // buildup: push into the addition
    } else {
      focusShot(SHOTS[$subStepIndex] ?? WIDE);
    }
  }

  // --- Shared-element morph: the position row you saw in the table IS the
  // one being added (same crossfade pattern as EmbeddingViz.svelte). ---
  const [send, receive] = crossfade({
    duration: (dist) => (prefersReducedMotion ? 0 : Math.sqrt(dist * 250)),
    easing: cubicOut,
    fallback(node) {
      return { duration: motionMs(260), css: (t) => `opacity: ${t}; transform: scale(${0.9 + t * 0.1})` };
    },
  });

  $: stageKey = [$dataMode, activeSentence.join('|'), $subStepIndex, $replayTick].join('::');
  $: if (stageKey) swapped = false; // fresh swap state each time the problem beat is (re)entered

  function selectToken(i) {
    selectedTokenPos.set($selectedTokenPos === i ? null : i);
  }
</script>

<CameraStage>
  <div class="pe-scene">
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

    {#key stageKey}
      <div class="stage">

        {#if $subStepIndex === STEP.PROBLEM && seqLen > 0}
          <p class="prompt">Can the model tell which word came first?</p>

          <div class="row-block" in:fade={{ duration: motionMs(250) }}>
            <span class="row-caption">Original order</span>
            <div class="token-vec-row">
              {#each activeSentence as word, i (word + '-' + i)}
                <div class="token-vec" in:fly={{ y: -8, duration: motionMs(300), delay: motionMs(i * 80) }}>
                  <span class="tv-word">{word}</span>
                  <VectorHeatmap vector={embeddingVectors[i] ?? []} size="small" />
                </div>
              {/each}
            </div>
          </div>

          {#if !swapped && seqLen >= 2}
            <button class="swap-btn" on:click={() => (swapped = true)}>
              Swap the first two words
            </button>
          {:else if seqLen < 2}
            <p class="reveal-line">Add at least two words above to try the swap.</p>
          {:else}
            <div class="row-block" in:fade={{ duration: motionMs(300) }}>
              <span class="row-caption">Swapped order — <strong>same vectors, just reordered</strong></span>
              <div class="token-vec-row">
                {#each swappedSentence as word, i (word + '-swap-' + i)}
                  <div class="token-vec" in:fly={{ y: -8, duration: motionMs(300), delay: motionMs(i * 80) }}>
                    <span class="tv-word">{word}</span>
                    <VectorHeatmap vector={swappedVectors[i] ?? []} size="small" />
                  </div>
                {/each}
              </div>
            </div>
            <p class="reveal-line" in:fade={{ duration: motionMs(400), delay: motionMs(300) }}>
              Every vector is identical to before — just attached to a different word now. The embeddings themselves carry no memory of position.
            </p>
          {/if}
        {/if}

        {#if $subStepIndex === STEP.TABLE && seqLen > 0}
          <div class="pe-table" in:fly={{ y: 16, duration: motionMs(400), delay: motionMs(150) }}>
            <div class="table-title">
              Positional Encoding Table <span class="shape-note">[position → d_model]</span>
            </div>
            {#each activeSentence as _, pos (pos)}
              <div
                class="table-row highlighted"
                role="button"
                tabindex="0"
                aria-label={`Position ${pos} encoding — synced with the equation and code below`}
                class:synced={$highlightedTermId === 'eq-pe-formula'}
                on:mouseenter={() => setHighlight('eq-pe-formula')}
                on:mouseleave={clearHighlight}
                on:focus={() => setHighlight('eq-pe-formula')}
                on:blur={clearHighlight}
              >
                <span class="row-label">pos {pos}</span>
                <div class="glow-wrap">
                  <span class="glow-halo" style="animation-delay: {motionMs(300 + pos * 140)}ms"></span>
                  <div class="row-heatmap" in:receive={{ key: 'pe-pos-' + pos }} out:send={{ key: 'pe-pos-' + pos }}>
                    <VectorHeatmap vector={positionVectors[pos] ?? []} size="small" />
                  </div>
                </div>
              </div>
            {/each}
            {#each extraPositionRows as row (row.pos)}
              <div class="table-row">
                <span class="row-label">pos {row.pos}</span>
                <div class="row-heatmap">
                  <VectorHeatmap vector={row.vector} size="small" />
                </div>
              </div>
            {/each}
            <p class="table-footnote">…positions {seqLen} and {seqLen + 1} still compute fine — this table isn't capped at our sentence length.</p>
          </div>
        {/if}

        {#if $subStepIndex === STEP.COMBINE && seqLen > 0}
          <div class="combine-column">
            {#each activeSentence as word, i (word + '-combine-' + i)}
              <div class="combine-row" in:fade={{ duration: motionMs(250), delay: motionMs(i * 160) }}>
                <span class="combine-label">{word}</span>
                <VectorHeatmap vector={embeddingVectors[i] ?? []} size="small" />
                <span class="op-glyph">+</span>
                <div in:receive={{ key: 'pe-pos-' + i }} out:send={{ key: 'pe-pos-' + i }}>
                  <VectorHeatmap vector={positionVectors[i] ?? []} size="small" />
                </div>
                <span class="op-glyph">=</span>
                <div
                  class="glow-wrap"
                  role="button"
                  tabindex="0"
                  aria-label={`${word}'s position-aware vector — synced with the equation and code below`}
                  class:synced={$highlightedTermId === 'eq-pe-add'}
                  on:mouseenter={() => setHighlight('eq-pe-add')}
                  on:mouseleave={clearHighlight}
                  on:focus={() => setHighlight('eq-pe-add')}
                  on:blur={clearHighlight}
                >
                  <span class="glow-halo" style="animation-delay: {motionMs(i * 160 + 450)}ms"></span>
                  <VectorHeatmap vector={summedVectors[i] ?? []} size="small" />
                </div>
              </div>
            {/each}
          </div>
          <p class="combine-note" in:fade={{ duration: motionMs(400), delay: motionMs(seqLen * 160 + 500) }}>
            Sixteen numbers plus sixteen numbers, added elementwise — sixteen numbers out. No new dimensions.
          </p>
        {/if}

        {#if $subStepIndex === STEP.OUTPUT && seqLen > 0}
          <div class="output-row">
            {#each activeSentence as word, i (word + '-out-' + i)}
              <button
                class="token-vec output"
                class:selected={$selectedTokenPos === i}
                on:click={() => selectToken(i)}
                in:fly={{ y: -10, duration: motionMs(300), delay: motionMs(i * 100) }}
              >
                <span class="tv-word">{word}</span>
                <VectorHeatmap vector={summedVectors[i] ?? []} size="small" />
              </button>
            {/each}
          </div>
          <p class="output-note" in:fade={{ duration: motionMs(400), delay: motionMs(seqLen * 100 + 400) }}>
            Same {seqLen} tokens, same {currentDModel} dimensions each — shape unchanged.
          </p>
        {/if}

        {#if $subStepIndex === STEP.SUMMARY}
          <div class="beat" in:fade={{ duration: motionMs(300) }}>
            <BeforeAfterSummary
              before={{ label: 'Embeddings', shape: [seqLen, currentDModel] }}
              after={{ label: 'Position-Aware Embeddings', shape: [seqLen, currentDModel] }}
              whatChanged={copy.beforeAfter.whatChanged}
              structured={copy.beforeAfter.structured}
            />
          </div>
        {/if}

        {#if $subStepIndex === STEP.QUIZ}
          <div class="beat" in:fade={{ duration: motionMs(300) }}>
            <QuickCheck
              question={copy.quickCheck.question}
              choices={copy.quickCheck.choices}
              correctIndex={copy.quickCheck.correctIndex}
              explanation={copy.quickCheck.explanation}
              transition={copy.quickCheck.transition}
              distractorNotes={copy.quickCheck.distractorNotes}
            />
          </div>
        {/if}

        <div class="shape-trace-wrap">
          <TensorShapeTrace
            steps={[
              { label: 'Embedding', shape: [seqLen, currentDModel], highlightId: 'eq-pe-formula' },
              { label: 'Position-Aware', shape: [seqLen, currentDModel], highlightId: 'eq-pe-add' },
            ]}
            activeIndex={$subStepIndex >= STEP.COMBINE ? 1 : 0}
          />
        </div>
      </div>
    {/key}
  </div>
</CameraStage>

<style>
  .pe-scene {
    width: 100%; height: 100%;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 1rem; padding: 1rem;
  }

  .word-picker {
    display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; justify-content: center;
    font-family: 'Inter', sans-serif; font-size: 0.75rem; max-width: 44rem;
  }
  .hint { opacity: 0.6; margin-right: 0.25rem; }
  .chip {
    background: rgba(122, 162, 255, 0.12); border: 1px solid var(--accent-dim, #4a5578);
    color: var(--text-primary, #e6e8ef); border-radius: 999px; padding: 0.25rem 0.65rem;
    cursor: pointer; font-size: 0.75rem;
  }
  .chip:disabled { opacity: 0.3; cursor: not-allowed; }
  .chip.reset { background: transparent; border-color: var(--border-subtle, #232838); opacity: 0.7; }

  .stage { display: flex; flex-direction: column; align-items: center; gap: 0.75rem; width: 100%; }
  .beat { display: flex; justify-content: center; width: 100%; padding: 1rem 0; }

  .prompt {
    font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-style: italic;
    opacity: 0.9; text-align: center; margin: 0;
  }
  .row-block { display: flex; flex-direction: column; align-items: center; gap: 0.4rem; }
  .row-caption { font-family: 'Inter', sans-serif; font-size: 0.72rem; opacity: 0.6; }
  .token-vec-row { display: flex; gap: 0.75rem; flex-wrap: wrap; justify-content: center; }
  .token-vec {
    display: flex; flex-direction: column; align-items: center; gap: 0.25rem;
    background: var(--bg-elevated, #141722); border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.6rem; padding: 0.45rem 0.7rem; color: inherit; cursor: default;
  }
  .token-vec.output { cursor: pointer; transition: border-color 0.2s, transform 0.2s; }
  .token-vec.output:hover { border-color: var(--accent-dim, #4a5578); transform: translateY(-2px); }
  .token-vec.output.selected { border-color: var(--accent, #7aa2ff); box-shadow: 0 0 0 1px var(--accent, #7aa2ff); }
  .tv-word { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.85rem; }

  .swap-btn {
    background: rgba(255, 184, 107, 0.12); border: 1px solid var(--accent-2, #ffb86b);
    color: var(--accent-2, #ffb86b); border-radius: 999px; padding: 0.4rem 1rem;
    font-family: 'Inter', sans-serif; font-size: 0.8rem; cursor: pointer;
  }
  .reveal-line {
    max-width: 32rem; text-align: center; font-family: 'Inter', sans-serif;
    font-size: 0.85rem; opacity: 0.85; line-height: 1.5;
  }

  .pe-table {
    background: var(--bg-elevated, #141722); border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.75rem; padding: 0.75rem 1rem; width: min(30rem, 90vw);
  }
  .table-title { font-family: 'Inter', sans-serif; font-size: 0.72rem; opacity: 0.7; margin-bottom: 0.5rem; }
  .shape-note { opacity: 0.5; margin-left: 0.3rem; }
  .table-row {
    display: flex; align-items: center; gap: 0.6rem; padding: 0.2rem 0.3rem; border-radius: 0.35rem;
    opacity: 0.4; transition: opacity 0.2s, background 0.2s;
  }
  .table-row.highlighted { opacity: 1; background: rgba(122, 162, 255, 0.08); cursor: default; }
  .table-row.highlighted.synced { background: rgba(122, 162, 255, 0.16); }
  .row-label { width: 3.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; text-align: right; flex-shrink: 0; }
  .row-heatmap { display: flex; }
  .table-footnote { margin: 0.4rem 0 0; font-size: 0.68rem; opacity: 0.5; font-style: italic; }

  .combine-column { display: flex; flex-direction: column; gap: 0.5rem; align-items: center; }
  .combine-row { display: flex; align-items: center; gap: 0.5rem; }
  .combine-label { width: 3.5rem; font-family: 'Inter', sans-serif; font-size: 0.75rem; text-align: right; flex-shrink: 0; }
  .op-glyph { font-family: 'JetBrains Mono', monospace; opacity: 0.5; font-size: 0.9rem; }
  .combine-note { max-width: 30rem; text-align: center; font-family: 'Inter', sans-serif; font-size: 0.8rem; opacity: 0.75; }

  .output-row { display: flex; gap: 0.75rem; flex-wrap: wrap; justify-content: center; }
  .output-note { font-family: 'Inter', sans-serif; font-size: 0.8rem; opacity: 0.75; }

  .glow-wrap { position: relative; display: inline-flex; border-radius: 4px; }
  .glow-wrap[role='button'] { cursor: default; }
  .glow-wrap.synced { outline: 1px solid var(--accent, #7aa2ff); }
  .glow-halo {
    position: absolute; inset: -8px; border-radius: 8px;
    background: radial-gradient(circle, rgba(122, 162, 255, 0.55), transparent 70%);
    opacity: 0; transform: scale(0.85);
    animation: halo-settle 900ms ease-out 1 both;
    pointer-events: none; will-change: opacity, transform;
  }
  @keyframes halo-settle {
    0% { opacity: 0; transform: scale(0.85); }
    55% { opacity: 0.9; transform: scale(1.1); }
    100% { opacity: 0.22; transform: scale(1); }
  }

  .shape-trace-wrap { margin-top: 0.25rem; }

  .table-row:focus-visible, .glow-wrap:focus-visible {
    outline: 2px solid var(--accent, #7aa2ff);
    outline-offset: 2px;
  }

  @media (prefers-reduced-motion: reduce) {
    .glow-halo { animation: none; opacity: 0.25; }
  }
</style>
