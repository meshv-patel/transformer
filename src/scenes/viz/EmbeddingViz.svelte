<script>
  import { fly, fade } from 'svelte/transition';
  import { crossfade } from 'svelte/transition';
  import { cubicOut, quintOut } from 'svelte/easing';
  import { subStepIndex, replayTick, dataMode } from '../../core/stores/sceneStore.js';
  import { dModel as configDModel } from '../../core/stores/configStore.js';
  import { forwardPassData } from '../../core/data-loader.js';
  import { selectedTokenPos } from '../../core/stores/tokenJourneyStore.js';
  import { highlightedTermId, setHighlight, clearHighlight } from '../../core/stores/highlightStore.js';
  import { generateEmbeddingVector, pseudoTokenId } from '../../core/embedding-utils.js';
  import { cosineSimilarity, nearestNeighbor } from '../../core/vector-math.js';
  import { VOCAB_WORDS } from '../../data/vocab.js';
  import { SCENE_COPY } from '../../data/scene-copy.js';
  import { focusShot } from '../../core/camera/cameraStore.js';
  import { motionMs, prefersReducedMotion } from '../../core/motion.js';
  import TensorShapeTrace from '../../components/TensorShapeTrace.svelte';
  import CameraStage from '../../components/CameraStage.svelte';
  import BeforeAfterSummary from '../../components/BeforeAfterSummary.svelte';
  import QuickCheck from '../../components/QuickCheck.svelte';
  import VectorHeatmap from '../../components/VectorHeatmap.svelte';

  const copy = SCENE_COPY.embedding;

  // Named sub-step constants — replaces magic-number comparisons
  // ($subStepIndex === 2, etc.) with readable names. See
  // docs/reference-implementation.md "External review response" §1.
  const STEP = { LOOKUP: 0, MATERIALIZE: 1, SUMMARY: 2, QUIZ: 3 };

  const DEFAULT_INTERACTIVE_SENTENCE = ['cat', 'chased', 'dog'];
  const MAX_INTERACTIVE_TOKENS = 6;

  let interactiveSentence = [...DEFAULT_INTERACTIVE_SENTENCE];
  let compareMode = false;
  let compareSelection = []; // up to 2 token indices

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

  // --- Data selection: Lecture (real precomputed) vs Interactive (live, seeded) ---
  $: lectureMeta = $forwardPassData?.meta ?? null;
  $: lectureEmbeddingStage = $forwardPassData?.stages?.find((s) => s.id === 'embedding') ?? null;
  $: lectureEmbeddingSample = $forwardPassData?.embeddingSample ?? [];

  $: activeSentence = $dataMode === 'lecture' ? (lectureMeta?.sentence ?? []) : interactiveSentence;
  $: currentDModel = $dataMode === 'lecture' ? (lectureMeta?.dModel ?? 16) : $configDModel;
  $: activeTokenIds = $dataMode === 'lecture' ? (lectureMeta?.tokenIds ?? []) : interactiveSentence.map(pseudoTokenId);
  $: activeVectors = $dataMode === 'lecture'
    ? (lectureEmbeddingStage?.tokenVectors ?? [])
    : interactiveSentence.map((w) => generateEmbeddingVector(w, $configDModel));

  $: tableRows = $dataMode === 'lecture' ? lectureEmbeddingSample : buildInteractiveTable(interactiveSentence, $configDModel);
  $: usedRows = tableRows.filter((r) => r.used);

  function buildInteractiveTable(sentenceWords, dModelVal) {
    const decoys = VOCAB_WORDS.filter((w) => !sentenceWords.includes(w)).slice(0, 4);
    const words = [...new Set([...sentenceWords, ...decoys])];
    return words.map((w) => ({
      word: w,
      tokenId: pseudoTokenId(w),
      vector: generateEmbeddingVector(w, dModelVal),
      used: sentenceWords.includes(w),
    }));
  }

  // --- Camera Director: authored shots per sub-step (see core/camera) ---
  // Sub-steps 2/3 (summary, quiz) deliberately get NO camera motion — once
  // the lesson has landed, an "invisible" camera means getting out of the
  // way entirely, not finding a new place to point. Sub-step 0 gets a
  // two-stage intro (wide) -> buildup (push toward the table) rather than
  // one flat shot, so the motion itself narrates "here's the whole table,
  // now here's where we're looking."
  const WIDE = { x: 0, y: 0, scale: 1 };
  const SHOTS = [
    { x: 0, y: 22, scale: 1.04 },  // 0 lookup-table: gentle push toward the table (after intro beat below)
    { x: 0, y: -14, scale: 1.07 }, // 1 vector-materialize: pan up into the vectors
    WIDE,                          // 2 before-after: camera gets out of the way
    WIDE,                          // 3 quick-check: camera gets out of the way
  ];
  let cameraBuildupTimer;
  $: {
    $replayTick; // dependency: re-run the camera move on replay too
    clearTimeout(cameraBuildupTimer);
    if ($subStepIndex === STEP.LOOKUP) {
      focusShot(WIDE); // intro: show the whole table exists
      cameraBuildupTimer = setTimeout(() => focusShot(SHOTS[0]), 500); // buildup: push toward it
    } else {
      focusShot(SHOTS[$subStepIndex] ?? WIDE);
    }
  }

  // --- Shared-element morph: a table row IS the vector it becomes ---
  const [send, receive] = crossfade({
    duration: (dist) => (prefersReducedMotion ? 0 : Math.sqrt(dist * 250)),
    easing: cubicOut,
    fallback(node) {
      return { duration: motionMs(260), css: (t) => `opacity: ${t}; transform: scale(${0.9 + t * 0.1})` };
    },
  });

  // Token row persists across sub-step changes (only remounts on real change)
  // so the camera move reads as "the same scene, different focus" rather
  // than a hard cut.
  $: sentenceKey = [$dataMode, activeSentence.join('|'), $replayTick].join('::');
  $: stageKey = [$dataMode, activeSentence.join('|'), $subStepIndex, $replayTick].join('::');

  function selectToken(i) {
    selectedTokenPos.set($selectedTokenPos === i ? null : i);
  }

  function toggleCompareMode() {
    compareMode = !compareMode;
    compareSelection = [];
  }

  function toggleCompareSelect(i) {
    if (compareSelection.includes(i)) {
      compareSelection = compareSelection.filter((x) => x !== i);
    } else if (compareSelection.length < 2) {
      compareSelection = [...compareSelection, i];
    } else {
      compareSelection = [compareSelection[1], i];
    }
  }

  $: compareResult = compareSelection.length === 2
    ? {
        a: activeSentence[compareSelection[0]],
        b: activeSentence[compareSelection[1]],
        similarity: cosineSimilarity(activeVectors[compareSelection[0]], activeVectors[compareSelection[1]]),
      }
    : null;

  $: nearestForSelected = compareSelection.length === 1
    ? nearestNeighbor(
        activeVectors[compareSelection[0]],
        tableRows.map((r) => ({ word: r.word, vector: r.vector })),
        activeSentence[compareSelection[0]]
      )
    : null;

  function similarityLabel(sim) {
    if (sim > 0.6) return 'quite similar';
    if (sim > 0.2) return 'somewhat related';
    if (sim > -0.2) return 'roughly unrelated';
    return 'pointing in different directions';
  }
</script>

<CameraStage>
  <div class="embedding-scene">
    {#if $dataMode === 'interactive'}
      <div class="word-picker" transition:fade={{ duration: motionMs(150) }}>
        <span class="hint">Build a sentence:</span>
        {#each VOCAB_WORDS as w}
          <button class="chip" on:click={() => addWord(w)} disabled={interactiveSentence.length >= MAX_INTERACTIVE_TOKENS}>
            + {w}
          </button>
        {/each}
        <button class="chip reset" on:click={resetSentence}>Reset</button>
        <span class="disclosure">Illustrative vectors — this toy model isn't trained, so values show the mechanism, not learned meaning.</span>
      </div>
    {/if}

    {#key sentenceKey}
      <div class="token-row" class:dimmed={$subStepIndex >= 2} in:fade={{ duration: motionMs(250) }}>
        {#each activeSentence as word, i (word + '-' + i)}
          <button
            class="token-card"
            class:selected={$selectedTokenPos === i}
            on:click={() => selectToken(i)}
            in:fly={{ y: -10, duration: motionMs(350), delay: motionMs(i * 90), easing: quintOut }}
          >
            <span class="token-word">{word}</span>
            <span class="token-id">id {activeTokenIds[i]}</span>
            {#if $dataMode === 'interactive'}
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
          <p class="empty-hint">Add a word above to see it embedded.</p>
        {/if}
      </div>
    {/key}

    {#key stageKey}
      <div class="stage">
        {#if $subStepIndex === STEP.LOOKUP && activeSentence.length > 0}
          <div class="lookup-column" in:fade={{ duration: motionMs(200) }}>
            {#each activeSentence as _, i (i)}
              <div
                class="lookup-arrow"
                role="button"
                tabindex="0"
                aria-label="Lookup step — synced with the equation and code below"
                class:active={$highlightedTermId === 'eq-lookup'}
                on:mouseenter={() => setHighlight('eq-lookup')}
                on:mouseleave={clearHighlight}
                on:focus={() => setHighlight('eq-lookup')}
                on:blur={clearHighlight}
                in:fly={{ y: -8, duration: motionMs(300), delay: motionMs(420 + i * 90) }}
              >
                <span class="arrow-glyph">↓</span>
              </div>
            {/each}
          </div>

          <div class="embedding-table" in:fly={{ y: 16, duration: motionMs(400), delay: motionMs(650) }}>
            <div class="table-title">
              Embedding Table <span class="mono">E</span>
              <span class="shape-note">[vocab_size × d_model]</span>
            </div>
            {#each tableRows as row (row.word)}
              <div class="table-row" class:highlighted={row.used}>
                <span class="row-label">{row.word}</span>
                {#if row.used}
                  <div class="glow-wrap">
                    <span class="glow-halo" style="animation-delay: {950 + usedRows.indexOf(row) * 140}ms"></span>
                    <div
                      class="row-heatmap"
                      in:receive={{ key: 'tok-' + usedRows.indexOf(row) }}
                      out:send={{ key: 'tok-' + usedRows.indexOf(row) }}
                    >
                      <VectorHeatmap vector={row.vector.slice(0, currentDModel)} size="small" />
                    </div>
                  </div>
                {:else}
                  <div class="row-heatmap" out:fade={{ duration: motionMs(200) }}>
                    <VectorHeatmap vector={row.vector.slice(0, currentDModel)} size="small" />
                  </div>
                {/if}
              </div>
            {/each}
            <p class="table-footnote">…and the rest of the vocabulary, not shown.</p>
          </div>
        {/if}

        {#if $subStepIndex === STEP.MATERIALIZE && activeSentence.length > 0}
          <div class="compare-bar">
            <button class="compare-toggle" class:active={compareMode} on:click={toggleCompareMode}>
              {compareMode ? 'Comparing — click 2 vectors' : 'Compare two tokens'}
            </button>
          </div>

          <div class="vector-row">
            {#each activeVectors as vec, i (i)}
              <div class="glow-wrap">
                <span class="glow-halo" style="animation-delay: {i * 130 + currentDModel * 10 + 150}ms"></span>
                <div
                  class="vector-strip"
                  role="button"
                  tabindex={compareMode ? 0 : -1}
                  aria-pressed={compareSelection.includes(i)}
                  aria-label={`${activeSentence[i]} embedding vector${compareMode ? ' — press to compare' : ''}`}
                  class:active={$highlightedTermId === 'eq-lookup'}
                  class:compare-selected={compareSelection.includes(i)}
                  on:mouseenter={() => setHighlight('eq-lookup')}
                  on:mouseleave={clearHighlight}
                  on:focus={() => setHighlight('eq-lookup')}
                  on:blur={clearHighlight}
                  on:click={() => compareMode && toggleCompareSelect(i)}
                  on:keydown={(e) => { if (compareMode && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); toggleCompareSelect(i); } }}
                  in:receive={{ key: 'tok-' + i }}
                  out:send={{ key: 'tok-' + i }}
                >
                  <VectorHeatmap
                    vector={vec}
                    getTooltip={(v, ci) => `${activeSentence[i]}[${ci}] = ${v.toFixed(3)}`}
                    staggerDelay={(ci) => motionMs(i * 130 + ci * 10)}
                  />
                </div>
              </div>
            {/each}
          </div>

          {#if compareResult}
            <div class="compare-result" in:fade={{ duration: motionMs(200) }}>
              <strong>{compareResult.a}</strong> ↔ <strong>{compareResult.b}</strong>:
              cosine similarity <span class="mono">{compareResult.similarity.toFixed(3)}</span>
              — {similarityLabel(compareResult.similarity)}
              <span class="tie-back">(this is the "nearby vectors" idea from the Why panel below, made concrete)</span>
            </div>
          {:else if nearestForSelected}
            <div class="compare-result" in:fade={{ duration: motionMs(200) }}>
              Nearest neighbor to <strong>{activeSentence[compareSelection[0]]}</strong> (among the visible sample):
              <strong>{nearestForSelected.word}</strong>
              <span class="mono">({nearestForSelected.similarity.toFixed(3)})</span>
            </div>
          {/if}

          <div class="legend" in:fade={{ duration: motionMs(300), delay: motionMs(400) }}>
            <span class="legend-neg">−</span>
            <span class="legend-bar"></span>
            <span class="legend-pos">+</span>
            <span class="legend-label">embedding value at each dimension</span>
          </div>

          <p class="curiosity-hook" in:fade={{ duration: motionMs(500), delay: motionMs(900) }}>
            Every token has meaning now — but does the model know which one came first?
          </p>
        {/if}

        {#if $subStepIndex === STEP.SUMMARY}
          <div class="beat" in:fade={{ duration: motionMs(300) }}>
            <BeforeAfterSummary
              before={{ label: 'Token IDs', shape: [activeSentence.length || 0] }}
              after={{ label: 'Embedding Vectors', shape: [activeSentence.length || 0, currentDModel] }}
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
              { label: 'Token IDs', shape: [activeSentence.length || 0], highlightId: 'eq-lookup' },
              { label: 'Embedding', shape: [activeSentence.length || 0, currentDModel], highlightId: 'eq-lookup' },
            ]}
            activeIndex={$subStepIndex === STEP.LOOKUP ? 0 : 1}
          />
        </div>
      </div>
    {/key}
  </div>
</CameraStage>

<style>
  .embedding-scene {
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
  .disclosure { width: 100%; text-align: center; opacity: 0.5; font-size: 0.7rem; margin-top: 0.15rem; }

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
  .token-id { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; opacity: 0.6; }
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

  .lookup-column { display: flex; gap: 0.6rem; justify-content: center; }
  .lookup-arrow { width: 2.6rem; display: flex; justify-content: center; opacity: 0.5; transition: opacity 0.2s, transform 0.2s; }
  .lookup-arrow.active { opacity: 1; transform: scale(1.15); color: var(--accent, #7aa2ff); }
  .arrow-glyph { animation: pulse-down 1.4s ease-in-out infinite; display: inline-block; }
  @keyframes pulse-down {
    0%, 100% { transform: translateY(0); opacity: 0.6; }
    50% { transform: translateY(4px); opacity: 1; }
  }

  .embedding-table {
    background: var(--bg-elevated, #141722);
    border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.75rem;
    padding: 0.75rem 1rem;
    width: min(34rem, 90vw);
  }
  .table-title { font-family: 'Inter', sans-serif; font-size: 0.72rem; opacity: 0.7; margin-bottom: 0.5rem; }
  .mono { font-family: 'JetBrains Mono', monospace; }
  .shape-note { opacity: 0.5; margin-left: 0.3rem; }
  .table-row {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.2rem 0.3rem; border-radius: 0.35rem;
    opacity: 0.4; transition: opacity 0.2s, background 0.2s;
  }
  .table-row.highlighted { opacity: 1; background: rgba(122, 162, 255, 0.08); }
  .row-label { width: 4.5rem; font-family: 'Inter', sans-serif; font-size: 0.72rem; text-align: right; flex-shrink: 0; }
  .row-heatmap { display: flex; gap: 1px; }
  .table-footnote { margin: 0.4rem 0 0; font-size: 0.68rem; opacity: 0.5; font-style: italic; }

  .glow-wrap { position: relative; display: inline-flex; }
  .glow-halo {
    position: absolute;
    inset: -8px;
    border-radius: 8px;
    background: radial-gradient(circle, rgba(122, 162, 255, 0.55), transparent 70%);
    opacity: 0;
    transform: scale(0.85);
    animation: halo-settle 900ms ease-out 1 both;
    pointer-events: none;
    will-change: opacity, transform;
  }
  @keyframes halo-settle {
    0% { opacity: 0; transform: scale(0.85); }
    55% { opacity: 0.9; transform: scale(1.1); }
    100% { opacity: 0.22; transform: scale(1); }
  }

  .compare-bar { display: flex; justify-content: center; }
  .compare-toggle {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-subtle, #232838);
    color: var(--text-secondary, #9aa1b5);
    border-radius: 999px;
    padding: 0.3rem 0.8rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    cursor: pointer;
  }
  .compare-toggle.active { border-color: var(--accent-2, #ffb86b); color: var(--accent-2, #ffb86b); }

  .vector-row { display: flex; gap: 1.2rem; flex-wrap: wrap; justify-content: center; }
  .vector-strip {
    display: flex; gap: 1px; padding: 0.4rem; border-radius: 0.5rem;
    border: 1px solid transparent; transition: border-color 0.2s; cursor: default;
  }
  .vector-strip.active { border-color: var(--accent, #7aa2ff); }
  .vector-strip.compare-selected { border-color: var(--accent-2, #ffb86b); box-shadow: 0 0 10px rgba(255, 184, 107, 0.3); }

  .compare-result {
    font-family: 'Inter', sans-serif; font-size: 0.8rem; opacity: 0.9;
    background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border-subtle, #232838);
    border-radius: 0.5rem; padding: 0.5rem 0.85rem;
  }
  .tie-back { display: block; margin-top: 0.25rem; opacity: 0.6; font-size: 0.72rem; font-style: italic; }

  .curiosity-hook {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-style: italic;
    opacity: 0.85;
    text-align: center;
    max-width: 30rem;
    margin: 0.25rem 0 0;
  }

  .legend { display: flex; align-items: center; gap: 0.4rem; font-family: 'Inter', sans-serif; font-size: 0.7rem; opacity: 0.65; }
  .legend-bar {
    width: 5rem; height: 0.5rem; border-radius: 999px;
    background: linear-gradient(90deg, rgba(122,162,255,0.9), rgba(20,23,34,0.9), rgba(255,184,107,0.9));
  }

  .shape-trace-wrap { margin-top: 0.25rem; }

  .lookup-arrow:focus-visible,
  .vector-strip:focus-visible {
    outline: 2px solid var(--accent, #7aa2ff);
    outline-offset: 2px;
  }

  @media (prefers-reduced-motion: reduce) {
    .glow-halo, .arrow-glyph { animation: none; opacity: 0.25; }
  }
</style>
