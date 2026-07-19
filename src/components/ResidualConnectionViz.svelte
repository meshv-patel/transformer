<script>
  import { fade } from 'svelte/transition';
  import { subStepIndex, replayTick, dataMode, currentScene } from '../core/stores/sceneStore.js';
  import { dModel as configDModel, numHeads as configNumHeads } from '../core/stores/configStore.js';
  import { forwardPassData } from '../core/data-loader.js';
  import { highlightedTermId, setHighlight, clearHighlight } from '../core/stores/highlightStore.js';
  import { computeAttentionPipeline, addMatrices } from '../core/tensor-ops.js';
  import { getSceneCopy } from '../data/scene-copy.js';
  import { focusShot } from '../core/camera/cameraStore.js';
  import { motionMs } from '../core/motion.js';
  import { VOCAB_WORDS } from '../data/vocab.js';
  import TensorShapeTrace from './TensorShapeTrace.svelte';
  import CameraStage from './CameraStage.svelte';
  import VectorHeatmap from './VectorHeatmap.svelte';
  import BeforeAfterSummary from './BeforeAfterSummary.svelte';
  import QuickCheck from './QuickCheck.svelte';

  $: sceneId = $currentScene?.id;
  $: copy = getSceneCopy(sceneId);

  // Sub-step index definitions
  const STEP = { MAIN: 0, SKIP: 1, ADDITION: 2, SUMMARY: 3, QUIZ: 4 };

  // Expose configuration variables
  export let input1StageId = undefined;
  export let input2StageId = undefined;
  export let outputStageId = undefined;
  export let input1Label = '';
  export let input2Label = '';
  export let outputLabel = '';
  export let highlightId = '';

  // Reactively resolve parameters based on props or sceneId
  $: activeInput1StageId = input1StageId || (sceneId === 'residual-2' ? 'layer-norm-1' : 'positional-enc');
  $: activeInput2StageId = input2StageId || (sceneId === 'residual-2' ? 'ffn' : 'output-proj');
  $: activeOutputStageId = outputStageId || sceneId || 'residual-1';

  $: activeInput1Label = input1Label || (sceneId === 'residual-2' ? 'Input to FFN' : 'Input X_pe');
  $: activeInput2Label = input2Label || (sceneId === 'residual-2' ? 'FFN Output' : 'Attention Output A(X)');
  $: activeOutputLabel = outputLabel || (sceneId === 'residual-2' ? 'Residual Output ②' : 'Residual Output ①');
  $: activeHighlightId = highlightId || (sceneId === 'residual-2' ? 'eq-residual-2' : 'eq-residual-1');

  $: stream = $currentScene?.config?.stream ?? 'encoder';
  $: isDecoderStream = stream === 'decoder';

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

  $: activeSentence = $dataMode === 'lecture'
    ? (isDecoderStream ? ['the', 'dog', 'ran', 'slowly'] : ($forwardPassData?.meta?.sentence ?? []))
    : (isDecoderStream ? interactiveTargetSentence : interactiveSentence);

  $: currentDModel = $dataMode === 'lecture' ? ($forwardPassData?.meta?.dModel ?? 16) : $configDModel;
  $: configNumHeadsVal = $dataMode === 'lecture' ? ($forwardPassData?.meta?.numHeads ?? 4) : $configNumHeads;
  $: seqLen = activeSentence.length;

  // --- Dynamic calculations using centralized pipeline ---
  $: interactiveData = computeAttentionPipeline({
    encoderSentence: isDecoderStream ? ['cat', 'chased', 'dog'] : activeSentence,
    decoderSentence: isDecoderStream ? activeSentence : ['the', 'dog', 'ran'],
    dModel: currentDModel,
    numHeads: configNumHeadsVal,
    lectureWeights: $forwardPassData?.weights ?? {}
  });

  $: attentionType = $currentScene?.config?.attentionType ?? 'self';
  $: stageConfig = $currentScene?.config?.stage ?? 'residual-1';
  $: isDecoderResidual3 = isDecoderStream && (stageConfig === 'residual3' || stageConfig === 'residual-3');
  $: isCrossAttention = (attentionType === 'cross' || stageConfig === 'residual-2') && !isDecoderResidual3;

  $: activePipeline = isCrossAttention
    ? interactiveData?.decoder?.crossAttention
    : (isDecoderStream ? interactiveData?.decoder : interactiveData);

  $: precomputedInput1 = $forwardPassData?.stages?.find((s) => s.id === activeInput1StageId)?.tokenVectors ?? [];
  $: precomputedInput2 = $forwardPassData?.stages?.find((s) => s.id === activeInput2StageId)?.tokenVectors ?? [];
  $: precomputedOutput = $forwardPassData?.stages?.find((s) => s.id === activeOutputStageId)?.tokenVectors ?? [];

  // Input 1 calculation (skip connection from input/ln1/ln2)
  $: input1 = $dataMode === 'lecture' && !isDecoderStream
    ? precomputedInput1
    : (
      isDecoderResidual3
        ? (interactiveData?.decoder?.crossAttention?.ln2_outputs ?? [])
        : (isCrossAttention ? (interactiveData?.decoder?.ln1_outputs ?? []) : (activePipeline?.xPe ?? []))
    );

  // Input 2 calculation (sublayer output, e.g. ffn output / cross-attention output projection)
  $: input2 = $dataMode === 'lecture' && !isDecoderStream
    ? precomputedInput2
    : (
      isDecoderResidual3
        ? (interactiveData?.decoder?.ffn?.outputs ?? [])
        : (activePipeline?.outputProj ?? [])
    );

  // Output calculation (residual sum)
  $: output = $dataMode === 'lecture' && !isDecoderStream
    ? precomputedOutput
    : (
      isDecoderResidual3
        ? (interactiveData?.decoder?.ffn?.residual3 ?? addMatrices(input1, input2))
        : (isCrossAttention ? (activePipeline?.residual2 ?? addMatrices(input1, input2)) : (activePipeline?.residual1 ?? addMatrices(input1, input2)))
    );

  // Highlighting synchronization
  let activeHoverIdx = null;

  function handleHover(i) {
    activeHoverIdx = i;
    setHighlight(activeHighlightId);
  }

  function handleUnhover() {
    activeHoverIdx = null;
    clearHighlight();
  }

  // --- Camera Choreography ---
  const WIDE = { x: 0, y: 0, scale: 1 };
  $: {
    $replayTick;
    focusShot(WIDE);
  }

  $: sentenceKey = [$dataMode, activeSentence.join('|'), $replayTick].join('::');
  $: stageKey = [$dataMode, activeSentence.join('|'), $subStepIndex, $replayTick].join('::');

  // Trace steps setup
  $: traceSteps = [
    { label: activeInput1Label, shape: [seqLen, currentDModel], highlightId: activeHighlightId },
    { label: activeInput2Label, shape: [seqLen, currentDModel], highlightId: activeHighlightId },
    { label: activeOutputLabel, shape: [seqLen, currentDModel], highlightId: activeHighlightId }
  ];
</script>

<CameraStage>
  <div class="residual-scene-wrap">
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
      <div class="main-content">
        {#if $subStepIndex < STEP.SUMMARY}
          <!-- Process flow view -->
          <div class="flow-layout" in:fade={{ duration: motionMs(300) }}>
            
            <!-- SVG skip path diagram -->
            <div class="flow-diagram-wrap">
              <svg width="100%" height="80" viewBox="0 0 800 80" class="skip-svg">
                <!-- Main flow path from attention to sum -->
                <path d="M 400 70 L 400 45" fill="none" stroke="#555" stroke-width="2" marker-end="url(#arrow)" />
                <!-- Skip path routing from input to sum -->
                <path
                  d="M 120 40 C 200 -10, 300 -10, 400 20"
                  fill="none"
                  stroke={$subStepIndex === STEP.SKIP ? 'var(--accent, #7aa2ff)' : '#555'}
                  stroke-width={$subStepIndex === STEP.SKIP ? 3 : 2}
                  stroke-dasharray={$subStepIndex === STEP.SKIP ? '0' : '6,4'}
                  class="skip-line"
                />
              </svg>
            </div>

            <div class="matrices-container">
              <!-- Matrix 1: Input (Main / Skip source) -->
              <div
                class="matrix-card skip-path"
                class:highlighted={$subStepIndex === STEP.SKIP}
                class:dimmed={$subStepIndex === STEP.MAIN}
              >
                <h4>{activeInput1Label}</h4>
                <div class="matrix-grid">
                  {#each input1 as row, rIdx}
                    <div class="matrix-row" class:hovered={activeHoverIdx === rIdx}>
                      <span 
                        class="token-label" 
                        role="button" 
                        tabindex="0" 
                        on:click={() => removeWord(rIdx)}
                        on:keydown={(e) => e.key === 'Enter' && removeWord(rIdx)}
                      >
                        {activeSentence[rIdx] ?? ''}
                        {#if $dataMode === 'interactive'}<span class="delete-icon">×</span>{/if}
                      </span>
                      <VectorHeatmap vector={row} />
                    </div>
                  {/each}
                </div>
                <div class="matrix-shape">[{seqLen}, {currentDModel}]</div>
              </div>

              <!-- Plus operator -->
              <div class="operator" class:dimmed={$subStepIndex < STEP.ADDITION}>+</div>

              <!-- Matrix 2: Attention Output (Main path result) -->
              <div
                class="matrix-card main-path"
                class:highlighted={$subStepIndex === STEP.MAIN}
                class:dimmed={$subStepIndex === STEP.SKIP}
              >
                <h4>{activeInput2Label}</h4>
                <div class="matrix-grid">
                  {#each input2 as row, rIdx}
                    <div class="matrix-row" class:hovered={activeHoverIdx === rIdx}>
                      <span class="token-label">{activeSentence[rIdx] ?? ''}</span>
                      <VectorHeatmap vector={row} />
                    </div>
                  {/each}
                </div>
                <div class="matrix-shape">[{seqLen}, {currentDModel}]</div>
              </div>

              <!-- Equals operator -->
              <div class="operator" class:dimmed={$subStepIndex < STEP.ADDITION}>=</div>

              <!-- Matrix 3: Sum Output -->
              <div class="matrix-card output-path" class:dimmed={$subStepIndex < STEP.ADDITION}>
                <h4>{activeOutputLabel}</h4>
                <div class="matrix-grid">
                  {#each output as row, rIdx}
                    <div
                      class="matrix-row interactive-row"
                      role="presentation"
                      class:hovered={activeHoverIdx === rIdx}
                      on:mouseenter={() => handleHover(rIdx)}
                      on:mouseleave={handleUnhover}
                    >
                      <span class="token-label">{activeSentence[rIdx] ?? ''}</span>
                      <VectorHeatmap vector={row} />
                    </div>
                  {/each}
                </div>
                <div class="matrix-shape">[{seqLen}, {currentDModel}]</div>
              </div>
            </div>

          </div>
        {/if}

        {#if $subStepIndex === STEP.SUMMARY}
          <div class="beat" in:fade={{ duration: motionMs(300) }}>
            {#if copy.beforeAfter}
              <BeforeAfterSummary
                before={{ label: copy.beforeAfter.before.label, shape: [seqLen, currentDModel] }}
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

        <!-- Tensor shape trace panel -->
        <div class="shape-trace-wrap">
          <TensorShapeTrace
            steps={traceSteps}
            activeIndex={$subStepIndex === STEP.MAIN ? 1 : ($subStepIndex === STEP.SKIP ? 0 : 2)}
          />
        </div>
      </div>
    {/key}
  </div>
</CameraStage>

<style>
  .residual-scene-wrap {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    padding: 1rem;
  }

  .main-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    position: relative;
  }

  .flow-layout {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    margin-bottom: 2rem;
  }

  .flow-diagram-wrap {
    width: 100%;
    max-width: 600px;
    height: 60px;
    margin-bottom: -10px;
  }

  .skip-svg {
    width: 100%;
    height: 100%;
  }

  .skip-line {
    transition: stroke-width 0.2s, stroke 0.2s;
  }

  .matrices-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    width: 100%;
  }

  .matrix-card {
    background: rgba(25d, 25d, 25d, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
    transition: opacity 0.3s, transform 0.3s, border-color 0.3s;
    position: relative;
  }

  .matrix-card.dimmed {
    opacity: 0.25;
  }

  .matrix-card.highlighted {
    border-color: var(--accent, #7aa2ff);
    transform: scale(1.02);
    box-shadow: 0 0 12px rgba(122, 162, 255, 0.15);
  }

  h4 {
    margin: 0 0 0.6rem 0;
    font-size: 0.85rem;
    color: var(--text-muted, #8a8a8a);
    font-weight: 500;
  }

  .matrix-grid {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .matrix-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 2px 4px;
    border-radius: 4px;
    transition: background-color 0.15s;
  }

  .matrix-row.hovered {
    background: rgba(255, 255, 255, 0.06);
  }

  .matrix-row.interactive-row {
    cursor: pointer;
  }

  .token-label {
    min-width: 4rem;
    text-align: right;
    font-family: var(--font-mono, monospace);
    font-size: 0.75rem;
    color: var(--text-main, #e0e0e0);
    position: relative;
    cursor: pointer;
  }

  .token-label:hover .delete-icon {
    opacity: 1;
  }

  .delete-icon {
    position: absolute;
    right: -10px;
    top: 50%;
    transform: translateY(-50%);
    color: #ff5555;
    font-weight: bold;
    opacity: 0;
    transition: opacity 0.15s;
  }

  .matrix-shape {
    font-size: 0.7rem;
    color: var(--text-muted, #8a8a8a);
    margin-top: 0.5rem;
    font-family: var(--font-mono, monospace);
  }

  .operator {
    font-size: 2rem;
    font-weight: 300;
    color: var(--text-muted, #8a8a8a);
    transition: opacity 0.3s;
    user-select: none;
  }

  .operator.dimmed {
    opacity: 0.15;
  }

  .word-picker {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
    background: rgba(0, 0, 0, 0.2);
    padding: 0.5rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 0.5rem;
    width: 100%;
    max-width: 700px;
  }

  .word-picker .hint {
    font-size: 0.8rem;
    color: var(--text-muted, #8a8a8a);
  }

  .chip {
    background: rgba(255, 255, 255, 0.08);
    border: none;
    border-radius: 4px;
    color: var(--text-main, #e0e0e0);
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    cursor: pointer;
    transition: background 0.15s;
  }

  .chip:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.15);
  }

  .chip.reset {
    background: rgba(255, 85, 85, 0.15);
    color: #ff5555;
  }

  .chip.reset:hover {
    background: rgba(255, 85, 85, 0.25);
  }

  .beat {
    width: 100%;
    max-width: 600px;
    margin-bottom: 2rem;
  }

  .shape-trace-wrap {
    margin-top: 1rem;
    width: 100%;
    display: flex;
    justify-content: center;
  }
</style>
