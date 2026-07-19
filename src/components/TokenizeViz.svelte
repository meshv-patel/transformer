<script>
  import { forwardPassData } from '../core/data-loader.js';
  import { dModel as configDModel } from '../core/stores/configStore.js';
  import { generateEmbeddingVector } from '../core/embedding-utils.js';

  $: sentence = $forwardPassData?.meta?.sentence ?? ['cat', 'chased', 'dog'];
  $: currentDModel = $configDModel || 16;
  $: tokenData = sentence.map((word, idx) => ({
    word,
    id: (word.charCodeAt(0) * 7 + idx) % 1000,
    vector: generateEmbeddingVector(word, currentDModel)
  }));
</script>

<div class="tokenize-container">
  <div class="header-banner">
    <span class="eyebrow">SETUP STAGE 2: TOKENIZATION & EMBEDDING LOOKUP</span>
    <h3 class="title">From Raw Text to Vector Representations</h3>
  </div>

  <div class="pipeline-flow">
    <!-- Step 1: Raw Sentence -->
    <div class="flow-card">
      <span class="step-num">Step 1</span>
      <h4>Raw Sentence</h4>
      <div class="sentence-box">"{sentence.join(' ')}"</div>
    </div>

    <div class="flow-arrow">→</div>

    <!-- Step 2: Tokenization -->
    <div class="flow-card">
      <span class="step-num">Step 2</span>
      <h4>Tokenization</h4>
      <div class="chip-list">
        {#each sentence as word}
          <span class="word-chip">{word}</span>
        {/each}
      </div>
    </div>

    <div class="flow-arrow">→</div>

    <!-- Step 3: Vocabulary IDs -->
    <div class="flow-card">
      <span class="step-num">Step 3</span>
      <h4>Vocabulary IDs</h4>
      <div class="chip-list">
        {#each tokenData as item}
          <span class="id-chip">ID: {item.id}</span>
        {/each}
      </div>
    </div>

    <div class="flow-arrow">→</div>

    <!-- Step 4: Dense Vector Embeddings -->
    <div class="flow-card wide">
      <span class="step-num">Step 4</span>
      <h4>Dense Embeddings [d_model = {currentDModel}]</h4>
      <div class="vec-table">
        {#each tokenData as item}
          <div class="vec-row">
            <span class="row-label">{item.word}:</span>
            <div class="vec-cells">
              {#each item.vector.slice(0, 6) as val}
                <span class="v-cell" style="background: rgba(56, 189, 248, {Math.abs(val)});">
                  {val.toFixed(2)}
                </span>
              {/each}
              {#if currentDModel > 6}
                <span class="more-cells">... +{currentDModel - 6} dims</span>
              {/if}
            </div>
          </div>
        {/each}
      </div>
    </div>
  </div>
</div>

<style>
  .tokenize-container {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    padding: 1.5rem;
    background: #0f172a;
    color: #f8fafc;
    border-radius: 12px;
    border: 1px solid #334155;
  }
  .header-banner { display: flex; flex-direction: column; gap: 0.25rem; }
  .eyebrow { font-size: 0.75rem; color: #38bdf8; font-weight: bold; letter-spacing: 0.05em; }
  .title { margin: 0; font-size: 1.25rem; color: #f8fafc; }

  .pipeline-flow { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }
  .flow-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.85rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    min-width: 140px;
  }
  .flow-card.wide { flex: 1; min-width: 260px; }
  .step-num { font-size: 0.65rem; color: #38bdf8; font-weight: bold; }
  .flow-card h4 { margin: 0; font-size: 0.85rem; color: #cbd5e1; }
  .flow-arrow { font-size: 1.25rem; color: #64748b; font-weight: bold; }

  .sentence-box { background: #0f172a; padding: 0.4rem 0.6rem; border-radius: 4px; font-family: monospace; font-size: 0.85rem; color: #a7f3d0; }
  .chip-list { display: flex; gap: 0.35rem; flex-wrap: wrap; }
  .word-chip { background: #0284c7; color: #fff; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-family: monospace; }
  .id-chip { background: #475569; color: #f8fafc; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-family: monospace; }

  .vec-table { display: flex; flex-direction: column; gap: 0.4rem; }
  .vec-row { display: flex; align-items: center; gap: 0.5rem; }
  .row-label { font-size: 0.75rem; color: #94a3b8; font-weight: bold; width: 50px; font-family: monospace; }
  .vec-cells { display: flex; gap: 0.25rem; align-items: center; }
  .v-cell { font-family: monospace; font-size: 0.7rem; padding: 0.15rem 0.35rem; border-radius: 3px; color: #fff; border: 1px solid rgba(255,255,255,0.1); }
  .more-cells { font-size: 0.7rem; color: #64748b; font-style: italic; }
</style>
