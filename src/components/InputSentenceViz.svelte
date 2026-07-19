<script>
  import { forwardPassData } from '../core/data-loader.js';

  $: encSentence = $forwardPassData?.meta?.sentence ?? ['cat', 'chased', 'dog'];
  $: decSentence = $forwardPassData?.meta?.targetSentence ?? ['the', 'dog', 'ran'];
</script>

<div class="input-sentence-container">
  <div class="header-banner">
    <span class="eyebrow">SETUP STAGE 1: SEQUENCE INPUTS</span>
    <h3 class="title">Encoder & Decoder Dual Sequence Inputs</h3>
  </div>

  <div class="dual-grid">
    <!-- Encoder Input Panel -->
    <div class="stream-panel encoder-stream">
      <div class="stream-header">
        <span class="stream-badge enc">Source Stream (Encoder)</span>
        <h4>Input Context Sequence</h4>
      </div>
      <p class="desc">The Encoder processes the full source sequence in parallel.</p>
      <div class="token-list">
        {#each encSentence as word, i}
          <div class="token-card">
            <span class="pos-tag">Position {i}</span>
            <span class="word-text">"{word}"</span>
          </div>
        {/each}
      </div>
    </div>

    <!-- Decoder Input Panel -->
    <div class="stream-panel decoder-stream">
      <div class="stream-header">
        <span class="stream-badge dec">Target Stream (Decoder)</span>
        <h4>Target Context Sequence</h4>
      </div>
      <p class="desc">The Decoder processes target tokens with causal masking during training/inference.</p>
      <div class="token-list">
        {#each decSentence as word, i}
          <div class="token-card dec-card">
            <span class="pos-tag">Position {i}</span>
            <span class="word-text">"{word}"</span>
          </div>
        {/each}
      </div>
    </div>
  </div>
</div>

<style>
  .input-sentence-container {
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

  .dual-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .stream-panel {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .stream-header { display: flex; flex-direction: column; gap: 0.25rem; }
  .stream-header h4 { margin: 0; font-size: 1rem; color: #cbd5e1; }
  .stream-badge {
    align-self: flex-start;
    font-size: 0.7rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-weight: bold;
  }
  .stream-badge.enc { background: #0284c7; color: #fff; }
  .stream-badge.dec { background: #7c3aed; color: #fff; }
  .desc { font-size: 0.8rem; color: #94a3b8; margin: 0; }
  .token-list { display: flex; gap: 0.5rem; flex-wrap: wrap; }
  .token-card {
    background: #0f172a;
    border: 1px solid #0284c7;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .token-card.dec-card { border-color: #8b5cf6; }
  .pos-tag { font-size: 0.65rem; color: #64748b; }
  .word-text { font-size: 0.9rem; font-weight: bold; color: #f8fafc; font-family: monospace; }
</style>
