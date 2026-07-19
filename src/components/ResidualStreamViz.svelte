<script>
  import { dModel as configDModel } from '../core/stores/configStore.js';
  $: currentDModel = $configDModel || 16;
</script>

<div class="residual-stream-container">
  <div class="header-banner">
    <span class="eyebrow">SETUP STAGE 3: RESIDUAL STREAM HIGHWAY</span>
    <h3 class="title">Transformer Residual & Normalization Architecture</h3>
  </div>

  <div class="architecture-diagram">
    <div class="node input-node">
      <span class="node-title">Input Vectors X</span>
      <span class="node-dim">[seq_len, {currentDModel}]</span>
    </div>

    <div class="connector-line">│</div>
    <div class="fork-join">
      <div class="skip-branch">
        <span class="branch-label">Skip Connection (Identity)</span>
      </div>
      <div class="sublayer-branch">
        <span class="branch-label">Multi-Head Attention</span>
      </div>
    </div>
    <div class="connector-line">↓</div>

    <div class="node add-node">
      <span class="node-title">Residual Addition ①</span>
      <span class="node-formula">X + MHA(X)</span>
    </div>

    <div class="connector-line">↓</div>

    <div class="node norm-node">
      <span class="node-title">Layer Normalization ①</span>
      <span class="node-formula">mean = 0.0, var = 1.0</span>
    </div>

    <div class="connector-line">│</div>
    <div class="fork-join">
      <div class="skip-branch">
        <span class="branch-label">Skip Connection (Identity)</span>
      </div>
      <div class="sublayer-branch">
        <span class="branch-label">Position-wise FFN</span>
      </div>
    </div>
    <div class="connector-line">↓</div>

    <div class="node add-node">
      <span class="node-title">Residual Addition ②</span>
      <span class="node-formula">X_ln1 + FFN(X_ln1)</span>
    </div>

    <div class="connector-line">↓</div>

    <div class="node norm-node final">
      <span class="node-title">Layer Normalization ②</span>
      <span class="node-dim">Block Output [seq_len, {currentDModel}]</span>
    </div>
  </div>
</div>

<style>
  .residual-stream-container {
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

  .architecture-diagram {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
    background: #1e293b;
    padding: 1.25rem;
    border-radius: 10px;
  }
  .node {
    background: #0f172a;
    border: 1px solid #3b82f6;
    padding: 0.6rem 1.25rem;
    border-radius: 6px;
    text-align: center;
    min-width: 260px;
  }
  .node.add-node { border-color: #10b981; background: #064e3b; }
  .node.norm-node { border-color: #a855f7; background: #4c1d95; }
  .node.final { border-color: #f59e0b; background: #78350f; }
  .node-title { display: block; font-weight: bold; font-size: 0.9rem; color: #f8fafc; }
  .node-dim, .node-formula { font-size: 0.75rem; color: #cbd5e1; font-family: monospace; }

  .connector-line { color: #38bdf8; font-weight: bold; font-size: 0.9rem; }
  .fork-join {
    display: flex;
    gap: 1rem;
    width: 100%;
    max-width: 380px;
    justify-content: space-around;
  }
  .skip-branch, .sublayer-branch {
    background: #334155;
    border: 1px dashed #64748b;
    padding: 0.4rem 0.75rem;
    border-radius: 4px;
    font-size: 0.75rem;
    color: #94a3b8;
  }
  .sublayer-branch { border-style: solid; border-color: #38bdf8; color: #38bdf8; font-weight: bold; }
</style>
