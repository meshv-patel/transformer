# Architecture Addendum — Enhancements 1–8

All eight fit into the Phase 1 architecture without a redesign. Net effect: the scene-engine gains two new cross-cutting stores (config, highlight-sync), the content model gains two new fields per scene (level-tiered body, speaker notes), and one new companion panel (Token Journey) subscribes to the full pipeline instead of one scene. Framework choice is now decided. Details below, then Phase 2 begins.

---

### 8. Framework decision — **Svelte** (not plain JS, not React)

Reasoning, given the new brief is "optimize for maintainability and animation quality":
- **Animation primitives are native.** `svelte/motion` (`spring`, `tweened`) and `svelte/transition` give the morph/fade/grow/flow behavior your prompt asks for, without hand-rolling a tween engine (which the plain-JS plan would have required).
- **True reactivity fits this app's shape exactly.** Requirement 2 (configurable `d_model`/heads/seq_len) and requirement 1 (interactive recomputation) mean state changes must cascade through many components (viz, shape trace, equation, code panel, token journey) simultaneously. Svelte stores do this with no boilerplate; plain JS would mean writing a pub/sub system by hand (~doable, but that's exactly the "reinventing a framework" trap); React would mean fighting re-render performance for 23 scenes × SVG-heavy animation.
- **Same stack family as Transformer Explainer**, so patterns translate directly — their Sankey/ribbon and matrix-heatmap components are Svelte + D3, which is now literally the same toolkit, not just the same "inspiration."
- **Compiles away.** No virtual DOM shipped to students; output is small, fast vanilla JS — matches "keep everything responsive" and is friendly to a department static-hosting box.
- Cost: needs a one-time `npm install` + `vite build` before deploy (answers your open question #3 from Phase 1 — I'm assuming a build step is acceptable since you approved this direction; flag if the server truly cannot run Node even once locally to produce the static `dist/` folder, which is all that ships).

### 1. Lecture Mode vs. Interactive Mode
Both modes read from the same scene registry and rendering components; they differ only in **where the numbers come from**:
- **Lecture Mode:** numbers come from the static precomputed `forward-pass.json` (deterministic, fixed sentence, safe for live teaching pacing).
- **Interactive Mode:** numbers come from a **live in-browser forward pass** — the same tiny toy model (weights included in the shipped bundle), run against student-entered input.
- **Scoping constraint (important):** a toy model this small (`d_model` ≤ 64, tiny training corpus) cannot meaningfully process arbitrary English. Interactive Mode uses a **curated vocabulary** (~150–300 common words covering pronouns, verbs, objects — enough for "the dog chased the cat" style sentences), with unknown words mapped to a visible `<unk>` token rather than silently failing. This is disclosed in-UI ("try these words...") rather than presented as a free-text GPT toy, which would produce misleading/garbage attention patterns and undermine the lecture.
- The live forward pass itself is trivial compute (tiny matmuls) — implemented as a ~150-line hand-rolled linear-algebra module (`core/tensor-ops.js`: matmul, softmax, layernorm, no external ML dependency needed at this scale).
- Presenter Mode defaults to Lecture Mode for predictability; a presenter can drop into Interactive Mode mid-lecture as a live demo ("let's try a different sentence") and return.

### 2. Configurable tensor dimensions
`configStore` holds `{ dModel, numHeads, seqLen }` (plus derived `dK = dModel / numHeads`). All scene components read shapes from this store rather than hardcoding `[4,16]` etc. — the tensor-shape-bar, matrix, and ribbon components are already being built parametric (§4.2/4.3 of the original doc), so this is additive, not a rework.
- **Lecture Mode:** config is locked to the values baked into `forward-pass.json` (so the precomputed data stays valid) — changing config while in Lecture Mode prompts a switch to Interactive Mode, where a fresh live forward pass is computed for the new shape.
- Guardrails: UI-exposed ranges kept small (`d_model` 8–64, heads 1–8 and must divide `d_model`, seq_len 2–8) — large enough to show *effect* (e.g., what happens with 1 head vs. 4), small enough to stay legible on a projector and cheap to compute live.

### 3. Animation timeline (scrub + jump + replay)
A new `TimelineScrubber` component sits below the presenter nav, driven by `sceneStore`:
- Top-level markers = 23 scenes (existing progress dots, now clickable to jump directly, not just sequential next/prev).
- Nested tick marks = sub-steps within the current scene's tween sequence (from §4.3 of the original doc).
- "Replay" re-runs the current scene's (or current sub-step's) tween sequence from its start state without changing position in the overall lecture.
- This is a UI/state addition on top of the existing scene-engine — no change to how individual scene animations are authored.

### 4. Three explanation levels
`scene-copy.js` entries gain a tiered body instead of one `body` string:
```js
body: {
  beginner: 'Plain-language, metaphor-driven — assumes no DL background.',
  mtech:    'Your current EXPLANATIONS-style technical prose (kept as-is).',
  research: 'Complexity, edge cases, citations to the original paper — extends the existing Deep Dive content rather than duplicating it.'
}
```
A global `levelStore` (Beginner / MTech / Research Notes) drives which tier renders, in both Presenter and Explore modes. `research` absorbs your existing Deep Dive math/misconceptions content rather than creating a fourth tier — Deep Dive becomes "Research Notes, expanded," so there's no redundant authoring surface.

### 5. Token Journey / Memory Inspector
A companion panel, toggleable in either mode, that is **not** scoped to one scene — it subscribes to the *entire* precomputed (or live) pipeline and a `selectedToken` id:
- Renders a compact row-per-stage strip (Token IDs → Embedding → +PE → Q/K/V → Attention output → post-Add&Norm → FFN out → final) showing that one token's vector as a small heatmap/sparkline at each stage, plus which other tokens it attended to most at the attention stage.
- Clicking any token anywhere in the main visualization sets `selectedToken` and opens/updates this panel — this is the natural connective tissue between individually-built scenes, and is why the data layer (§4.1 of the original doc) stores the *whole* forward pass up front rather than per-scene slices.

### 6. Synchronized visualization ↔ equation ↔ code
A `highlightStore` (a shared "current term id," e.g. `matmul-q` or `head-2-softmax`) is broadcast by whichever panel the student/presenter interacts with (hover a matrix cell, hover an equation term, hover a code line) and consumed by all three simultaneously.
- Requires each scene's data record to carry a small correspondence table: `{ vizElementId, equationTermId, codeLineId }` triples (extends the content model in §4.5 of the original doc, additive field `syncMap`).
- Rendering-wise this is three components (`EquationPanel`, `CodePanel`, existing viz) all subscribing to one store — cheap in Svelte, would be notably more wiring in plain JS or React.

### 7. Presenter-only speaker notes
`scene-copy.js` gains a `speakerNotes` object per scene, rendered only when `mode === 'presenter'`:
```js
speakerNotes: {
  teachingTips: ['Pause here and ask students to predict the shape before revealing it.'],
  misconceptions: ['Students often think heads process different tokens — clarify each head sees all tokens, but a smaller slice of each vector.'],
  suggestedQuestions: ['Why divide by √d_k specifically, not d_k?']
}
```
Purely additive to the content model; no new rendering machinery beyond a panel that's hidden in Explore Mode.

---

## Updated file structure (delta from Phase 1 doc, §4.2)

```
/src
  /core
    stores/
      sceneStore.js       — current scene, substep, mode (presenter/explore), lecture/interactive
      configStore.js       — dModel, numHeads, seqLen, locked-vs-live
      levelStore.js         — beginner/mtech/research
      highlightStore.js     — shared sync-highlight bus
      tokenJourneyStore.js  — selectedToken, derived per-stage trace
    tensor-ops.js          — matmul/softmax/layernorm for live Interactive Mode
    scene-engine.js        — unchanged from Phase 1
  /scenes                  — 23 files, unchanged in principle, now Svelte components
  /components
    TimelineScrubber.svelte
    ModeToggle.svelte          — Lecture / Interactive
    LevelToggle.svelte         — Beginner / MTech / Research Notes
    ConfigPanel.svelte         — d_model / heads / seq_len sliders (Interactive only)
    TokenJourneyPanel.svelte
    EquationPanel.svelte
    CodePanel.svelte
    SpeakerNotes.svelte
    ... (tensor-shape-bar, token-node, attention-heatmap, ribbon-flow — unchanged)
  /data
    forward-pass.json       — precomputed, Lecture Mode
    vocab.json              — curated Interactive-Mode vocabulary + toy weights
    scene-copy.js
/tools
  generate_forward_pass.py  — unchanged, offline numpy script
```

No other part of the Phase 1 document changes. Proceeding to Phase 2.
