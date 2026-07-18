# Transformer Lecture App — Embedding + Positional Encoding

Two scenes now built at reference quality: **Word Embedding** (frozen —
see `docs/reference-implementation.md` §1–11 for its rationale and the
external-review response in §12) and **Positional Encoding** (new — §13
covers what it needed and what it reused unchanged).

## Positional Encoding — the short version

Six click-through beats: **Problem** (swap two words live, show the
embedding vectors are byte-for-byte identical — the "gotcha" the whole
scene is built around), **Table** (the real sinusoidal position matrix,
including two "beyond this sentence" rows to preface extrapolation),
**Combine** (embedding + position, cell-by-cell, for every token, with the
table row from the previous beat visibly morphing into the operand via the
same crossfade technique Embedding used), **Output** (confirm the shape
never changed — `[T, D]` in, `[T, D]` out), then the standard **Before/After**
and **Quick Check** closing beats.

Reused completely unchanged: Camera Director, the crossfade morph
technique, `VectorHeatmap`, `TensorShapeTrace`, `BeforeAfterSummary`,
`QuickCheck`, `core/motion.js`. Two shared files (`SceneShell.svelte`,
`WhyPanel.svelte`) needed genuine, backward-compatible generalization —
details in doc §13. Embedding's own behavior was verified unchanged by
rebuilding before any new content was written.

## What changed in response to the external architecture review

Evaluated critically, not implemented wholesale — see §12 of the doc for
full Accept / Accept-later / Reject reasoning. Adopted: a shared
`VectorHeatmap` component, named sub-step constants, a one-hot × Wₑ
matrix-multiplication framing in Deep Dive, concrete (not asserted)
position-agnosticism language in the Quick Check, responsive breakpoints
for the fixed-position chrome, a `core/motion.js` helper so
`prefers-reduced-motion` reaches Svelte's own JS-driven transitions (not
just hand-rolled CSS), lightweight JSON validation, and `aria-live` on the
quiz reveal. Rejected: a WebGL rendering migration, a speculative
`StateMachine`/`Director` framework, and DOM-measurement-based camera
targeting — each for a specific, stated reason, not just "we like it this
way."

## Run it

```bash
npm install
npm run dev
```

---

## Prior round notes (still accurate)

## What changed in this final pass

- **Two-stage camera moves** — the lookup step now gets a wide intro shot
  before pushing toward the table (search, then land), instead of one flat
  shot. Recap/quiz beats get zero camera motion — an invisible camera means
  getting out of the way, not finding somewhere new to point.
- **GPU-only glow** — replaced the looping `box-shadow` pulse with a one-shot
  `opacity`/`transform` halo layer, staggered per element, so nothing
  animates indefinitely on a screen that might sit open for minutes.
- **Sequential entrance choreography** — tokens, then arrows, then table,
  then each matched row igniting in turn — four distinct beats instead of
  overlapping motion.
- **On-screen curiosity hook** — "Every token has meaning now — but does the
  model know which one came first?" appears at the end of the materialize
  step, not just in the spoken narration.
- **Natural narration rewrite** — all four scripts rewritten for spoken
  cadence (pauses, contractions, rhetorical beats) instead of documentation
  tone.
- **Why panel analogies** — every entry now has an intuitive analogy (radio
  dial tuning, library shelf numbers, locker numbers, dimmer switches,
  wedding seating).
- **Structured Before/After** — four scannable lines (entered / happened /
  changed / leaves) instead of one paragraph.
- **Distractor-specific quiz rebuttals** — picking "Attention" gets a
  targeted rebuttal, not the same generic explanation as any other wrong
  answer.
- **Keyboard parity everywhere** — every hover-triggered sync-highlight
  (equation terms, code lines, lookup arrows, vector strips) now also
  responds to focus, with visible focus rings.
- **Contrast/legibility bumps** for projector use; compare-result now
  explicitly ties back to the Why panel's language.

## Run it

```bash
npm install
npm run dev
```

---

## Prior polish-pass notes (still accurate)


- **Camera Director** (`src/core/camera/`, `CameraStage.svelte`) — reusable
  pan/zoom/focus system. The camera pushes toward the lookup table, then
  pans up into the materializing vectors, dimming the periphery
  automatically. Fully authored (shot-list based), not DOM-measurement
  based — reusable by any future scene via one `SHOTS` array.
- **Crossfade morph** — the highlighted table row *becomes* the vector strip
  (shared-element transition via `svelte/transition`'s `crossfade`), instead
  of one fading out while an unrelated one fades in. This is the "living
  lookup table" behavior.
- **Four-question framing** — every scene's copy now answers What's
  happening / Why / What changed / What to observe, shown as a compact grid
  under the explanation text.
- **Full narration system** — per-sub-step: speaking duration, teaching
  objective, a ready-to-read script, an audience question with expected
  answer, a common misconception, and the transition line into the next
  beat. Open via "Show narration," presenter mode only.
- **Why panel** — persistent, expandable, distinct from Deep Dive: why
  embeddings are learned, why one-hot is inefficient, why IDs can't be
  compared, why continuous spaces help optimization — with a small inline
  one-hot-vs-dense visual.
- **Before/After beat** and **Quick Check beat** — two new structural
  sub-steps every scene should end with (see doc §5). The quiz's correct
  answer teases Positional Encoding by name.
- **Compare & cosine similarity** — toggle "Compare two tokens," click two
  vectors, see their cosine similarity plus a plain-language read
  ("quite similar" / "roughly unrelated" / …), or select one to see its
  nearest neighbor among the visible sample.
- **Accessibility** — `prefers-reduced-motion` disables camera motion and
  glow animations; all interactive elements are keyboard-operable.

## Run it

```bash
npm install
npm run dev
```

---

## Prior Phase 3 notes (still accurate)

Navigate to the "Word Embedding" scene (2nd stop) via the top timeline, or
just press → twice from the start.

## What to look at

- **Lecture Mode** (default): "The cat sat down" — the embedding vectors
  shown are the *actual output* of a real (if toy) forward pass, computed
  once offline (`tools/generate_forward_pass.py`, numpy, seeded) and shipped
  as `public/data/forward-pass.json`. Nothing here is decorative.
- **Sub-step 1 ("lookup-table")**: token cards animate in, arrows pulse down
  into a real excerpt of the embedding table (4 sentence words highlighted
  among a few real decoy rows) — click → (or the timeline's sub-tick) to
  advance to sub-step 2.
- **Sub-step 2 ("vector-materialize")**: the highlighted rows "become" full
  16-cell heatmap strips under each token, materializing cell-by-cell.
  Hover any cell for its exact value; hover a whole vector strip (or the
  equation/code panel below) to see the synced highlight light up across
  viz + equation + code together.
- **Replay** (⟲, bottom bar) reruns the current sub-step's entrance
  animation from scratch, without changing your position.
- **Timeline scrubber** (top) — click any dot to jump directly to that
  scene; sub-ticks under the active dot jump within Embedding's two
  sub-steps.
- **Switch to Interactive** (top right) — a word-picker appears; build your
  own short sentence from the curated vocabulary and watch new vectors
  generate live (see disclosure note in-app: this toy model isn't trained,
  so values are illustrative — the point is the *mechanism*, which is real:
  same lookup, same shape logic, live-recomputed).
- **Level toggle** (Beginner / MTech / Research Notes) — swaps the
  explanation body text; Research Notes pulls from the same Deep Dive
  content as the accordion.
- **Deep Dive accordion** (below the explanation, or press `D`) — math,
  complexity, misconceptions, notes.
- **Speaker notes** (bottom right, presenter mode only) — teaching tips,
  a common misconception, and a discussion question, authored for this scene.
- **Click a token** — sets `selectedTokenPos`; opens the Token Journey panel
  (still shows placeholder stage-cells for now — it becomes fully useful once
  more scenes exist for a token to travel through in Phase 4).

## What's new in the code since Phase 2

```
src/scenes/viz/EmbeddingViz.svelte   — the reference scene component
src/scenes/viz/index.js              — SCENE_VIZ map: scene id -> real component
                                        (single source of truth for "is this
                                        scene implemented?", replacing the old
                                        implemented:false flag)
src/components/TensorShapeTrace.svelte — reusable shape-flow component
src/core/embedding-utils.js          — seeded embedding generator (Interactive Mode)
src/data/vocab.js                    — curated word list (source of truth;
                                        public/data/vocab.json is the served copy)
```

`SceneShell.svelte` now renders `SCENE_VIZ[scene.id]` when present, falls
back to the "not yet implemented" placeholder otherwise, and gained a working
Deep Dive accordion (`deepDiveOpen` store in `sceneStore.js`, wired to the
`D` key). `tools/generate_forward_pass.py` now also exports `embeddingSample`
— real rows from the embedding matrix used for the lookup-table visual.

## Pattern to copy for every Phase 4 scene

1. Add real computation to `tools/generate_forward_pass.py` if the stage
   needs new precomputed data (most already do, from the encoder pass).
2. Write `src/scenes/viz/<SceneName>Viz.svelte` following `EmbeddingViz.svelte`:
   - Read data via `$forwardPassData` (Lecture) or compute live via
     `core/tensor-ops.js` (Interactive), gated on `$dataMode`.
   - Gate visuals on `$subStepIndex`; wrap the animated region in
     `{#key runKey}` (see the pattern in `EmbeddingViz.svelte`) so Replay and
     sub-step changes restart transitions cleanly.
   - Use `TensorShapeTrace` for the shape callout.
   - Wire any hover-highlightable element to `highlightStore` with an id
     matching the scene's `syncMap` entries in `scene-copy.js`.
   - On click, if the interaction is "this represents one token," set
     `selectedTokenPos`.
3. Add the scene's copy (`body` in 3 levels, `deepDive`, `pytorch`,
   `syncMap`, `speakerNotes`) to `src/data/scene-copy.js`.
4. Register it: `import` + one line in `src/scenes/viz/index.js`.

That's the whole contract — `SceneShell`, the timeline, explore nav, and all
chrome pick it up automatically.

## Still open / unchanged from Phase 2

Every other scene still shows the placeholder. Token Journey's per-stage
data will fill in as more scenes come online. Interactive Mode's config
sliders (d_model/heads/seq_len) are visible and functional as controls, but
only `d_model` is actually consumed by this scene so far — heads/seq_len
matter starting with the attention scenes in Phase 4.
