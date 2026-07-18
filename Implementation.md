# Master Implementation Plan — Transformer Lecture App

**Status:** Single source of truth, effective immediately. Supersedes
`architecture-addendum.md` and all narrative/roadmap content in
`docs/reference-implementation.md` wherever either conflicts with this
document or with the actual repository. Where a prior document and the
repository disagreed, the repository won — every such case is called out
explicitly in the section it affects, not hidden.

Everything in this document was verified against the merged repository
(`npm run build` passes clean) as of this pass. Nothing here was taken on
either prior document's word alone.

---

# 1. Repository State

## 1.1 What already exists (implemented, working, build-verified)

- **2 of 21 registry scenes are fully implemented:** `embedding` and
  `positional-enc`. Both render, both pass a clean `npm run build`.
- **Full scene-engine infrastructure**: routing (`sceneStore.js`), camera
  (`cameraStore.js` + `CameraStage.svelte`), sync-highlight bus
  (`highlightStore.js` + `CodePanel`/`EquationPanel`/`TensorShapeTrace`),
  content model (`scene-copy.js` + `getSceneCopy()`), timeline/explore
  navigation, presenter/explore mode split, Lecture/Interactive mode split,
  three-tier explanation levels, Deep Dive accordion, Why panel, speaker
  notes, Before/After + Quick Check structural beats.
- **A real, non-toy data pipeline**: `tools/generate_forward_pass.py` runs
  an actual numpy forward pass (seeded, reproducible) through the full
  encoder — tokenize → embed → positional-encode → Q/K/V projection →
  multi-head attention (real per-head softmax, not aggregated) → output
  projection → residual+norm → FFN → residual+norm → encoder output — and
  ships the result as `public/data/forward-pass.json` / `src/data/forward-pass.json`
  (identical copies, verified byte-for-byte).
- **A working live-compute layer for Interactive Mode**:
  `core/tensor-ops.js` (matmul, transpose, softmax, layerNorm, relu,
  splitHeads, concatHeads, and a complete `attention()` function),
  `core/embedding-utils.js` (seeded pseudo-embeddings), and
  `core/positional-encoding.js` (the exact real sinusoidal formula, not an
  approximation).

## 1.2 What is finished and frozen

Per `docs/reference-implementation.md` §9 and verified unchanged in the
merged repo: **`EmbeddingViz.svelte`'s own internal behavior is frozen.**
It is the reference implementation every future scene's *pattern* should
copy, not a file future work edits. `PositionalEncodingViz.svelte` is
likewise complete and frozen — it's the second, confirmatory data point
that the shared-infrastructure generalizations (below) are correctly
shaped, not a scene to keep iterating on.

**Do not modify these files as part of building new scenes:**
- `src/scenes/viz/EmbeddingViz.svelte`
- `src/scenes/viz/PositionalEncodingViz.svelte`
- `src/data/scene-copy.js`'s existing `embedding` and `positional-enc`
  entries (new scenes add new entries; they don't touch these two)
- `src/core/scene-registry.js`'s existing 21 entries' `id`/`group`/`title`
  fields (a scene's `subSteps` array may still need small additions later
  if a structural before-after/quick-check pair was omitted — see §4 gap
  list — but the identity fields are stable)
- `tools/generate_forward_pass.py`'s existing stages (`tokenize` through
  `encoder-output`) — the encoder forward pass is already fully computed
  and correct; only *additive* exports (like the `weights` block) should
  ever be appended, never the existing stage computations edited

## 1.3 What is reusable (see §3 for full detail)

Every shared component in `src/components/`, every store in
`src/core/stores/` + `src/core/camera/`, every math utility in
`src/core/`, and the whole content/rendering contract in `SceneShell.svelte`
+ `scene-copy.js`. This is the majority of the engineering surface area —
remaining work is overwhelmingly *authoring* (data + copy + one viz
component per scene), not infrastructure building.

## 1.4 What should never be modified without a Rule-of-Three trigger

Anything the reference doc marks "copy verbatim" (§9 of
`docs/reference-implementation.md`): `CameraStage`, the camera store usage
pattern, the crossfade pattern, `TensorShapeTrace`, `BeforeAfterSummary`,
`QuickCheck`, `WhyPanel`, `SpeakerNotes`, the `sentenceKey`/`stageKey`
remount split, the `$dataMode` branch pattern, the `glow-wrap`/`glow-halo`
one-shot ignite pattern, and the `STEP` named-constant pattern. These are
already proven across two independent scenes (Embedding, Positional
Encoding) built by different authors — that's the Rule of Three's
"observed duplication, not guessed" bar already cleared for these
specific patterns. They are not to be redesigned; only extended the way
`SceneShell`/`WhyPanel` were (additive, backward-compatible).

---

# 2. Architecture Decisions (final — do not revisit)

- **SceneShell philosophy.** One shell renders every scene's chrome
  (four-question grid, code/equation sync panels, Why panel, Deep Dive,
  speaker notes) from a declarative `scene-copy.js` entry; only the
  *visualization* is scene-specific (`SCENE_VIZ[scene.id]`). Verified
  correct and unchanged in the merged `SceneShell.svelte` — this is the
  entire reason 19 remaining scenes require near-zero shell work.
- **`copy.codeLines` / `copy.pytorch` / `copy.equationTerms` — final
  merged form.** The merge combined two independently-evolved
  generalizations of the same mechanism. The final, verified-working
  priority order in `SceneShell.svelte` is: explicit `copy.codeLines`
  array (if present) → array-form `copy.pytorch` (if an array, e.g. PE's
  two-line case) → single-string `copy.pytorch` (legacy, auto-wrapped).
  `copy.equationTerms` array → single `copy.deepDive.math` string
  (auto-wrapped) fallback. **New scenes should default to the array forms**
  (`codeLines: [{id, code}]`, `equationTerms: [{id, tex}]`) directly —
  the string-fallback paths exist for backward compatibility only and
  should not be used by newly-authored scenes.
- **Camera philosophy.** Shot-authored, not DOM-measured
  (`docs/reference-implementation.md` §2, reaffirmed in §12's rejection of
  DOM-rect-measured targeting). This is now validated by two scenes
  independently authoring their own `SHOTS` arrays with no shared
  camera-logic changes needed. **Final. Do not revisit** — this exact
  proposal was raised and rejected once already by the external review.
- **Rule of Three.** Explicitly the operating principle for this codebase
  (invoked directly in doc §12 and §13 to justify *not* building a
  `StateMachine`/`Director` framework from one data point). Extend to
  every future extraction decision: **wait for the third real instance
  of duplication** before promoting scene-owned markup/logic to a shared
  component — except where the repository already provides verified
  evidence that a third consumer is imminent and known (see §6, the one
  case that applies).
- **Interactive vs. Lecture mode.** Confirmed as designed: Lecture reads
  static `forward-pass.json`; Interactive computes live via
  `tensor-ops.js`/`embedding-utils.js`/`positional-encoding.js`, gated by
  `$dataMode`, with `configStore.js` locking to `LECTURE_DEFAULTS`
  (`dModel:16, numHeads:4, seqLen:4`) whenever Lecture Mode is active.
  Every remaining scene should branch on `$dataMode` exactly as
  `EmbeddingViz.svelte` does. **Final, no changes needed.**
- **Highlight/sync system.** One `highlightedTermId` store, consumed by
  `CodePanel`, `EquationPanel`, and any scene's own SVG/DOM elements via
  matching `id` strings declared in a scene's `syncMap`. Confirmed generic
  and scene-agnostic already — no further generalization needed.
- **Animation philosophy.** `transform`/`opacity`-only animation,
  `glow-wrap`/`glow-halo` one-shot pattern (not looping `box-shadow`),
  sequential (not simultaneous) entrance choreography, two-stage
  "wide intro then push" camera moves for search-then-land beats, zero
  camera motion on recap/quiz beats. All confirmed present and consistent
  across both implemented scenes. **Final.**
- **Accessibility philosophy.** `prefers-reduced-motion` handled at two
  levels — the camera spring collapses to an instant snap
  (`cameraStore.js`), and `core/motion.js`'s `motionMs()` wraps every
  Svelte-JS-driven transition duration/delay (`fly`/`fade`/`crossfade`
  cannot be neutralized by CSS media queries alone, since they're
  JS-computed inline styles). Keyboard parity is mandatory for every
  hover-triggered highlight element (`tabindex="0"` + `role="button"` +
  matching `on:focus`/`on:blur` — confirmed as a checklist item, not a
  judgment call, in doc §10). **Final — apply to every new scene without
  exception.**
- **Registry philosophy.** `scene-registry.js` is a flat, ordered array of
  `{id, group, title, subSteps}` with `subSteps` declared up front (even
  for unimplemented scenes) so the timeline scrubber has real tick marks
  before a scene's animation exists. Confirmed this is still exactly how
  it works — `IMPLEMENTED_SCENE_IDS` (derived from `SCENE_VIZ`'s keys in
  `viz/index.js`) is the single source of truth for "is this scene real
  yet," consumed by both `ExploreNav` and `TimelineScrubber`. **Final.**
- **Data-generation philosophy.** One offline numpy script
  (`tools/generate_forward_pass.py`), one seed, one JSON output consumed
  by both Lecture Mode and (for parity-checking) Interactive Mode's
  from-scratch JS reimplementation. **Final — confirmed the encoder pass
  already fully executes end-to-end through `encoder-output` in the
  script**, so no future scene needs new *stage computations*, only
  (per §4) a few additive weight-matrix exports mirroring the pattern
  already used for Q/K/V.

## 2.1 Where a previous report's recommendation is rejected

- **Architecture-addendum.md's "23 scenes" claim (appears 3×, including
  in the file-structure comment and the React-vs-Svelte performance
  argument) is stale.** The actual registry (`scene-registry.js`,
  verified by direct count) has **21 scenes**, and the pipeline is
  **encoder-only** — there is no decoder, no final softmax-over-vocabulary
  scene, and no cross-attention. This document's scene count and ordering
  (§5) reflect the real registry. Nothing about the "23 scenes" framing
  changes any architectural decision (Svelte's reactivity argument holds
  regardless of exact count), so no further action is needed beyond using
  the correct number going forward.
- **The external architecture review's WebGL rendering migration,
  `StateMachine`/`Director` framework, and DOM-measurement-based camera
  targeting were already rejected in `docs/reference-implementation.md`
  §12, with reasons that were re-verified here and still hold** (canvas
  content isn't screen-reader-navigable; only one/two scenes exist, too
  early to guess the right shared-state abstraction; shot-authored camera
  is deliberately not measurement-based for cross-viewport predictability).
  This plan does not reopen any of the three.
- **`vocab.js`'s own comment promises growth toward "the full ~150-300
  word list" "in later phases."** The repository currently ships 12 words.
  This is not contradicted by anything — it's an honestly-labeled,
  not-yet-done item, tracked as a real gap in §4, not a false claim to
  correct.

---

# 3. Verified Infrastructure

### Stores

| Store | Purpose | Current users | Future users | Frozen? |
|---|---|---|---|---|
| `sceneStore.js` (`sceneIndex`, `subStepIndex`, `appMode`, `dataMode`, `currentScene`, `replayTick`, `deepDiveOpen`, `next()`/`prev()`/`goToScene()`/`replayCurrent()`) | Scene/sub-step navigation, presenter/explore + lecture/interactive mode flags | `SceneShell`, `PresenterControls`, `TimelineScrubber`, `ExploreNav`, both viz components, `App.svelte` | Every future scene | Yes |
| `configStore.js` (`dModel`, `numHeads`, `seqLen`, `dK`, `RANGES`, `isValidHeadSplit`, `resetToLectureDefaults()`) | Interactive-Mode tensor-shape config, with Lecture-Mode lock | `ConfigPanel`, `ModeToggle`, `EmbeddingViz` | Every Attention/FeedFwd scene that supports Interactive Mode | Yes |
| `levelStore.js` (`explanationLevel`, `LEVELS`) | Beginner/MTech/Research toggle | `SceneShell`, `LevelToggle` | Every future scene (reads `copy.body[level]`) — nothing to build | Yes |
| `highlightStore.js` (`highlightedTermId`, `setHighlight()`, `clearHighlight()`) | Cross-panel sync-highlight bus | `CodePanel`, `EquationPanel`, `TensorShapeTrace`, both viz components' SVG elements | Every future scene's `syncMap`-driven elements | Yes |
| `tokenJourneyStore.js` (`selectedTokenPos`, `tokenTrace`) | Per-token cross-stage trace, derived from the whole `forwardPassData` pipeline | `TokenJourneyPanel`, `EmbeddingViz` (click-to-select) | Becomes materially more useful once every stage exists — currently a placeholder strip (`stage.vector.length` only, no real heatmap render) | Yes (store logic); **the panel's render body is a known, tracked gap — see §4** |
| `cameraStore.js` (`camera`, `focusShot()`, `resetCamera()`, `isFocused()`) | Shot-authored pan/zoom/scale | `CameraStage`, both viz components | Every future scene | Yes |

### Reusable components

| Component | Purpose | Current users | Future users | Frozen? |
|---|---|---|---|---|
| `SceneShell.svelte` | Renders all shared per-scene chrome from `scene-copy.js` | Both scenes (via `App.svelte`) | All 19 remaining scenes, with zero changes required | Yes |
| `CameraStage.svelte` | Wraps a scene's visual; applies the camera transform + vignette | Both viz components | All remaining scenes | Yes |
| `VectorHeatmap.svelte` (+ `core/heatmap-color.js`) | 1D vector-as-colored-cells strip | Both viz components (word rows, position rows, materialized vectors) | Any scene rendering a `[seq, d_model]` or `[seq, d_k]` vector row — proj-q/k/v, weighted-sum, output-proj, residual/norm scenes, ffn | Yes |
| `TensorShapeTrace.svelte` | Shape-flow breadcrumb (`[label, shape][]`) | Both scenes | All remaining scenes | Yes |
| `BeforeAfterSummary.svelte` | Structural before/after beat | Both scenes | All remaining scenes (every scene's structural closing pair) | Yes |
| `QuickCheck.svelte` (incl. `distractorNotes`) | Structural quiz beat | Both scenes | All remaining scenes | Yes |
| `WhyPanel.svelte` (generalized: `left`/`right`/`leftLabel`/`rightLabel`/`leftNote`/`rightNote`/`leftStyle`/`rightStyle`) | Persistent "why does this exist" panel | Both scenes | Any future scene with a two-way conceptual comparison — verified genuinely generic now (2-for-2), not just theoretically so | Yes |
| `SpeakerNotes.svelte` | Presenter-only per-substep narration panel | Both scenes | All remaining scenes | Yes |
| `CodePanel.svelte` / `EquationPanel.svelte` | Sync-highlighted code/equation panels | Both scenes via `SceneShell` | All remaining scenes | Yes |
| `TimelineScrubber.svelte` / `ExploreNav.svelte` | Global navigation chrome | App-level | No scene-level changes needed — driven entirely by `scene-registry.js` + `IMPLEMENTED_SCENE_IDS` | Yes |
| `ConfigPanel.svelte` / `ModeToggle.svelte` / `LevelToggle.svelte` | Global mode/config chrome | App-level | No changes needed | Yes |
| `TokenJourneyPanel.svelte` | Cross-stage token trace panel | App-level | Render body needs real content once ≥3–4 stages exist (see §4) | Shell frozen; render body is a tracked gap |
| `PresenterControls.svelte` / `Layout.svelte` | App chrome / keyboard shortcuts | App-level | No changes needed | Yes |

### Reusable utilities

| Utility | Purpose | Current users | Future users | Frozen? |
|---|---|---|---|---|
| `core/tensor-ops.js` (`matmul`, `transpose`, `addMatrices`, `scale`, `softmaxRow(s)`, `layerNorm`, `relu`, `splitHeads`, `concatHeads`, `attention`) | Live compute for Interactive Mode | Not yet wired into any scene (Phase 2 stubs, per its own header comment) — **but the math is complete and correct**, including a full scaled-dot-product `attention()` | `proj-q/k/v` (matmul), `split-heads` (splitHeads), `qk-matmul`/`scale-softmax`/`weighted-sum` (attention), `concat` (concatHeads), `layer-norm-1/2` (layerNorm), `ffn` (matmul+relu) | Yes — this is verified complete, not a gap |
| `core/vector-math.js` (`dot`, `norm`, `cosineSimilarity`, `nearestNeighbor`) | Vector comparison | `EmbeddingViz`'s compare-tokens feature | Any future scene wanting a similar "compare two vectors" interaction | Yes |
| `core/embedding-utils.js` | Seeded pseudo-embeddings for Interactive Mode | `EmbeddingViz` | No other scene needs this specifically (embedding-only concern) | Yes |
| `core/positional-encoding.js` | Real sinusoidal PE formula (JS) | `PositionalEncodingViz` | No other scene needs this specifically | Yes |
| `core/heatmap-color.js` | Shared value→color scale | `VectorHeatmap.svelte` | The as-yet-unbuilt `MatrixHeatmap.svelte` (see §6) — already written with this exact future consumer in mind | Yes |
| `core/motion.js` (`motionMs`, `prefersReducedMotion`) | Reduced-motion-aware transition timing | Both viz components | All remaining scenes' transitions | Yes |
| `core/data-loader.js` (`forwardPassData`, `loadLectureData()`, `setInteractiveData()`, structural validation) | Loads/validates Lecture data, holds whichever dataset is "live" | `App.svelte`, `tokenJourneyStore.js` | All remaining scenes read `$forwardPassData` the same way | Yes |
| `core/scene-registry.js` | Scene metadata + ordering | Nav chrome, both scenes | All remaining scenes (already declared, no edits needed to add a viz — only to `SCENE_VIZ`) | Yes (existing 21 entries) |

---

# 4. Outstanding Infrastructure Gaps (verified only)

Each gap below was confirmed by direct inspection — nothing here is
guessed from a document.

### Gap 1 — `Wo`/`b_o` and `W_ff1`/`b_ff1`/`W_ff2`/`b_ff2` are not exported to `forward-pass.json`
- **Why it exists:** the recent infra update exported `Wq/bq/Wk/bk/Wv/bv`
  to the `weights` JSON block (needed for proj-q/k/v's "weight-matrix"
  sub-step) but stopped there. `W_o`, `b_o`, `W_ff1`, `b_ff1`, `W_ff2`,
  `b_ff2` all exist as in-memory numpy arrays in
  `tools/generate_forward_pass.py` (confirmed by direct read, lines
  ~58–61) and are already used to compute the `output-proj` and `ffn`
  stages — they simply were never added to the `weights` export block.
- **Required?** Yes, for `output-proj`'s and `ffn`'s "weight-matrix"-style
  sub-steps to render real numbers instead of a placeholder — this is the
  exact same problem already solved once for Q/K/V, not a new one.
- **Priority:** High for `output-proj` and `ffn` specifically; blocks
  nothing else.
- **Which future component needs it:** `output-proj` scene (visualizing
  `Wo`), `ffn` scene (visualizing `W_ff1`/`W_ff2` for the expand/project-back
  sub-steps).
- **Not speculative** — this is a verified, concrete, narrow gap with an
  exact precedent already in the codebase to copy.

### Gap 2 — No `MatrixHeatmap.svelte` component exists yet
- **Why it exists:** `core/heatmap-color.js`'s own comment already names
  `MatrixHeatmap.svelte` as a planned consumer ("Self-Attention scenes"),
  but no such file exists anywhere in `src/components/` (confirmed by
  direct search). `VectorHeatmap.svelte` only renders 1D strips; nothing
  currently renders a 2D `[seq, seq]` or `[heads, seq, seq]` grid.
- **Required?** Yes — `qk-matmul` (`attentionScores`, shape
  `[heads, seq, seq]`, already computed and present in
  `forward-pass.json`), `scale-softmax` (`attention`, same shape, already
  present), and `heads-compare` (`per-head-heatmaps` sub-step, same data)
  all need a 2D matrix-grid visual. All three are consecutive scenes in
  the registry.
- **Priority:** High — first blocking dependency for the entire back half
  of the Attention group.
- **Which future component needs it:** `qk-matmul`, `scale-softmax`,
  `heads-compare` (all three, immediately, in sequence).
- **Rule-of-Three note:** see §6 for why this is treated as a build-once
  shared component rather than scene-owned-first, despite zero scenes
  currently using it — the exception is justified there, not assumed here.

### Gap 3 — `TokenJourneyPanel.svelte`'s render body is a placeholder
- **Why it exists:** confirmed by direct read — the component currently
  renders only `[${stage.vector.length}]` text per stage, with an explicit
  comment: `<!-- Phase 3/4: render an actual heatmap/sparkline of
  stage.vector here -->`. The store logic (`tokenJourneyStore.js`) is
  complete and already derives the right data shape; only the visual
  render is unbuilt.
- **Required?** Not blocking for any individual scene — but becomes
  genuinely useful (per doc §11's own self-critique) only once several
  more stages exist for a token to visibly travel through.
- **Priority:** Medium — revisit once ≥4–5 Attention-group scenes are
  implemented (enough stages for the strip to be worth looking at).
  Trivial to build (`VectorHeatmap` at `size="small"` per stage cell) —
  no new component needed, just wiring.
- **Which future component needs it:** none block on it; it improves in
  quality as a side effect of other scenes shipping.

### Gap 4 — `vocab.js` has 12 words, not the ~150–300 originally scoped
- **Why it exists:** `vocab.js`'s own comment says this directly — "Phase
  3 scope: a small curated list... grows toward the full ~150-300 word
  list in later phases without changing the component contract." This is
  an honestly-labeled incomplete item, not a bug.
- **Required?** Only for Interactive Mode's sentence-building UI to feel
  less repetitive; not required for any scene's correctness, since
  Interactive Mode already degrades gracefully (a small vocab is still a
  fully working demo).
- **Priority:** Low. Purely a content/breadth task, zero architectural
  risk, can happen any time — batching it with any scene's work is fine,
  it doesn't need its own slot in the dev order.
- **Which future component needs it:** none specifically; general
  Interactive Mode quality-of-life.

### Explicitly NOT gaps (verified, to close the door on speculative asks)
- **Per-head attention data for `heads-compare`/`concat`/`split-heads`
  does *not* need new Python generation work.** Verified directly:
  `scale-softmax.attention` is already `[numHeads, seqLen, seqLen]`
  (`np.einsum("hqd,hkd->hqk", ...)`, per-head, not aggregated). `split-heads`
  and `concat` are pure reshapes of already-exported `proj-q/k/v` and
  `weighted-sum` tensor data, exactly what `tensor-ops.js`'s `splitHeads`/
  `concatHeads` already implement. No data-generation-script work is
  needed for any of these three scenes.
- **No in-browser tensor-execution runtime needs building.**
  `core/tensor-ops.js` already is one, complete, at the correct scale, with
  the correct dependency-free design decision already made and defended
  in doc §12.
- **No new orchestration/state-machine framework is a gap.** Explicitly
  rejected already (§12) and re-confirmed still correct here — nothing in
  the current 21-scene registry needs it; `sceneStore.js`'s existing
  two-phase `next()`/`prev()` already correctly handles both single- and
  multi-substep scenes.

---

# 5. Scene Implementation Plan

Registry order (verified from `scene-registry.js`, 21 total, 2 done, 19
remaining). Every remaining scene reuses the same shared-component set
unless noted; only per-scene specifics are listed under "New" fields.

**Shared across every remaining scene (not re-listed per scene):**
Shared components: `CameraStage`, `TensorShapeTrace`, `BeforeAfterSummary`,
`QuickCheck`, `SceneShell`-rendered `WhyPanel`/`SpeakerNotes`/`CodePanel`/
`EquationPanel`. Stores: `sceneStore`, `highlightStore`, `dataMode`,
`configStore` (if Interactive-capable). Utilities: `core/motion.js`
(`motionMs`), camera `SHOTS` pattern, `glow-wrap`/`glow-halo` pattern,
`sentenceKey`/`stageKey` remount split. Educational components: Before/After
+ Quick Check as the last two sub-steps (per doc §5's structural pattern).
Speaker Notes: one `narration[]` entry per sub-step, always.

### `residual-stream` (Setup)
- **Viz:** New, small (`ResidualStreamViz.svelte`) — conceptual "this
  vector persists and gets added to at every later stage" intro, not a
  new computation.
- **Reused:** `VectorHeatmap` (re-displays `positional-enc`'s output).
- **Interactive:** trivial — same `$dataMode` branch, no new compute.
- **Camera:** single wide shot; this is a narrative pause scene, not a
  computation.
- **Complexity:** Low. **Dependencies:** none (data already exists).
- **Expected reusable abstractions:** none new.

### `proj-q`, `proj-k`, `proj-v` (Attention — build as one authoring pass, 3 near-identical scenes)
- **Viz:** New (`LinearProjectionViz.svelte`, parameterized by Q/K/V role
  — this is the Rule-of-Three case *forming*, not yet crossed; author
  scene-owned first for `proj-q`, then check after `proj-k` whether the
  third repetition (`proj-v`) still looks like a pure parameterization or
  reveals real per-role differences before extracting).
- **Shared components reused:** `VectorHeatmap` (input row, weight matrix
  rows, output row), `TensorShapeTrace`.
- **Data:** `proj-q`/`proj-k`/`proj-v` stages (`tokenVectors`, already
  present) + `weights.Wq/bq` etc. (already present, per repo state).
- **Interactive:** live via `tensor-ops.js`'s `matmul`.
- **Camera:** push toward the weight matrix, then toward the output row —
  two-stage, matches the "search then land" pattern.
- **Color convention:** use `--accent` (Q), `--accent-2` (K), `--accent-3`
  (V) — already defined in `app.css`, already labeled in comments in
  `heatmap-color.js`. Use for chrome/borders/labels only, never as cell
  fill (would collide with the value-sign color scale).
- **Complexity:** Medium (×3, but the third and later ones should be
  fast once the pattern is set).
- **Dependencies:** none blocking.
- **Expected reusable abstractions:** a `LinearProjectionViz` component
  parameterized by role, *if* `proj-v` confirms the pattern holds
  (Rule of Three).

### `split-heads` (Attention)
- **Viz:** New (`HeadsSplitViz.svelte`) — visual reshape of one of the
  Q/K/V outputs into per-head slices.
- **Shared components reused:** `VectorHeatmap` (per-head sub-strips),
  `TensorShapeTrace` (shows `[seq,d_model] → [heads,seq,d_k]`).
- **Data:** derived client-side via `tensor-ops.js`'s `splitHeads()` from
  an already-exported stage — **no new data-generation work** (§4,
  "explicitly not a gap").
- **Interactive:** trivial, same reshape function either mode.
- **Complexity:** Low–Medium.
- **Dependencies:** none.
- **Expected reusable abstractions:** none new.

### `qk-matmul` (Attention)
- **Viz:** New (`QkMatmulViz.svelte`).
- **Shared components reused:** `MatrixHeatmap` (**new — see §6**),
  `TensorShapeTrace`.
- **Data:** `qk-matmul.attentionScores`, already present
  (`[heads,seq,seq]`).
- **Interactive:** live via `tensor-ops.js`'s `matmul(q, transpose(k))`
  scaled by `1/sqrt(dK)` — `attention()`'s first half.
- **Camera:** push toward the grid.
- **Complexity:** Medium (first scene needing `MatrixHeatmap` — expect
  this slot to include that component's build time).
- **Dependencies:** `MatrixHeatmap.svelte` must exist first (§6).
- **Expected reusable abstractions:** `MatrixHeatmap.svelte` gets built
  here (see §6, §8 dev order).

### `scale-softmax` (Attention)
- **Viz:** New (`ScaleSoftmaxViz.svelte`).
- **Shared components reused:** `MatrixHeatmap` (now already built).
- **Data:** `scale-softmax.attention`, already present.
- **Interactive:** live via `tensor-ops.js`'s `softmaxRows`.
- **Complexity:** Low–Medium (infra already paid for by `qk-matmul`).
- **Dependencies:** `MatrixHeatmap.svelte`.
- **Expected reusable abstractions:** none new.

### `weighted-sum` (Attention)
- **Note:** registry already carries `before-after`/`quick-check` in its
  `subSteps` (added by the recent infra merge) — this scene's structural
  closing pair is already scaffolded in the registry, just needs
  authoring.
- **Viz:** New (`WeightedSumViz.svelte`).
- **Shared components reused:** `VectorHeatmap`, `MatrixHeatmap` (to show
  the weights being applied), `TensorShapeTrace`, `BeforeAfterSummary`,
  `QuickCheck`.
- **Data:** `weighted-sum.tokenVectors` (already the per-head, concatenated
  output — see §4's "explicitly not a gap" note), `scale-softmax.attention`.
- **Interactive:** live via `tensor-ops.js`'s `attention()` (full
  function, not just a half).
- **Complexity:** Medium.
- **Dependencies:** `MatrixHeatmap.svelte`.
- **Expected reusable abstractions:** none new.

### `heads-compare` (Attention)
- **Viz:** New (`HeadsCompareViz.svelte`) — one `MatrixHeatmap` per head,
  side by side.
- **Shared components reused:** `MatrixHeatmap` (×numHeads instances).
- **Data:** `scale-softmax.attention[h]` per head — already present, no
  generation work (§4).
- **Complexity:** Low (mostly layout, infra fully paid for by this point).
- **Dependencies:** `MatrixHeatmap.svelte`.
- **Expected reusable abstractions:** none new.

### `concat` (Attention)
- **Viz:** New (`ConcatViz.svelte`) — visual stitching of per-head
  outputs back into one row.
- **Shared components reused:** `VectorHeatmap` (per-head input strips +
  concatenated output strip).
- **Data:** derived via `tensor-ops.js`'s `concatHeads()`; final result
  already equals `weighted-sum.tokenVectors` (verified — the Python
  script's `attn_concat` *is* what's recorded as `weighted-sum`), useful
  as a client-side correctness check.
- **Complexity:** Low.
- **Dependencies:** none blocking.
- **Expected reusable abstractions:** none new.

### `output-proj` (Attention)
- **Viz:** New (`OutputProjectionViz.svelte`) — same shape as
  `proj-q/k/v`'s pattern, single weight matrix `Wo`.
- **Shared components reused:** `VectorHeatmap`, `TensorShapeTrace`.
- **Data:** `output-proj.tokenVectors` (present) + `weights.Wo/bo`
  (**gap — see §4, Gap 1**).
- **Complexity:** Low (pattern fully established by `proj-q/k/v`).
- **Dependencies:** Gap 1 (Wo/bo export) must land first.
- **Expected reusable abstractions:** likely reuses whatever
  `LinearProjectionViz` shape emerged from `proj-q/k/v`, if that
  extraction happened (Rule of Three).

### `residual-1` (Attention)
- **Viz:** New, small (`ResidualAddViz.svelte`) — elementwise add,
  reusable pattern for `residual-2` too (2nd instance, not yet 3 — author
  scene-owned, watch for a third residual-add scene before extracting).
- **Shared components reused:** `VectorHeatmap`.
- **Data:** `residual-1.tokenVectors`, present.
- **Complexity:** Low.
- **Dependencies:** none.

### `layer-norm-1` (Attention)
- **Viz:** New, small (`LayerNormViz.svelte`) — same reuse note as
  residual (2nd instance with `layer-norm-2`).
- **Shared components reused:** `VectorHeatmap`.
- **Data:** `layer-norm-1.tokenVectors`, present. `tensor-ops.js`'s
  `layerNorm()` already implements the exact math for Interactive Mode.
- **Complexity:** Low.
- **Dependencies:** none.

### `ffn` (FeedFwd)
- **Viz:** New (`FfnViz.svelte`) — expand → ReLU → project-back, 3
  sub-steps already declared in the registry.
- **Shared components reused:** `VectorHeatmap`, `TensorShapeTrace`.
- **Data:** `ffn.tokenVectors` (present) + `weights.W_ff1/b_ff1/W_ff2/b_ff2`
  (**gap — see §4, Gap 1**).
- **Interactive:** live via `tensor-ops.js`'s `matmul` + `relu`.
- **Complexity:** Medium (3 sub-steps to choreograph).
- **Dependencies:** Gap 1 (FFN weights export).

### `residual-2` (FeedFwd)
- Same shape as `residual-1`. This is the **third** residual-add
  instance's sibling — after this one ships, if `ResidualAddViz` wasn't
  already extracted, this is the trigger point (Rule of Three: 2 concrete
  instances by now, still under the bar — extraction happens only if a
  genuine third *residual* scene appears, which the registry doesn't
  currently have beyond these two; **do not extract preemptively**).
- **Complexity:** Low. **Dependencies:** none.

### `layer-norm-2` (FeedFwd)
- Same shape/reuse note as `layer-norm-1`.
- **Complexity:** Low. **Dependencies:** none.

### `encoder-output` (FeedFwd)
- **Viz:** New (`EncoderOutputViz.svelte`) — final recap: full encoder
  pipeline shape confirmation, natural place for a richer
  `TokenJourneyPanel` moment since every stage now exists (this is the
  natural point to close Gap 3).
- **Shared components reused:** `VectorHeatmap`, `TensorShapeTrace`,
  `BeforeAfterSummary` (whole-pipeline before/after), `QuickCheck`.
- **Data:** `encoder-output.tokenVectors`, present.
- **Complexity:** Medium (ties the whole pipeline together narratively).
- **Dependencies:** ideally built last, or near-last, since it benefits
  from every other stage existing.

### `input-sentence`, `tokenize` (Setup — earliest in registry, simplest, listed last here only for narrative grouping)
- **Viz:** New, small viz components each (`InputSentenceViz.svelte`,
  `TokenizeViz.svelte`).
- **Data:** `meta.sentence` / `tokenize` stage, both already present.
- **Complexity:** Low, low.
- **Dependencies:** none — genuinely the simplest scenes in the registry,
  and the ones a presenter sees first, so despite low complexity they
  matter for first impressions.

---

# 6. Shared Component Roadmap

Only one new shared component is justified by verified evidence. Everything
else stays scene-owned until a real third instance appears, per Rule of
Three.

### `MatrixHeatmap.svelte` — the one justified exception
- **Purpose:** render a `[seq, seq]` (or one head-slice of
  `[heads, seq, seq]`) matrix as a colored grid, using the same
  value→color scale as `VectorHeatmap` (via `core/heatmap-color.js`).
- **Consumers:** `qk-matmul`, `scale-softmax`, `heads-compare` — three
  scenes, back-to-back in the registry, all needing the identical data
  shape (verified: `attentionScores` and `attention` are both already
  `[heads, seq, seq]` in `forward-pass.json`).
- **Whether Rule of Three justifies extraction:** **Yes, as a documented
  exception, not a violation.** Rule of Three exists to prevent guessing
  an abstraction's shape from a single data point. Here, the shape isn't
  a guess — `core/heatmap-color.js` was *already* written and merged with
  this exact component named as its future consumer, and the JSON data
  format that all three consumers will read is already fixed and
  identical across all three. Building it scene-owned inside `qk-matmul`
  first and copy-pasting it into `scale-softmax` days later would
  contradict the very reasoning `heatmap-color.js`'s own extraction
  already used ("same reasoning, extended once a second real consumer
  appeared" — here we can already see the second *and* third consumer
  before writing the first line). Build it once, when `qk-matmul` is
  implemented, directly as a shared component.
- **Files affected:** new `src/components/MatrixHeatmap.svelte`; consumed
  by the three viz files above; no changes needed to
  `core/heatmap-color.js` (already correctly shaped for this).

### Everything else — explicitly deferred, not rejected
- **`LinearProjectionViz` (proj-q/k/v):** 3 near-simultaneous, likely
  near-identical instances — track during authoring; extract only if
  `proj-v` confirms the pattern holds with no real per-role differences.
- **`ResidualAddViz` / `LayerNormViz`:** only 2 concrete instances exist
  in the current 21-scene registry (`residual-1`/`residual-2`,
  `layer-norm-1`/`layer-norm-2`). **Do not extract** — 2 is not 3, and
  there is no verified third instance coming in this registry.

---

# 7. Risk Register

Only risks with a concrete, verified basis in the repository.

| Risk | Category | Basis |
|---|---|---|
| `Wo`/`W_ff1`/`W_ff2` weights are not yet exported to JSON | Data | Verified directly (§4 Gap 1) — blocks `output-proj` and `ffn` from having real weight numbers to render, same failure mode Q/K/V already hit once. |
| `MatrixHeatmap.svelte` doesn't exist yet | Visualization | Verified by direct search (§4 Gap 2) — three consecutive scenes are blocked until it's built. |
| No live-browser verification has ever occurred | Visualization / Maintainability | Stated explicitly and repeatedly by the prior author in doc §11 ("As a frontend engineer... hasn't been eyeballed on an actual screen") — every scene shipped so far is "correct by Svelte's documented semantics and a clean build," not visually confirmed. This compounds with every new scene until someone runs `npm run dev` and actually watches a full sub-step sequence. |
| `TokenJourneyPanel`'s render body is a placeholder | Visualization | Verified directly (§4 Gap 3) — not a blocker, but a known incompleteness that will look unfinished in any demo that opens the panel before `encoder-output` ships. |
| Interactive Mode vocabulary is 12 words, not the ~150–300 originally scoped | Interactive Mode | Verified directly, and honestly self-labeled in `vocab.js`'s own comment (§4 Gap 4) — low severity, but real: Interactive Mode will feel thin on vocabulary for any scene relying on varied sentence construction (Attention scenes especially, where different sentence structures showcase different attention patterns). |
| `attention()` in `tensor-ops.js` is implemented but wired into zero scenes so far | Interactive Mode | Verified directly — the function is correct and complete, but every Attention-group scene's Interactive Mode support is unbuilt. Not a design risk (the function is right), but a real amount of remaining wiring work concentrated in one area of the registry. |
| Touch-device tooltips rely on the native `title` attribute | Accessibility | Stated directly in doc §11 ("documented, not fixed, in this pass") and not addressed by the infra merge — every `VectorHeatmap`/future `MatrixHeatmap` cell's per-value tooltip is inaccessible to touch users. Carries forward into every remaining scene using these components unless addressed once, centrally. |
| Registry scene count mismatch between `architecture-addendum.md` (23) and the actual registry (21) | Maintainability | Verified directly (§2.1) — low risk on its own (doesn't affect any running code), but a real source of confusion for anyone still reading the addendum as current; this document is the correction. |

**Explicitly not risks** (would be speculative to list): WebGL/canvas
performance at scale (already rejected, app doesn't target that scale by
design), a missing orchestration framework (already rejected, and the
existing `sceneStore.js` two-phase advance logic has handled every
multi-substep scene correctly so far, including PE's 6 sub-steps).

---

# 8. Recommended Development Order

Optimized for maximum reuse, minimum merge conflicts, maximum testability,
minimum rework — **not** the same as registry/lecture order.

1. **Close Gap 1** (export `Wo/bo/W_ff1/b_ff1/W_ff2/b_ff2` to
   `forward-pass.json` via `generate_forward_pass.py`). Tiny, isolated,
   touches only the data-generation script and its JSON output — zero
   collision surface with any scene work, and unblocks two scenes later
   in the order. Do this first, alone.
2. **`input-sentence`, `tokenize`, `residual-stream`** — the three
   lowest-complexity remaining Setup scenes, no new shared infrastructure
   needed, no dependency on Gap 2. Good early wins that exercise the
   already-frozen `SceneShell`/registry/nav path end-to-end again (3rd,
   4th, 5th scenes through the pipeline — this is exactly the kind of
   repetition that would surface a `SceneShell` regression early, the
   same discipline the PE build used against Embedding).
3. **`proj-q`** — first Attention scene. Author scene-owned first (Rule
   of Three not yet met for `LinearProjectionViz`). Establishes the QKV
   color convention in practice, not just in `app.css` comments.
4. **`proj-k`, `proj-v`** — second and third instances of the same
   pattern. After `proj-v`, make the one-time call: extract
   `LinearProjectionViz` or keep all three scene-owned, based on whether
   real per-role differences appeared (not before).
5. **`split-heads`** — no new infra, low complexity, natural next step
   after Q/K/V exist; also a good place to first exercise
   `tensor-ops.js`'s `splitHeads()` against real data, ahead of the
   higher-stakes `attention()` call.
6. **Build `MatrixHeatmap.svelte`** as part of implementing **`qk-matmul`**
   (§6) — the one justified shared-component build. Do this as its own
   reviewable unit before or alongside the `qk-matmul` viz, since two more
   scenes depend on it immediately next.
7. **`scale-softmax`** — immediately after, while `MatrixHeatmap` context
   is fresh; near-zero new infra cost.
8. **`heads-compare`** — same reason, completes the `MatrixHeatmap`
   three-consumer set while it's fresh, and is the cheapest of the three
   once the component exists.
9. **`weighted-sum`** — first scene to exercise `tensor-ops.js`'s full
   `attention()` function end-to-end for Interactive Mode; natural to do
   right after the three `MatrixHeatmap` scenes since the data
   (`scale-softmax.attention`) is already front-of-mind.
10. **`concat`** — low complexity, cheap after `weighted-sum` establishes
    the per-head-to-concatenated relationship visually.
11. **`output-proj`** — depends on Gap 1 (already closed in step 1) and
    likely reuses whatever `LinearProjectionViz` shape emerged in step 4.
12. **`residual-1`, `layer-norm-1`** — low complexity, no blockers; do
    these together since they're adjacent in the registry and share a
    "small, quick recap-style scene" authoring rhythm.
13. **`ffn`** — depends on Gap 1 (closed in step 1); medium complexity,
    3 sub-steps.
14. **`residual-2`, `layer-norm-2`** — same shape as step 12's pair; this
    is the natural point to make the final Rule-of-Three call on whether
    a shared `ResidualAddViz`/`LayerNormViz` is justified (verified answer
    per §6: no, only 2 instances exist — keep scene-owned).
15. **`encoder-output`** — last, deliberately: benefits most from every
    prior stage existing, and is the natural point to finally close Gap 3
    (`TokenJourneyPanel`'s real render body), since by this point every
    stage the panel walks through actually has a implemented scene behind
    it.
16. **Gap 4** (vocabulary expansion) — no fixed slot; batch in whenever
    convenient, ideally once several Attention scenes have Interactive
    Mode wired (step 9 onward), so richer vocabulary has scenes worth
    showing it off in.

This order front-loads the one real shared-component build (step 6) at
the earliest point it's justified, clusters its three consumers
immediately after so the component's design gets pressure-tested fast
(same discipline as the original Embedding→PE two-scene proof), and
defers the lowest-urgency gap (vocabulary) to whenever there's slack.

---

# 9. Final Checklist

Before considering the project production-ready:

- [ ] All 21 registry scenes have a `SCENE_VIZ` entry (`viz/index.js`) and
      a full `scene-copy.js` entry (fourQuestions, body×3 levels,
      deepDive, whyPanel, beforeAfter, quickCheck, codeLines/pytorch,
      equationTerms, syncMap, narration, speakerNotes).
- [ ] `npm run build` passes clean after every scene addition — not just
      at the end (per the established discipline: PE's own build was
      verified before writing new content on top of the Embedding
      generalization; keep doing this per scene, not just per phase).
- [ ] Gap 1 closed: `weights.Wo/bo/W_ff1/b_ff1/W_ff2/b_ff2` present in
      both `public/data/forward-pass.json` and `src/data/forward-pass.json`
      (kept identical — verify with `cmp`, as the merge did).
- [ ] `MatrixHeatmap.svelte` built and consumed identically by
      `qk-matmul`, `scale-softmax`, `heads-compare` (no per-scene
      forked copies).
- [ ] Every new scene's hover-highlightable elements have `tabindex="0"`,
      `role="button"`, and matching `on:focus`/`on:blur` (accessibility
      checklist item, not optional).
- [ ] Every new scene's transition `duration`/`delay` values are wrapped
      in `motionMs()`.
- [ ] Every new scene ends in a `before-after` + `quick-check` sub-step
      pair, with `QuickCheck`'s `transition` field teasing the next scene
      by name.
- [ ] Interactive Mode is wired for every scene where it's meaningful
      (all Attention/FeedFwd scenes, using `tensor-ops.js`) — with the
      same "toy model, not learned, disclosed in-UI" framing where
      applicable (weights are real but fixed/untrained), matching the
      precedent set by Embedding/PE.
- [ ] `TokenJourneyPanel`'s render body closed out (Gap 3) once
      `encoder-output` ships — real per-stage heatmap cells, not the
      `[length]` placeholder.
- [ ] `vocab.js` expanded (Gap 4) — not blocking, but verify it happened
      before calling the project done, since it's the one explicitly
      self-labeled "not yet" item in the whole repository.
- [ ] A real browser session (`npm run dev`) has actually been used to
      watch every scene's full sub-step sequence at least once — closing
      the single biggest verified gap between "should work" and
      "verified" that has applied to every scene shipped so far.
- [ ] Touch-device tooltip accessibility (native `title` attribute
      limitation) addressed once, centrally — ideally as part of
      `MatrixHeatmap.svelte`'s build, so the fix covers both vector and
      matrix cells at once instead of needing a second pass later.
- [ ] This document's scene count (21) and registry order are still
      accurate at ship time — if `scene-registry.js` changes, this
      document must be updated in the same change, not left stale the
      way `architecture-addendum.md`'s "23 scenes" claim was left.