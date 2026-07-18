# Embedding Scene — Reference Implementation Guide

This is the implementation guide for every remaining Transformer scene.
Nothing here is Embedding-specific by design — the systems documented below
(Camera Director, narration, four-question framing, crossfade morph,
before/after + quick-check beats) are generic and already reusable as-is.

---

## 1. Why each system exists

| System | Problem it solves |
|---|---|
| **Camera Director** (`core/camera/`, `CameraStage.svelte`) | Without it, every element on screen competes for attention equally. The camera tells students *where to look*, automatically, the way a documentary editor frames a shot. |
| **Crossfade morph** (`svelte/transition`'s `crossfade`) | A hard cut between "table row" and "materialized vector" reads as two different things. A shared-element morph reads as *the same data, transformed* — which is the actual lesson. |
| **Four-question framing** (`fourQuestions` in scene-copy) | Prevents scenes from becoming "here's what this operation is called." Every scene must answer what/why/changed/observe, and the viz, equation, code, and narration all point at the same four answers from different angles. |
| **Narration system** (`narration[]` per sub-step) | A presenter script that isn't just "notes to self" — it has timing, a concrete objective, a question to ask, the expected answer, a misconception to preempt, and the exact transition line into what's next. This is what makes a scene deliverable as a stand-alone 5–7 minute lecture segment. |
| **Why panel** (persistent, separate from Deep Dive) | Deep Dive answers "what's the math." Why panel answers "why does this design exist at all" — a different, permanent question, not a footnote. |
| **Before/After beat** | Every scene should end with one sentence + one shape diagram a student could write in their notes verbatim. |
| **Quick Check beat** | Forces retrieval before moving on, and its correct answer is written to *be* the hook into the next scene — the quiz doubles as the transition. |

---

## 2. Camera Director

**Files:** `src/core/camera/cameraStore.js`, `src/components/CameraStage.svelte`

The camera is a single spring-animated `{x, y, scale}` store. It is **shot-based, not measurement-based**: scenes author a small `SHOTS` array/map indexed by sub-step, not by measuring DOM rects at runtime. This was a deliberate choice — authored shots are predictable across screen sizes, projector aspect ratios, and window resizes, where live-geometry approaches are fragile and can't be verified without a real browser in the loop.

**Reuse contract for every future scene:**
```js
import { focusShot } from '../../core/camera/cameraStore.js';

const SHOTS = [
  { x: 0, y: 20, scale: 1.05 },  // sub-step 0
  { x: 0, y: -10, scale: 1.1 }, // sub-step 1
  // ...one entry per sub-step; { x:0, y:0, scale:1 } = wide/reset shot
];

$: {
  $replayTick; // include as a dependency so Replay re-triggers the camera move too
  focusShot(SHOTS[$subStepIndex] ?? SHOTS[0]);
}
```
Wrap the scene's whole visual in `<CameraStage>...</CameraStage>` — nothing else is required. `CameraStage` also renders the background vignette (dims the periphery automatically whenever `scale > 1.02`), so "focus" and "dim everything else" are one coherent effect, not two systems to keep in sync by hand.

The spring is intentionally slightly underdamped (`stiffness: 0.07, damping: 0.55`) — this alone produces natural anticipation/follow-through (a small overshoot before settling) without any hand-authored easing curve. `prefers-reduced-motion` collapses the spring to an instant snap.

**Framing guidance for future scenes:** a "wide" shot (`{x:0,y:0,scale:1}`) is correct whenever a sub-step needs the whole stage visible (summary/quiz beats). A "push" shot (`scale` 1.05–1.15, small `y` offset) is correct when one region of the stage is doing the active computation — don't exceed ~1.15 scale, past that the periphery clips awkwardly on narrow viewports.

---

## 3. Crossfade morph (the "living lookup table")

**File:** `src/scenes/viz/EmbeddingViz.svelte`

`svelte/transition`'s `crossfade()` factory produces a matched `[send, receive]` pair. Any two elements — even in entirely different conditional blocks — that use the same transition `key` will morph into each other (position + size interpolation) instead of independently fading. This is what makes the highlighted table row visually *become* the materialized vector strip rather than one disappearing while an unrelated one appears.

**Reuse pattern:**
```js
const [send, receive] = crossfade({
  duration: (d) => Math.sqrt(d * 250),
  easing: cubicOut,
  fallback(node) { return { duration: 260, css: (t) => `opacity: ${t}` }; },
});
```
Apply `in:receive={{ key }}` / `out:send={{ key }}` to both the "before" element and the "after" element, using the **same key** (we use `'tok-' + tokenIndex`, since ordering is guaranteed stable — see `usedRows` alignment note in the file). Apply both directives to both elements (not just one each) so the morph works symmetrically whether the student is moving forward or backward (Prev).

This only works for elements that are genuinely "the same thing, transformed" — don't reach for crossfade between unrelated elements; it'll produce a confusing, semantically-wrong morph.

---

## 4. Content model (per scene, in `src/data/scene-copy.js`)

Every field below is now part of the contract `getSceneCopy()` guarantees exists (falling back to empty/null for unauthored scenes):

```js
{
  eyebrow, title,
  fourQuestions: { whatIsHappening, why, whatChanged, whatToObserve },
  body: { beginner, mtech, research },
  deepDive: { math, complexity, misconceptions, notes },
  whyPanel: { items: [{ title, body }], example: { oneHot, dense, caption } },
  beforeAfter: { before: {label, shape}, after: {label, shape}, whatChanged },
  quickCheck: { question, choices, correctIndex, explanation, transition },
  pytorch, syncMap,
  narration: [ { duration, objective, script, audienceQuestion, expectedAnswer, misconception, transition } ], // index-aligned with subSteps
  speakerNotes: { teachingTips, misconceptions, suggestedQuestions }, // scene-level, in addition to per-substep narration
}
```

**`beforeAfter.before.shape` / `after.shape` are `null` in the copy file on purpose** — Embedding's shapes depend on the live sentence length and `d_model`, so `EmbeddingViz.svelte` fills them in at render time from live state rather than the static copy. Scenes with fixed shapes can just author them directly in `scene-copy.js`.

---

## 5. Sub-step pattern: every scene should end in the same two beats

Embedding's `subSteps` in `scene-registry.js`:
```js
['lookup-table', 'vector-materialize', 'before-after', 'quick-check']
```
The first N sub-steps are scene-specific (the actual computation, staged). The **last two are structural and every future scene should include them**: a `before-after` beat (using `<BeforeAfterSummary>`) and a `quick-check` beat (using `<QuickCheck>`), gated the same way:
```svelte
{#if $subStepIndex === lastIndex - 1}
  <BeforeAfterSummary before={...} after={...} whatChanged={copy.beforeAfter.whatChanged} />
{/if}
{#if $subStepIndex === lastIndex}
  <QuickCheck {...copy.quickCheck} />
{/if}
```
`QuickCheck`'s `transition` field should always tease the *next* scene by name — that's what makes the quiz double as the connective tissue between scenes rather than a dead end.

---

## 6. Data flow (unchanged from Phase 2/3, still the source of truth)

`Lecture Mode` reads `$forwardPassData` (loaded once from `public/data/forward-pass.json`, generated offline by `tools/generate_forward_pass.py` — real numpy computation, not placeholder data). `Interactive Mode` computes live via `core/tensor-ops.js` or, for Embedding specifically, `core/embedding-utils.js`'s seeded generator (disclosed in-UI as illustrative, not trained — see Phase 2 architecture addendum §1 for why a toy model can't do free-text semantics credibly).

Every scene should branch on `$dataMode` the same way Embedding does:
```js
$: activeThing = $dataMode === 'lecture' ? realDataFrom($forwardPassData) : liveComputeFrom(interactiveState);
```

---

## 7. Remounting strategy — don't over-remount

A subtle but important pattern: **the token row is keyed separately from the rest of the stage** (`sentenceKey` vs `stageKey`). The token row only remounts when the sentence, data mode, or replay tick actually changes — *not* on every sub-step advance. This is why moving between sub-steps reads as "the camera moved to a new part of an unchanging scene" rather than "everything just flickered." When building a future scene, identify which elements are genuinely stable across its sub-steps and keep them outside the per-sub-step `{#key}` block.

---

## 8. Accessibility & performance baseline

- All interactive elements are real `<button>`s or have `role`/`tabindex`/keyboard handlers (see the vector-strip compare interaction for the pattern when a `<div>` must be clickable).
- `prefers-reduced-motion` disables the camera spring's motion (snaps instantly) and the CSS glow/pulse keyframe animations — implemented once in `CameraStage.svelte` and the scene's own `<style>` block; future scenes should keep the same media query on any custom keyframe animation they add.
- Animations use `transform`/`opacity` exclusively (no animated `width`/`height`/`top`/`left` outside the crossfade library internals), which keeps everything on the GPU-accelerated compositing path.
- Hover targets are real pointer targets (buttons/cells), not synthetic zones, so touch works without special-casing.

---

## 9. What to copy verbatim vs. what to author fresh, per future scene

**Copy verbatim (infrastructure):** `CameraStage`, camera store usage pattern, crossfade pattern, `TensorShapeTrace`, `BeforeAfterSummary`, `QuickCheck`, `WhyPanel`, the `SpeakerNotes`/narration panel, the `sentenceKey`/`stageKey` remount split, the `$dataMode` branch pattern.

## 10. Final polish pass — additions for future scenes to also copy

- **Two-stage camera moves.** For any sub-step where a "search, then land"
  narrative applies (ours: scanning the table, then focusing on the match),
  don't use one static shot — fire a wide intro shot immediately, then a
  `setTimeout`-delayed push shot ~500ms later. One line of extra code, and
  it's the difference between the camera "cutting" and "looking." Sub-steps
  that are pure recap/quiz should get **no** camera motion at all — camera
  disappearing entirely is itself a cue ("we're done moving, just think").
- **GPU-only glow via a halo layer, not `box-shadow` keyframes.** `box-shadow`
  animation is a paint operation, not a compositor operation — looping it
  indefinitely (our first pass did this) is wasted GPU work for a live
  lecture that might sit on one screen for minutes. The fix: a separate
  absolutely-positioned `.glow-halo` sibling that animates only `opacity`
  and `transform` once (`animation-iteration-count: 1`), staggered per
  element via inline `animation-delay`. This is the `.glow-wrap`/`.glow-halo`
  pattern in `EmbeddingViz.svelte` — copy it verbatim for any future
  "this just became the important thing" moment instead of reaching for
  `box-shadow` again.
- **Sequential, not simultaneous, entrance.** Tokens fly in, *then* arrows
  appear, *then* the table appears, *then* (per-row, staggered) the matched
  rows ignite — four visually distinct beats instead of one overlapping
  blob of motion. When choreographing a new scene, write out the beats in
  prose first ("A happens, then B, then C") and only then assign delay
  numbers — it's much easier to keep the story straight that way than to
  tune numbers until something looks plausible.
- **Curiosity hooks are text, not just narration.** The spoken script always
  had a transition line; now the *screen* also shows one
  ("Every token has meaning now — but does the model know which one came
  first?") right before the recap beats. Students who are reading, not just
  listening, should also feel the same pull toward the next scene.
- **Keyboard parity for every hover-triggered highlight.** Any element with
  `on:mouseenter`/`on:mouseleave` for the sync-highlight bus must also have
  `tabindex="0"`, `role="button"`, and matching `on:focus`/`on:blur` —
  otherwise the equation/code/viz sync (a core requirement) silently
  doesn't exist for keyboard-only or switch-device users. This is a
  checklist item, not a judgment call — apply it every time.
- **Distractor-specific quiz rebuttals.** `QuickCheck`'s `distractorNotes`
  prop lets a specific wrong answer get a specific, reasoning-oriented
  rebuttal instead of everyone seeing the same generic explanation. Author
  one per plausible wrong choice, not just for the correct one.
- **Structured Before/After over prose.** `BeforeAfterSummary`'s `structured`
  prop (`entered`/`happened`/`changed`/`leaves`) is preferred over the
  original single `whatChanged` paragraph — four short scannable lines beat
  one paragraph for a students'-notes-verbatim summary.

---

## 11. Honest self-critique (as of this pass)

Written from four angles, kept unresolved where genuinely unresolved rather
than papered over:

**As an AI professor:** the narration is a single fixed script — a real
professor adapts wording live based on the room's reaction. There's no
mechanism for "the audience seems confused, try a different phrasing," and
there's exactly one quiz question, not a bank to draw from across repeated
lectures.

**As a UX designer:** the Token Journey panel (Phase 2 infrastructure) is
still mostly a placeholder — clicking a token opens it, but it doesn't yet
animate the token traveling through stages, because most stages don't exist
yet. It will only become genuinely good once Phase 4 populates more scenes.
Per-dimension cell tooltips rely on the native `title` attribute, which
doesn't work well on touch devices — touch users lose that specific
interaction (documented, not fixed, in this pass).

**As an educational psychologist:** the Quick Check only tests recognition
among four options, not production or transfer — a stronger check might ask
a student to predict the *shape* of the next transformation before revealing
it. The narration's "give the room a moment to think" instruction is a good
impulse but has no enforced pause in the UI itself (a presenter could still
click through instantly).

**As a frontend engineer:** I have no live browser in this environment to
visually confirm the crossfade morph, camera easing, or animation timing
actually read the way they're described — everything here is correct by
Svelte's documented transition/spring semantics and passes a clean, warning-
free build, but hasn't been eyeballed on an actual screen. That's the
single biggest gap between "should work" and "verified." Before presenting
this live, run `npm run dev` and watch the full sub-step sequence at least
once.

---

## 12. External review response

An external architecture review was received and evaluated critically rather
than implemented wholesale. Full classification lives in the conversation
record; summary of what changed as a result:

**Adopted:** a shared `VectorHeatmap.svelte` component (removes the one
piece of *actually observed* duplication between the table-row and
vector-strip rendering — the reviewer's broader "God Component" framing was
overstated, but this specific extraction was real and cheap); named
`STEP` constants replacing magic-number `$subStepIndex` comparisons; a
one-hot × Wₑ matrix-multiplication framing added to Deep Dive (legitimate
MTech-rigor gap — bridges "lookup" intuition to the linear-projection
pattern every later scene reuses); concrete (not just asserted)
position-agnosticism language in the Quick Check; responsive breakpoints for
the fixed-position chrome (`ExploreNav` becomes a horizontal bottom bar
under 900px instead of a full-height sidebar that would otherwise cover the
viz — note this surfaced a real CSS cascade-order bug during implementation,
now fixed); a `core/motion.js` helper so `prefers-reduced-motion` actually
reaches Svelte's JS-driven `fly`/`fade`/`crossfade` transitions, not just
hand-rolled CSS keyframes; lightweight structural validation on
`forward-pass.json` load; `aria-live` on the Quick Check reveal.

**Explicitly rejected:** a WebGL/canvas rendering migration (solves a
`d_model=768` scale problem this app deliberately doesn't support, and would
regress the accessibility work already done — canvas content isn't
screen-reader-navigable without a parallel shadow DOM); a full
`StateMachine.js`/`Director.js` orchestration framework (premature
abstraction from a sample size of one implemented scene); DOM-rect-measured
camera targeting (reverses the Phase 3 shot-authored decision, made for
reasons that still hold — no live browser here to validate measurement-based
framing across breakpoints); an in-browser tensor-execution runtime
(`core/tensor-ops.js` already does this, at the appropriate scale, with no
dependency). Several of the review's cited "hidden bugs" (naked `{#each}`
keys, an unguarded camera race condition, raw-SVG rendering) were verified
against the actual code and don't exist here — they describe a generic or
hypothetical Svelte anti-pattern, not this codebase.

**Flagged for later, not acted on:** the review's point about attention
scenes needing to visually link *distant* token pairs (not just
neighboring ones) is a legitimate forward-planning concern. The camera and
crossfade patterns established here extend naturally to that (a ribbon/line
connector between two arbitrary DOM positions is a smaller addition than a
rendering-engine change), but there's nothing to build yet — worth
remembering when Phase 4 reaches the Attention scene specifically.

---

## 13. Positional Encoding — what changed for the second scene

Two shared files needed real (not speculative) generalization once a
*second* scene existed to prove what the right shape was — this is the
payoff of waiting for genuine duplication instead of guessing at an
abstraction from one data point (see §12's rejection of the premature
`StateMachine`/`Director` framework).

- **`SceneShell.svelte`** hardcoded Embedding's specific highlight IDs
  (`'code-embedding-line'`, `'eq-lookup'`) directly in its render logic.
  Positional Encoding needs *two* synced code/equation pairs (the PE
  formula, and the addition), not one. Fixed by having `SceneShell` read
  `copy.pytorch` / `copy.equationTerms` as arrays, falling back to
  auto-wrapping a legacy single string for any scene that hasn't been
  updated. Embedding's own copy was migrated to the explicit array form
  with its *exact same* IDs, so its behavior is byte-for-byte unchanged —
  verified by rebuilding before writing any Positional Encoding content.
- **`WhyPanel.svelte`**'s example comparison was hardcoded to "One-hot" vs
  "Dense" labels and a binary-cell rendering style specific to Embedding.
  Positional Encoding's Why-panel example (a fast-oscillating dimension vs.
  a slow one, both continuous) needed different labels and a different
  cell-rendering style. Generalized with `leftLabel`/`rightLabel`/
  `leftNote`/`rightNote`/`leftStyle`/`rightStyle` props, all defaulting to
  Embedding's exact original strings and binary/continuous split — again,
  zero behavior change for Embedding, verified by rebuild.

**What was reused completely unchanged:** `CameraStage`/`cameraStore.js`
(new authored `SHOTS` array only — no new camera code), the crossfade
morph technique (new `key` scheme, same `crossfade()` call shape), the
`glow-wrap`/`glow-halo` one-shot ignite pattern, `VectorHeatmap` (used
directly, no changes needed), `TensorShapeTrace`, `BeforeAfterSummary`,
`QuickCheck` (gained `distractorNotes` in the previous round, used as-is
here), `core/motion.js`'s `motionMs()` wrapped through every transition
call site, and the `STEP` named-constant pattern.

**New, scene-owned, not shared (per Rule of Three — only two scenes exist,
nothing has repeated three times yet):** the swap-demonstration UI (opening
beat), the position-table markup, the embedding+position combine-row
markup. If a third scene needs near-identical versions of any of these,
*that's* the signal to extract them — not before.

### Data flow note

Both new required datasets already existed. `tools/generate_forward_pass.py`
already computed and exported the `positional-enc` stage (summed vectors)
and its raw `pe` matrix in Phase 3 — nothing needed regenerating. For
Interactive Mode, `core/positional-encoding.js` implements the exact same
formula in JS, verified numerically identical to the Python output (see
implementation notes). This is a meaningful contrast with Embedding:
Positional Encoding needs no "not trained, illustrative only" disclosure in
Interactive Mode, because — unlike embeddings — it isn't learned in the
first place. The scene says this explicitly to students.

### Self-review

**As an AI professor:** the swap demonstration is the strongest part of
this scene — it makes the problem *observable* rather than asserted, which
is exactly what Embedding's opening lacked (Embedding had to assume the
"IDs are meaningless" problem was already accepted). The narration explicitly
tells the presenter to pause and let someone visually confirm two rows
match before revealing anything, which is the right instinct. Weakness:
there's no equivalent "let's verify sin/cos actually gives every position a
unique fingerprint" moment for the table step — a student could accept the
table on faith without ever checking two rows are actually different.

**As an educational psychologist:** the Quick Check ties directly back to
the opening swap demo ("same problem as before"), which is good — it's
testing the same mental model twice, from different angles, which is real
reinforcement rather than a new fact to memorize. The Deep Dive's sine/cosine
rotation-identity note is correctly demoted to research-tier and skippable;
good layering discipline. Weakness: the "why isn't order already in the
embedding" Why-panel item and the opening swap demo make almost the same
point twice, back to back — a stronger scene might have the Why panel go
somewhere the swap demo didn't (e.g., dig into *why* extrapolation to unseen
lengths matters practically) rather than partially re-explaining the setup.

**As a frontend engineer:** the crossfade morph (table row → combine operand)
is the same well-tested Svelte primitive Embedding used, applied to a new
key scheme — low risk. The two generalizations to `SceneShell`/`WhyPanel`
were verified by rebuilding *before* writing new content, specifically to
catch a regression early rather than discover it at the end. Same caveat as
every prior round: no live browser here, so the crossfade timing and camera
buildup for the COMBINE step are correct by Svelte's documented semantics
and a clean build, not eyeballed.

**As a UX designer:** reusing `VectorHeatmap` for both word-rows and
position-rows keeps the visual language consistent with Embedding — a
returning student doesn't have to learn a new visual grammar. The swap
button is a deliberate manual gate (not auto-triggered) so a presenter
controls the pause-and-ask beat — matches the narration's explicit
instruction to give the room time before revealing. Weakness: the COMBINE
step's per-token row (embedding + position = result, times up to 6 tokens)
gets visually dense fast in Interactive Mode with more tokens; it's
readable at Embedding's default lecture length (4 tokens) but is the first
place in the app where "more tokens" starts to cost real vertical space —
worth watching if Interactive Mode's token cap is ever raised.



