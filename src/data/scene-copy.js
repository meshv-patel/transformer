// Content model — see docs/reference-implementation.md for the full
// rationale. `embedding` is fully authored as the Phase 3 reference;
// every field here is the contract future scenes fill in.

export const SCENE_COPY = {
  embedding: {
    eyebrow: 'Encoder · Input',
    title: 'Word Embedding',

    // Four-question framework: viz, equation, code, and narration all
    // answer these same four questions from different angles.
    fourQuestions: {
      whatIsHappening: 'Each token ID is used to look up a row in a learned table, producing a dense vector.',
      why: 'Discrete IDs carry no notion of similarity — ID 4 isn\u2019t "more" or "less" than ID 9. A vector space lets the model place related words near each other.',
      whatChanged: 'Shape goes from [seq_len] (integers) to [seq_len, d_model] (dense vectors) — one lookup, no computation.',
      whatToObserve: 'Watch each token pull its own row out of the same shared table — every occurrence of a word gets the identical vector.',
    },

    body: {
      beginner:
        'Computers only understand numbers, not words. So the very first thing we do is give each word a list of numbers — a "fingerprint" — that captures a bit of its meaning. Words used in similar ways end up with similar fingerprints.',
      mtech:
        'Each token id indexes into a learned embedding matrix E \u2208 R^(vocab_size \u00d7 d_model), producing a dense vector \u2014 think of it as a hash lookup where the values themselves are learned rather than fixed. Unlike one-hot encodings, these vectors are learned jointly with the rest of the model, so geometric proximity in this space correlates with distributional similarity.',
      research:
        'Embedding lookup is a differentiable table lookup \u2014 gradient-wise equivalent to a one-hot vector times E. Ties to distributional semantics (Firth, "you shall know a word by the company it keeps") and to the original Transformer\u2019s weight-tying of the input embedding and output softmax projection (Vaswani et al., 2017, \u00a73.4).',
    },

    deepDive: {
      math: 'x_i = E[token_id_i],  E \u2208 R^{|V| \u00d7 d_model}',
      complexity: 'O(1) lookup per token \u2014 no matmul, just indexing.',
      matrixEquivalence:
        'x = e \u00b7 E, where e is a one-hot row vector (length |V|) with a 1 at the token\u2019s index. Mathematically, the lookup you just watched IS a matrix-vector multiply \u2014 it\u2019s just implemented as an index instead of a real matmul, because multiplying by a one-hot vector only ever picks out one row. This is why the operation is O(1) despite being "secretly" a matmul on paper \u2014 and it\u2019s the same pattern (project via a weight matrix) you\u2019ll see again for Q, K, and V.',
      misconceptions: [
        'Embeddings are not fixed/hand-designed \u2014 they are learned parameters, updated by backprop like any weight.',
      ],
      notes: 'd_model here is fixed at 16 for legibility; production models use 512\u201312288+.',
    },

    whyPanel: {
      items: [
        {
          title: 'Why are embeddings learned, not hand-designed?',
          body: 'No one could manually assign 16+ meaningful numbers to every word in a language. Instead, the vectors start random and are shaped entirely by gradient descent, driven by whatever task the model is trained on \u2014 like tuning a radio by feedback ("clearer... clearer... that\u2019s it") rather than pre-printing a frequency chart.',
        },
        {
          title: 'Why not just use one-hot vectors?',
          body: 'A one-hot vector is |vocab|-dimensional with a single 1 \u2014 for a 50,000-word vocabulary, that\u2019s 50,000 numbers to represent one word, and every pair of distinct words is equally "far apart." It\u2019s like labeling every book in a library only by its shelf slot number \u2014 technically unique, but the number tells you nothing about genre, author, or subject.',
        },
        {
          title: 'Why can\u2019t we compare token IDs directly?',
          body: 'IDs are arbitrary indices into a lookup list. If "cat" = 4 and "dog" = 6, that says nothing about their relationship \u2014 ID 5 might be "umbrella." Arithmetic on IDs is as meaningless as averaging two people\u2019s locker numbers and expecting to land on a locker between their actual lockers.',
        },
        {
          title: 'Why do continuous vector spaces help optimization?',
          body: 'Gradient descent needs smooth, differentiable surfaces to move along. A discrete ID has no gradient; a dense vector does \u2014 small nudges during training can smoothly pull related words closer together, like adjusting a dimmer switch rather than flipping a light on or off.',
        },
        {
          title: 'How does semantic similarity emerge?',
          body: 'Words that appear in similar contexts receive similar gradient updates during training, so their vectors drift toward each other in the space. Nobody tells the model "cat" and "dog" are related \u2014 it\u2019s a bit like people who keep showing up at the same social gatherings ending up seated near each other at a big wedding, without anyone assigning seats by friendship.',
        },
      ],
      example: {
        left: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        right: [0.21, -0.34, 0.08, 0.41, -0.12, 0.29, -0.05, 0.18, -0.27, 0.11, 0.03, -0.19],
        caption: 'One-hot (12-word toy vocab) vs. a dense vector of the same width — the dense version packs information into every dimension.',
      },
    },

    beforeAfter: {
      before: { label: 'Token IDs', shape: null }, // shape filled in from live seqLen at render time
      after: { label: 'Embedding Vectors', shape: null },
      whatChanged:
        'Each discrete token is now represented by a dense, continuous vector that carries learned information \u2014 nothing about meaning yet from context or position, just "which word is this."',
      structured: {
        entered: 'A list of integer token IDs — arbitrary vocabulary indices.',
        happened: 'One table lookup per token — no computation, just retrieval.',
        changed: 'Discrete labels became dense vectors carrying real, learned information.',
        leaves: 'One vector per token — same word anywhere in the sentence gets an identical vector, for now.',
      },
    },

    quickCheck: {
      question: 'What information is still missing after embeddings?',
      choices: ['Position', 'Attention', 'Vocabulary', 'Output'],
      correctIndex: 0,
      explanation:
        'Right \u2014 embeddings only encode which word each token is. Concretely: swap two words\u2019 positions in the sentence and each word still gets the exact same vector it had before \u2014 the embedding layer has no way to tell "cat sat" from "sat cat." So the model has no idea about word order yet.',
      transition: 'Now that every token has meaning, how do we tell the model where each token sits in the sentence? That\u2019s Positional Encoding, next.',
      distractorNotes: {
        1: 'Good instinct \u2014 attention is the famous mechanism \u2014 but attention itself needs to know token positions to be useful. Position has to come first.',
        2: 'Vocabulary is already baked in: it\u2019s literally the set of rows in the embedding table we just used.',
        3: 'Output comes at the very end of the whole network \u2014 we\u2019re still near the beginning, right after the input representation.',
      },
    },

    pytorch: [{ id: 'code-embedding-line', code: 'x = self.token_embedding(input_ids)  # [seq_len] -> [seq_len, d_model]' }],
    equationTerms: [{ id: 'eq-lookup', tex: 'x_i = E[token_id_i]' }],
    syncMap: [
      { vizElementId: 'embedding-lookup-arrow', equationTermId: 'eq-lookup', codeLineId: 'code-embedding-line' },
    ],

    // Per-sub-step narration assistant (index-aligned with scene-registry's
    // subSteps: lookup-table, vector-materialize, before-after, quick-check).
    narration: [
      {
        duration: '~90s',
        objective: 'Establish that raw token IDs carry no meaning, and that lookup is a real, simple mechanism.',
        script:
          'Okay — right now, all we have is a row of integers. Token IDs. And here\u2019s the question worth pausing on: do these numbers mean anything? ...No. Not really. "Cat" being ID 3 and "sat" being ID 4 — that\u2019s just where they happened to fall in the vocabulary list. It\u2019s bookkeeping, not meaning.\n\nSo what do we actually need? Something that carries real information about each word. That\u2019s what this table is. One row per vocabulary word, and every row is a vector — a list of learned numbers. Watch each token reach in and pull out its own row.',
        audienceQuestion: 'Why can\u2019t we just feed the token IDs straight into the network?',
        expectedAnswer: 'Because the IDs are arbitrary index numbers — there\u2019s no meaningful scale or relationship between them.',
        misconception: 'Students sometimes assume embeddings encode dictionary definitions. Push back gently: they encode usage patterns learned from data — nobody hand-wrote what any of these numbers mean.',
        transition: 'Alright — let\u2019s see what comes out the other side of that lookup.',
      },
      {
        duration: '~75s',
        objective: 'Make the shape change concrete: integers in, dense vectors out.',
        script:
          'There it is. Each token now has a dense vector — sixteen numbers, in our case, though real models use hundreds or thousands. Notice: every single dimension has some value. Nothing\u2019s wasted the way it would be with a one-hot encoding, where you\u2019d have thousands of zeros and one lonely 1.\n\nAnd this isn\u2019t a mockup — hover any cell and you\u2019ll see the actual number, from an actual (tiny) forward pass.',
        audienceQuestion: 'If two words basically never show up in similar contexts during training, what do you expect about the distance between their vectors?',
        expectedAnswer: 'They\u2019d probably end up far apart — nothing in training ever pulled their representations toward each other.',
        misconception: 'Some students assume "bigger vector = more meaning." Worth clarifying: dimensionality is a design trade-off, not a direct measure of quality.',
        transition: 'Let\u2019s zoom out for a second and pin down exactly what just changed.',
      },
      {
        duration: '~45s',
        objective: 'Consolidate the transformation as a single before/after fact students can hold onto.',
        script:
          'So, in one line: we walked in with a list of integers — shape [4]. We walk out with a list of dense vectors — shape [4, 16]. Same number of tokens. But now each one is actually carrying something.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Quick gut-check before we move on — don\u2019t skip this, it sets up the whole next scene.',
      },
      {
        duration: '~60s',
        objective: 'Surface the one thing embeddings deliberately do NOT solve yet, motivating the next scene.',
        script:
          'Here\u2019s the question. Give the room a real second before you reveal it — this one\u2019s worth the silence.',
        audienceQuestion: 'What information is still missing after embeddings?',
        expectedAnswer: 'Position — word order isn\u2019t represented yet.',
        misconception: 'A lot of students reach for "attention" here, since it\u2019s the famous piece. Redirect: attention needs positional information to even make sense of what it\u2019s attending to — so position has to come first.',
        transition: 'Exactly — position. Every token knows *what* it is now, but not *where* it is. That\u2019s next.',
      },
    ],

    speakerNotes: {
      teachingTips: [
        'Ask students to guess the embedding dimension before revealing it \u2014 most guess far higher than 16, good moment to explain why we shrink it for the lecture.',
      ],
      misconceptions: [
        'Students often think embeddings encode dictionary definitions. Clarify: they encode usage patterns learned from data, nothing hand-authored.',
      ],
      suggestedQuestions: [
        'If two words never appear in similar contexts in training, what would you expect about the distance between their embeddings?',
      ],
    },
  },

  'positional-enc': {
    eyebrow: 'Encoder · Input',
    title: 'Positional Encoding',

    fourQuestions: {
      whatIsHappening: 'A unique vector is added to each token\u2019s embedding, one per position in the sequence.',
      why: 'Attention and feed-forward layers have no built-in sense of order \u2014 without this, the model would process a sentence as an unordered bag of words.',
      whatChanged: 'The shape doesn\u2019t change ([T, D] stays [T, D]) \u2014 but every vector\u2019s actual values now depend on where the token sits, not just what it is.',
      whatToObserve: 'Watch the same word produce identical vectors regardless of position \u2014 then watch that stop being true the moment position is added.',
    },

    body: {
      beginner:
        'Right now, every "cat" in a sentence looks exactly the same to the model, no matter where it sits. That\u2019s a problem \u2014 "the dog bit the man" and "the man bit the dog" would look identical. So before anything else happens, we give each position in the sentence its own unique pattern of numbers, and add it on top of every token\u2019s embedding. Now position 1 and position 3 leave a different fingerprint, even on the same word.',
      mtech:
        'Self-attention and feed-forward layers are permutation-equivariant \u2014 reorder the input tokens and you get the same set of outputs, just reordered. There\u2019s no recurrence and no convolution to give the model an implicit sense of sequence, so position has to be injected explicitly. Positional encoding adds a position-dependent vector PE(pos) to each embedding, elementwise, keeping the shape at [T, d_model].',
      research:
        'The original Transformer (Vaswani et al., 2017, \u00a73.5) uses fixed sinusoidal functions of position rather than learned position embeddings, specifically so the model can extrapolate to sequence lengths not seen during training \u2014 sin/cos are defined for any real-valued position, unlike a learned lookup table capped at some max length. Later architectures (BERT, GPT-2) largely moved to learned absolute position embeddings; more recent ones (e.g. RoPE, ALiBi) encode relative position directly into the attention computation instead. This scene focuses on the original sinusoidal formulation as the clearest place to build the intuition.',
    },

    deepDive: {
      math: 'PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))',
      complexity: 'O(1) per position \u2014 the table is precomputed once (it doesn\u2019t depend on the input at all, only on sequence length and d_model) and simply added.',
      matrixEquivalence:
        'x\u2032 = x + PE, an elementwise vector addition \u2014 not a matmul, not a concatenation. Because it\u2019s addition, the shape is unchanged: [T, d_model] in, [T, d_model] out. Concatenating position onto embedding instead would work shape-wise too, but doubles every downstream weight matrix\u2019s input dimension for no proven benefit \u2014 addition is what the paper (and nearly everything since) actually uses.',
      misconceptions: [
        'Positional encoding is NOT learned in the original Transformer \u2014 it\u2019s a fixed mathematical formula, computed once from sequence length and d_model alone, with no parameters and no gradient flowing through it. (Contrast with the embedding table, which is entirely learned.)',
        'It isn\u2019t one number added per token \u2014 it\u2019s a full d_model-length vector, where each dimension oscillates at a different frequency.',
      ],
      notes:
        'Why sine AND cosine, not just sine: together they let a shift in position be expressed as a simple rotation (a linear function) of the encoding at any other position \u2014 useful for the model to learn "attend N positions away" patterns. This is a research-level detail; the diagrams above are enough to build the core intuition without it.',
    },

    whyPanel: {
      items: [
        {
          title: 'Why isn\u2019t word order already inside the embeddings?',
          body: 'The embedding lookup is a pure function of word identity \u2014 E["cat"] returns the same row no matter which sentence, or which position in that sentence, "cat" appears in. There\u2019s no "occurrence number" input to the lookup at all.',
        },
        {
          title: 'Why can\u2019t the model just infer order automatically?',
          body: 'An RNN processes tokens one at a time in order, so sequence is built into its structure. A Transformer processes all tokens in parallel, with no such structure \u2014 feed it the same set of token vectors in a different order and, without positional encoding, you get back the same set of outputs, just reordered. Order has to be added as information, because nothing about the architecture assumes it.',
        },
        {
          title: 'Why sine and cosine specifically?',
          body: 'They\u2019re smooth, bounded between \u22121 and 1, and \u2014 critically \u2014 defined for any position, including ones far longer than anything seen in training. A learned table of position vectors would simply have no entry for position 10,001 if it only ever saw sequences up to 10,000 tokens; sin/cos just keep going.',
        },
        {
          title: 'Why addition instead of concatenation?',
          body: 'Concatenation would work, but doubles the dimension the rest of the network has to handle at every layer, for no measured benefit. Addition keeps the shape constant and, in practice, the network learns to use the space efficiently \u2014 it turns out meaning and position don\u2019t need strictly separate "regions" of the vector to both be usable.',
        },
        {
          title: 'Why do nearby positions end up with similar encodings?',
          body: 'Think of a clock: the hour hand barely moves between 2:00 and 2:05, while the second hand sweeps all the way around. Positional encoding has "hands" moving at many different speeds (frequencies) \u2014 the slow ones barely change between adjacent positions, so nearby positions share most of their encoding and differ mainly in the fast-moving dimensions.',
        },
      ],
      example: {
        left: [0, 0.84, 0.91, 0.14, -0.76, -0.96, -0.28, 0.66, 0.99, 0.41, -0.54, -1],
        right: [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
        leftLabel: 'Dimension 0 (fast)',
        rightLabel: 'Dimension 8 (slow)',
        leftStyle: 'continuous',
        rightStyle: 'continuous',
        leftNote: 'swings through its full range in a handful of positions',
        rightNote: 'barely moves across the same 12 positions — the "hour hand"',
        caption: 'Two real dimensions of the same position vectors, across 12 consecutive positions — one oscillates fast, one crawls. That\u2019s the "clock hands," made visible.',
      },
    },

    beforeAfter: {
      before: { label: 'Embeddings', shape: null },
      after: { label: 'Position-Aware Embeddings', shape: null },
      whatChanged:
        'Every token\u2019s vector now depends on both what the word is and where it sits \u2014 the shape hasn\u2019t changed at all, only the values have.',
      structured: {
        entered: 'Embedding vectors — meaning, no position information.',
        happened: 'A unique positional vector, added elementwise, one per position.',
        changed: 'Values now encode meaning AND position — dimensionality is untouched.',
        leaves: 'Position-aware embeddings, still shape [T, D] — identical shape to what entered.',
      },
    },

    quickCheck: {
      question: 'If every token in the sentence received the exact same positional vector, would word order still be represented?',
      choices: [
        'Order would NOT be distinguishable — same problem as before',
        'Order would still be distinguishable, just shifted',
        'The model would automatically learn to ignore position',
        'The embeddings would become one-hot again',
      ],
      correctIndex: 0,
      explanation:
        'Right \u2014 if the same constant vector were added to every token, every token would shift by an identical amount. The differences between tokens\u2019 vectors would be exactly what they were before adding anything, so the model would be right back to being unable to tell "cat sat down" from "down sat cat." The whole point is that each position gets a distinct vector.',
      transition: 'Now every token knows both what it is and where it is — but nothing has looked at any other token yet. How does the model decide which words should pay attention to each other? That\u2019s Self-Attention, next.',
      distractorNotes: {
        1: 'Adding the same vector to everything is just a constant shift \u2014 it moves every token\u2019s vector by the same amount, so the relative differences between tokens (which is what downstream layers actually see) are completely unchanged.',
        2: 'The model has no mechanism to "notice and discard" a broken signal like this \u2014 it just computes with whatever vectors it\u2019s given.',
        3: 'Adding a constant vector doesn\u2019t undo the embedding lookup or push values back toward one-hot \u2014 the vectors stay dense, just uniformly shifted.',
      },
    },

    pytorch: [
      { id: 'code-pe-formula', code: 'pe = positional_encoding(seq_len, d_model)  # fixed, not learned' },
      { id: 'code-pe-add', code: 'x = x + pe  # elementwise add — shape unchanged: [T, D]' },
    ],
    equationTerms: [
      { id: 'eq-pe-formula', tex: 'PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(\u2026)' },
      { id: 'eq-pe-add', tex: 'x\u2032 = x + PE(pos)' },
    ],
    syncMap: [
      { vizElementId: 'pe-table-row', equationTermId: 'eq-pe-formula', codeLineId: 'code-pe-formula' },
      { vizElementId: 'combine-result-row', equationTermId: 'eq-pe-add', codeLineId: 'code-pe-add' },
    ],

    // Sub-steps: problem, table, combine, output, before-after, quick-check
    narration: [
      {
        duration: '~80s',
        objective: 'Prove, experientially, that embeddings alone can\u2019t distinguish word order.',
        script:
          'Quick question before we go further \u2014 look at these embedding vectors. Can the model tell which word came first? ...Watch this. I\u2019m going to swap two words in the sentence. There \u2014 order\u2019s different. Now look at the vectors again. Same numbers. Every single one. The word "cat" produced the exact same vector whether it was first or third \u2014 because the embedding lookup only ever asks "which word is this," never "where is it."',
        audienceQuestion: 'Why didn\u2019t swapping the words change any of the vectors?',
        expectedAnswer: 'Because embedding lookup is a function of word identity only — it has no input for position at all.',
        misconception: 'Some students assume the model "obviously" reads left to right the way we do. Push back: the network has no built-in reading direction — everything so far has been an independent, per-token operation.',
        transition: 'So we have a real problem. Let\u2019s look at the fix.',
      },
      {
        duration: '~70s',
        objective: 'Introduce the positional encoding table as a real, inspectable structure — not a black box.',
        script:
          'Here\u2019s the positional encoding table — one row per position in the sequence, and every row is a unique pattern. Look at the stripes: some columns barely change from row to row, others flip rapidly. That\u2019s not random — those are sine and cosine waves at different frequencies, and together they give every position a genuinely unique fingerprint.',
        audienceQuestion: 'Why not just use a single number — like 0, 1, 2, 3 — for position instead of a whole vector?',
        expectedAnswer: 'A single scalar has no obvious way to combine with a 16-dimensional embedding, and small integer differences (3 vs 4) would be tiny compared to the embedding values — a full vector, added elementwise, integrates naturally and lets many frequencies carry the signal.',
        misconception: 'Students sometimes think this table is learned like the embedding table. It isn\u2019t — it\u2019s computed once from a formula, with zero trainable parameters.',
        transition: 'Let\u2019s actually combine a position vector with an embedding and watch it happen.',
      },
      {
        duration: '~85s',
        objective: 'Make the addition mechanism concrete, cell by cell, for one token, then generalize.',
        script:
          'Here\u2019s one embedding, and here\u2019s its matching position row. Watch — we add them, dimension by dimension, and out comes a new vector. Notice what did NOT happen: no new dimensions appeared. Sixteen numbers went in from each side, sixteen numbers came out. Now let\u2019s do this for every token in the sentence at once.',
        audienceQuestion: 'If we concatenated instead of added, what would happen to the width of this row?',
        expectedAnswer: 'It would double — 16 embedding dimensions plus 16 position dimensions would become 32, instead of staying at 16.',
        misconception: 'Watch for students trying to track "which part of the result is meaning and which is position" — there isn\u2019t a clean split; addition blends them into the same 16 numbers.',
        transition: 'Let\u2019s look at what we\u2019re left with for the whole sentence.',
      },
      {
        duration: '~40s',
        objective: 'Confirm the shape explicitly — this is the fact students should write down.',
        script:
          'And that\u2019s the result: every token now has a position-aware vector. Same count of tokens, same sixteen dimensions each — shape [4, 16], exactly what we started with. Nothing about the shape gave away that anything happened at all — you\u2019d only know by comparing the actual numbers.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Let\u2019s lock in exactly what changed before we check ourselves.',
      },
      {
        duration: '~35s',
        objective: 'Consolidate before/after as a single fact.',
        script:
          'In one line: we walked in with vectors that knew meaning only. We walk out with vectors that know meaning AND position — same shape, [4, 16], the whole way through.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'One more check before we move on — this one matters for what comes next.',
      },
      {
        duration: '~55s',
        objective: 'Test the core mechanism (uniqueness per position), not a shape-memorization fact, and open the door to attention.',
        script:
          'Here\u2019s the question — give it real thought before revealing.',
        audienceQuestion: 'If every token got the exact same positional vector, would order still be represented?',
        expectedAnswer: 'No — a constant shift changes nothing about the relative differences between tokens, so we\u2019d be right back to the original problem.',
        misconception: 'Some students think "adding anything positional" automatically fixes the problem, regardless of whether it varies by position. Push back using the swap demo from the start of the scene as the counter-example.',
        transition: 'Exactly. Every token now knows what it is and where it is — but nothing has compared it to any other token yet. How does the model decide which words to pay attention to? That\u2019s next.',
      },
    ],

    speakerNotes: {
      teachingTips: [
        'The swap demonstration only works if students are actually looking at numbers, not just trusting you — consider pausing on the swapped vectors long enough for someone to visually confirm two rows are identical.',
      ],
      misconceptions: [
        'The most persistent one: assuming positional encoding is learned, the same way embeddings are. It\u2019s worth stating twice — once here, once again if it resurfaces during Attention.',
      ],
      suggestedQuestions: [
        'Why does the paper use both sine AND cosine, instead of just sine?',
      ],
    },
  },
};

export function getSceneCopy(id) {
  return (
    SCENE_COPY[id] ?? {
      eyebrow: '',
      title: id,
      fourQuestions: null,
      body: { beginner: '', mtech: '', research: '' },
      deepDive: null,
      whyPanel: null,
      beforeAfter: null,
      quickCheck: null,
      pytorch: null,
      equationTerms: [],
      codeLines: [],
      syncMap: [],
      narration: [],
      speakerNotes: null,
    }
  );
}
