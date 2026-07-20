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
  'proj-q': {
    eyebrow: 'Attention · Projection',
    title: 'Linear Projection to Q',
    fourQuestions: {
      whatIsHappening: 'We multiply the position-aware token vectors by a learned Query weight matrix, creating a new vector space for Queries.',
      why: 'Linear projection allows the model to learn what representation (e.g. Query, Key, Value) each token should assume when attending to other tokens.',
      whatChanged: 'Each token\'s [d_model] vector is projected to a [d_model] Query vector (shape remains [seq_len, d_model]).',
      whatToObserve: 'Observe how each token vector is multiplied by the weight matrix row-by-row to yield the corresponding Query vector.',
    },
    body: {
      beginner: 'To figure out how words relate to each other, each word must ask questions about the other words. We multiply our word vectors by a special "Query" key matrix to produce these questions (Queries).',
      mtech: 'Each position-aware vector x_pe is projected into the query space via Q = x_pe * W_q + b_q, where W_q is a learned parameter matrix of shape [d_model, d_model]. This projects the input into a space optimized for query-matching.',
      research: 'Linear projections construct the query vectors. Because W_q is parameterized and optimized during training, it projects tokens into sub-spaces that emphasize query features. Output dimension is d_model, split later into multiple attention heads.',
    },
    deepDive: {
      math: 'Q = X_{pe} W_q + b_q',
      complexity: 'O(seq_len × d_model^2) operations.',
      matrixEquivalence: 'Each token\'s Query vector is computed independently as a vector-matrix multiply. We multiply a [1, d_model] token vector by a [d_model, d_model] weight matrix to produce a [1, d_model] Query vector.',
      misconceptions: [
        'Queries are not computed from text labels; they are continuous activations derived entirely from the token embeddings and positional encodings.',
      ],
      notes: 'd_model = 16 for this visualization, yielding a 16x16 weight matrix.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why do we need a separate Query matrix?',
          body: 'A single vector cannot easily represent a word\'s meaning, its position, the questions it wants to ask, and the keys it answers all at the same time. The Query weight matrix lets the model extract just the "question-asking" features of each word.',
        },
        {
          title: 'Why is it called a "Query"?',
          body: 'Think of it like a search engine: the Query represents the search term ("what I am looking for"). In self-attention, each word queries all other words to find out which ones it should pay attention to.',
        }
      ],
      example: {
        left: [0.1, -0.2, 0.4, 0.3, -0.1, 0.2, 0.05, -0.15, 0.25, 0.1, -0.05, 0.12, 0.0, -0.1, 0.3, -0.2],
        right: [0.35, 0.12, -0.05, -0.2, 0.4, 0.05, -0.1, 0.15, -0.3, 0.2, 0.12, -0.05, 0.18, 0.25, -0.12, 0.05],
        caption: 'Input token vector vs. projected Query vector — the Query projection focuses on search features.'
      }
    },
    beforeAfter: {
      before: { label: 'Input Vectors', shape: null },
      after: { label: 'Query Vectors', shape: null },
      whatChanged: 'We projected the position-aware token vectors into the Query vector space.',
      structured: {
        entered: 'Position-aware token vectors X_pe of shape [seq_len, d_model].',
        happened: 'Matrix multiplication with learned Query weight matrix W_q.',
        changed: 'Vectors are rotated/scaled into the query feature space.',
        leaves: 'Query vectors Q of shape [seq_len, d_model], ready to match against Keys.',
      }
    },
    quickCheck: {
      question: 'What is the purpose of projecting a token vector to a Query vector?',
      choices: [
        'To prepare it to ask questions about other tokens in the attention step.',
        'To merge it with the positional encoding information.',
        'To reduce its dimensionality to save compute.',
        'To look up its semantic meaning in the vocabulary table.'
      ],
      correctIndex: 0,
      explanation: 'Correct! The Query vector represents "what this token is looking for" in other tokens. It will be dot-multiplied with Key vectors from other tokens to determine attention weights.',
      transition: 'Next, we project the tokens to Keys (K) so they can be matched against these Queries.',
      distractorNotes: {
        1: 'Incorrect. Positional encoding was already added in the previous scene.',
        2: 'Incorrect. Projection to Q maintains the same d_model dimensionality (16) here.',
        3: 'Incorrect. Vocabulary lookup is done during the Word Embedding scene.'
      }
    },
    pytorch: [
      { id: 'code-proj-q', code: 'Q = self.W_q(x)  # [seq_len, d_model] -> [seq_len, d_model]' }
    ],
    equationTerms: [
      { id: 'eq-proj-q', tex: 'Q = X_{pe} W_q' }
    ],
    syncMap: [
      { vizElementId: 'viz-matmul', equationTermId: 'eq-proj-q', codeLineId: 'code-proj-q' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Introduce the concept of a weight matrix for projection.',
        script: 'To find out how words relate, each word needs to ask questions of the others. We do this by projecting our vectors into a query space. Here is the query weight matrix W_q. It is a sixteen by sixteen table of learned parameters, initialized randomly and tuned during training.',
        audienceQuestion: 'Why is the weight matrix square at sixteen by sixteen?',
        expectedAnswer: 'Because both our input dimension and query output dimension are d_model, which is sixteen.',
        misconception: 'Students often think the weight matrix values are fixed; remind them these are learned parameters updated via backpropagation.',
        transition: 'Now, let\'s multiply our input vectors by this matrix to see how queries materialize.'
      },
      {
        duration: '~45s',
        objective: 'Demonstrate matrix multiplication element-by-element.',
        script: 'By multiplying our input vectors by the query weight matrix, we project them. Notice how each token\'s vector is dot-multiplied against the columns of the weight matrix. Each resulting cell represents a projection of that token\'s features onto a specific query dimension.',
        audienceQuestion: 'Does the calculation of one token\'s Query depend on the other tokens?',
        expectedAnswer: 'No, each token\'s projection is completely independent of the others at this stage.',
        misconception: 'Some think self-attention mixing happens here. Clarify that this is a purely token-local operation; mixing only happens in the attention score step.',
        transition: 'Let\'s summarize what this operation accomplished.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Draw an analogy to database queries: the Query is the search query, the Key is the index key, and the Value is the actual record content.'
      ],
      misconceptions: [
        'Emphasize that the projection is a purely linear transformation (no activation function is applied here, unlike in the Feed Forward Network).'
      ],
      suggestedQuestions: [
        'What would happen if we didn\'t use a projection matrix and instead matched the inputs directly?'
      ]
    }
  },
  'qk-matmul': {
    eyebrow: 'Attention · Matching',
    title: 'Q × Kᵀ (Scaled Attention Scores)',
    fourQuestions: {
      whatIsHappening: 'We multiply the Query matrix Q by the transpose of the Key matrix K, scaling the result by 1/sqrt(d_k).',
      why: 'This compares every word\'s query vector against every other word\'s key vector to calculate raw, pairwise relevance scores.',
      whatChanged: 'Two [seq_len, d_model] matrices are multiplied and scaled to produce a [heads, seq_len, seq_len] attention score tensor.',
      whatToObserve: 'Hover over cells to see how the Query of a row token matches the Key of a column token — values represent search relevance.',
    },
    body: {
      beginner: 'To figure out which words relate, each word matches its "Query" (the question it asks) against all other words\' "Keys" (the topics they cover). High scores mean a strong connection.',
      mtech: 'The raw attention scores are computed as S = (Q * K^T) / sqrt(d_k). This performs pairwise dot products between all query vectors and key vectors. The scaling factor of 1/sqrt(d_k) prevents the dot products from growing excessively large for high dimensions, which would push the softmax gradients to near-zero.',
      research: 'Scaled dot-product attention computes similarity between Query and Key representations. Transposing K enables parallel matmul. Scaling by 1/sqrt(d_k) is mathematically crucial: assuming elements of q and k are independent random variables with mean 0 and variance 1, their dot product has mean 0 and variance d_k. Scaling restores variance to 1, ensuring stable softmax updates.',
    },
    deepDive: {
      math: 'S = \\frac{Q K^T}{\\sqrt{d_k}}',
      complexity: 'O(seq_len^2 × d_model) operations per attention head.',
      matrixEquivalence: 'The scores matrix represents all pairwise dot products. Cell (r, c) represents the dot product of the r-th Query vector and the c-th Key vector, divided by the square root of d_k.',
      misconceptions: [
        'These scores are not probabilities yet — they can be negative or greater than 1. They are normalized into weights by the Softmax step next.',
      ],
      notes: 'd_k = d_model / num_heads = 16 / 4 = 4. The scaling factor is 1/sqrt(4) = 0.5.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why multiply Q by K transpose?',
          body: 'We want to match every Query row with every Key column. Transposing K flips its shape from [seq_len, d_k] to [d_k, seq_len], allowing a single standard matrix multiplication to compute all combinations at once.',
        },
        {
          title: 'Why divide by the square root of d_k?',
          body: 'In higher dimensions, dot products tend to grow very large. This makes the softmax function peak, putting almost all weight on a single item and causing vanishing gradients during training. Scaling keeps the variance stable.',
        }
      ],
      example: {
        left: [0.15, -0.05, 0.45, -0.3],
        right: [1.2, -0.4, 2.5, 0.85],
        caption: 'Unscaled dot product values can exceed 2.0; scaling divider pushes them back to stable ranges.'
      }
    },
    beforeAfter: {
      before: { label: 'Query & Key Vectors', shape: null },
      after: { label: 'Attention Scores', shape: null },
      whatChanged: 'We computed the pairwise relevance between every token\'s Query and Key representation.',
      structured: {
        entered: 'Query (Q) and Key (K) matrices of shape [seq_len, d_model] per head.',
        happened: 'Matrix multiplication of Q and transpose of K, divided by square root of d_k.',
        changed: 'Extracted raw similarity scores for all token pairs.',
        leaves: 'An attention score grid of shape [heads, seq_len, seq_len].',
      }
    },
    quickCheck: {
      question: 'Why do we scale the attention scores by 1/sqrt(d_k) before softmax?',
      choices: [
        'To prevent extremely large dot products from causing vanishing gradients.',
        'To compress the vectors to fit in browser memory.',
        'To force the attention scores to sum to exactly 1.0.',
        'To align the token vectors with the positional encodings.'
      ],
      correctIndex: 0,
      explanation: 'Correct! Without scaling, large dot products in high-dimensional spaces push the softmax function into regions with extremely small gradients, slowing down or halting backpropagation training.',
      transition: 'Now that we have raw matching scores, how do we turn them into real percentages? That\'s Softmax, next.',
      distractorNotes: {
        1: 'Incorrect. Browser memory is not affected by this mathematical scaling.',
        2: 'Incorrect. Softmax is what forces the rows to sum to 1.0, not the scaling division.',
        3: 'Incorrect. Positional encoding was already added earlier in the pipeline.'
      }
    },
    pytorch: [
      { id: 'code-qk-matmul', code: 'scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)' }
    ],
    equationTerms: [
      { id: 'eq-qk-matmul', tex: 'S = \\frac{Q K^T}{\\sqrt{d_k}}' }
    ],
    syncMap: [
      { vizElementId: 'viz-matmul', equationTermId: 'eq-qk-matmul', codeLineId: 'code-qk-matmul' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain the Query-Key dot product relevance matching.',
        script: 'Now we match our queries against the keys. We multiply the Query matrix Q by the transpose of the Key matrix K. Each cell in this grid represents the dot product of a specific Query vector and Key vector, scaled down by the square root of d_k.',
        audienceQuestion: 'What does a high value in the grid represent?',
        expectedAnswer: 'It means that the Query of the row token is highly relevant to the Key of the column token.',
        misconception: 'Some think these values are final attention percentages; clarify that they are unnormalized scores and can be negative.',
        transition: 'Let\'s see what we are left with shape-wise.'
      },
      {
        duration: '~25s',
        objective: 'Review the shape change to [seq_len, seq_len].',
        script: 'Our matrices went from d_model dimensions to a square grid of sequence length by sequence length. In this grid, every word is mapped against every other word, including itself. The shape is now heads by four by four.',
        audienceQuestion: 'Why does each head have its own grid?',
        expectedAnswer: 'Because each head represents a different attention subspace, looking for different types of relationships.',
        misconception: 'Explain that the heads are computed completely in parallel.',
        transition: 'Let\'s check our understanding of the scaling factor.'
      },
      {
        duration: '~40s',
        objective: 'Reinforce the mathematical motivation for scaling.',
        script: 'Take a look at the quick check question. Consider what would happen if the vector dimensions got extremely large during training, and how that relates to the softmax curves.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Perfect. Let\'s move on to normalizing these scores into probabilities.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Spend a minute explaining the transpose operation visually: it flips K so rows become columns, which makes the dot products align naturally in standard matrix multiplication.'
      ],
      misconceptions: [
        'Clarify that dividing by sqrt(d_k) is a scaling step to preserve variance, not a normalization step like layer norm.'
      ],
      suggestedQuestions: [
        'How does changing the number of heads impact the scale factor?'
      ]
    }
  },
  'proj-k': {
    eyebrow: 'Attention · Projection',
    title: 'Linear Projection to K',
    fourQuestions: {
      whatIsHappening: 'We multiply the position-aware token vectors by a learned Key weight matrix, creating a new vector space for Keys.',
      why: 'Linear projection allows the model to learn what features each token should contain when answering queries from other tokens.',
      whatChanged: 'Each token\'s [d_model] vector is projected to a [d_model] Key vector (shape remains [seq_len, d_model]).',
      whatToObserve: 'Observe how each token vector is multiplied by the Key weight matrix row-by-row to yield the corresponding Key vector.',
    },
    body: {
      beginner: 'Next, we need to create "Keys" for each word. If a Query represents a question, a Key represents a list of topics that word can speak about. We multiply our vectors by a "Key" weights matrix.',
      mtech: 'Each position-aware vector x_pe is projected into the key space via K = x_pe * W_k + b_k, where W_k is a learned parameter matrix of shape [d_model, d_model]. This extracts features optimized for key-matching.',
      research: 'Linear projections construct the key vectors. Like W_q, W_k is parameterized and optimized during training to map token embeddings into a matching space compatible with Query projections.',
    },
    deepDive: {
      math: 'K = X_{pe} W_k + b_k',
      complexity: 'O(seq_len × d_model^2) operations.',
      matrixEquivalence: 'Each token\'s Key vector is computed independently as a vector-matrix multiply of the input vector by the [d_model, d_model] weight matrix W_k.',
      misconceptions: [
        'Keys are not static; they represent dynamic contextual features that are compared to Queries during the dot-product matching step.',
      ],
      notes: 'd_model = 16 for this visualization, yielding a 16x16 weight matrix W_k.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why do we need a separate Key matrix?',
          body: 'A word should answer questions differently depending on what is being asked. The Key matrix focuses the word\'s representation on its potential to address other words\' Queries.',
        },
        {
          title: 'How does this compare to W_q?',
          body: 'While W_q extracts query features ("what I want to know"), W_k extracts key features ("what I can offer"). Having independent parameters lets the model learn asymmetric relationships.',
        }
      ],
      example: {
        left: [0.1, -0.2, 0.4, 0.3, -0.1, 0.2, 0.05, -0.15, 0.25, 0.1, -0.05, 0.12, 0.0, -0.1, 0.3, -0.2],
        right: [-0.15, 0.3, 0.12, -0.4, 0.05, -0.2, 0.1, -0.05, 0.22, -0.18, 0.05, 0.12, -0.05, 0.25, -0.12, 0.15],
        caption: 'Input token vector vs. projected Key vector — Key projection focuses on indexing features.'
      }
    },
    beforeAfter: {
      before: { label: 'Input Vectors', shape: null },
      after: { label: 'Key Vectors', shape: null },
      whatChanged: 'We projected the position-aware token vectors into the Key vector space.',
      structured: {
        entered: 'Position-aware token vectors X_pe of shape [seq_len, d_model].',
        happened: 'Matrix multiplication with learned Key weight matrix W_k.',
        changed: 'Vectors are rotated/scaled into the key index space.',
        leaves: 'Key vectors K of shape [seq_len, d_model], ready for dot-product matching.',
      }
    },
    quickCheck: {
      question: 'What is the relationship between the Query (Q) and Key (K) representations?',
      choices: [
        'Queries represent search terms; Keys represent searchable indexes.',
        'Queries represent values; Keys represent mathematical signs.',
        'Queries and Keys are mathematically identical vectors.',
        'Queries represent input text; Keys represent output translations.'
      ],
      correctIndex: 0,
      explanation: 'Correct! Think of Queries as search keywords and Keys as labels or indexes. We compute dot products between them to determine which Queries match which Keys.',
      transition: 'Now, we project the tokens to Values (V) so we have actual content to fetch.',
      distractorNotes: {
        1: 'Incorrect. Values are represented by a separate matrix (V) which we project next.',
        2: 'Incorrect. Queries and Keys are projected via separate weight matrices (W_q and W_k) and are not identical.',
        3: 'Incorrect. Both Q and K are intermediate layers inside the Attention mechanism.'
      }
    },
    pytorch: [
      { id: 'code-proj-k', code: 'K = self.W_k(x)  # [seq_len, d_model] -> [seq_len, d_model]' }
    ],
    equationTerms: [
      { id: 'eq-proj-k', tex: 'K = X_{pe} W_k' }
    ],
    syncMap: [
      { vizElementId: 'viz-matmul', equationTermId: 'eq-proj-k', codeLineId: 'code-proj-k' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Introduce the Key weight matrix W_k.',
        script: 'To be searchable by other words, each word must represent its key features. Here is the Key weight matrix W_k, which maps our input vectors into Key vectors.',
        audienceQuestion: 'Why can\'t we reuse W_q for Keys?',
        expectedAnswer: 'Because asking questions and answering them require different semantic focuses. Keeping them separate allows asymmetric attention.',
        misconception: 'Remind students that W_k contains parameters learned during training.',
        transition: 'Let\'s perform the matrix multiplication to compute the Keys.'
      },
      {
        duration: '~45s',
        objective: 'Demonstrate Key vector calculations.',
        script: 'We project the inputs into Key space. Observe the row-by-column multiplications of the positional-encoded tokens with W_k to produce the Key matrix.',
        audienceQuestion: 'Does the Key projection change the shape of our token sequence?',
        expectedAnswer: 'No, the sequence length and dimension remain four by sixteen.',
        misconception: 'Confirm that these are raw keys, not yet matched against queries.',
        transition: 'Let\'s consolidate this step.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Emphasize that the Key projection runs completely in parallel for all tokens.'
      ],
      misconceptions: [
        'Point out that W_k is completely separate from W_q and has different learned weights.'
      ],
      suggestedQuestions: [
        'What would happen if W_k was set to the identity matrix?'
      ]
    }
  },
  'proj-v': {
    eyebrow: 'Attention · Projection',
    title: 'Linear Projection to V',
    fourQuestions: {
      whatIsHappening: 'We multiply the position-aware token vectors by a learned Value weight matrix, creating a new vector space for Values.',
      why: 'Linear projection allows the model to extract the actual content (information) that each token will contribute to other tokens.',
      whatChanged: 'Each token\'s [d_model] vector is projected to a [d_model] Value vector (shape remains [seq_len, d_model]).',
      whatToObserve: 'Observe how each token vector is multiplied by the Value weight matrix row-by-row to yield the corresponding Value vector.',
    },
    body: {
      beginner: 'Finally, we need to create "Values" for each word. If Q is the question and K is the label, V is the actual content or meaning of the word. We multiply our vectors by a "Value" weights matrix.',
      mtech: 'Each position-aware vector x_pe is projected into the value space via V = x_pe * W_v + b_v, where W_v is a learned parameter matrix of shape [d_model, d_model]. This extracts semantic values.',
      research: 'Linear projections construct the value vectors. Like W_q and W_k, W_v is parameterized and optimized during training to extract the content representation that will be aggregated in the attention summation step.',
    },
    deepDive: {
      math: 'V = X_{pe} W_v + b_v',
      complexity: 'O(seq_len × d_model^2) operations.',
      matrixEquivalence: 'Each token\'s Value vector is computed independently as a vector-matrix multiply of the input vector by the [d_model, d_model] weight matrix W_v.',
      misconceptions: [
        'Value vectors are not static vocabulary definitions; they represent contextual representations shaped by training.',
      ],
      notes: 'd_model = 16 for this visualization, yielding a 16x16 weight matrix W_v.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why do we need a separate Value matrix?',
          body: 'A word\'s role in asking questions (Q) or indexing itself (K) is different from the actual content (V) it passes forward. The Value matrix extracts this raw content.',
        },
        {
          title: 'What happens to V after projection?',
          body: 'Values are weighted by the attention scores and summed. If attention scores determine "where to look," V represents "what is actually seen."',
        }
      ],
      example: {
        left: [0.1, -0.2, 0.4, 0.3, -0.1, 0.2, 0.05, -0.15, 0.25, 0.1, -0.05, 0.12, 0.0, -0.1, 0.3, -0.2],
        right: [0.25, -0.12, 0.05, 0.18, -0.3, 0.22, 0.08, -0.15, 0.1, 0.05, -0.25, 0.12, -0.05, 0.35, -0.2, 0.08],
        caption: 'Input token vector vs. projected Value vector — Value projection extracts content features.'
      }
    },
    beforeAfter: {
      before: { label: 'Input Vectors', shape: null },
      after: { label: 'Value Vectors', shape: null },
      whatChanged: 'We projected the position-aware token vectors into the Value vector space.',
      structured: {
        entered: 'Position-aware token vectors X_pe of shape [seq_len, d_model].',
        happened: 'Matrix multiplication with learned Value weight matrix W_v.',
        changed: 'Vectors are rotated/scaled into the value content space.',
        leaves: 'Value vectors V of shape [seq_len, d_model], ready to be aggregated.',
      }
    },
    quickCheck: {
      question: 'What is the role of Value (V) vectors in the attention calculation?',
      choices: [
        'They represent the actual content that is aggregated based on attention weights.',
        'They are used to calculate the matching scores between tokens.',
        'They represent the positional encoding shift applied to token embeddings.',
        'They are used to normalize the softmax probabilities.'
      ],
      correctIndex: 0,
      explanation: 'Correct! While Queries and Keys determine "how much" words should focus on each other, the Value vectors represent the actual content that is extracted and combined based on those weights.',
      transition: 'Now that we have projected Q, K, and V, we split them into multiple heads. Let\'s see that next.',
      distractorNotes: {
        1: 'Incorrect. Queries and Keys calculate the matching scores, not Values.',
        2: 'Incorrect. Positional encoding was added before the projection layer.',
        3: 'Incorrect. Softmax is a mathematical function that does not use Value vectors.'
      }
    },
    pytorch: [
      { id: 'code-proj-v', code: 'V = self.W_v(x)  # [seq_len, d_model] -> [seq_len, d_model]' }
    ],
    equationTerms: [
      { id: 'eq-proj-v', tex: 'V = X_{pe} W_v' }
    ],
    syncMap: [
      { vizElementId: 'viz-matmul', equationTermId: 'eq-proj-v', codeLineId: 'code-proj-v' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Introduce the Value weight matrix W_v.',
        script: 'Finally, we need to extract the actual content that each word will pass forward. We multiply our input vectors by the Value weight matrix W_v.',
        audienceQuestion: 'Why do we need a separate matrix for Values?',
        expectedAnswer: 'Because the content a word transmits should be separate from the features it uses to match queries and keys.',
        misconception: 'Make sure students realize that V contains the raw content to be mixed.',
        transition: 'Let\'s compute the Value vectors.'
      },
      {
        duration: '~45s',
        objective: 'Demonstrate Value vector calculations.',
        script: 'Multiplying by W_v yields the Value vectors. Each cell in the output represents projected content information for that token.',
        audienceQuestion: 'What is the shape of the Value output?',
        expectedAnswer: 'It is still four by sixteen, matching our input shape.',
        misconception: 'Emphasize that the actual mixing of values across words does not happen here.',
        transition: 'Let\'s summarize the Value projection.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Use the post-office analogy: Q is the return address/question, K is the routing index/key, and V is the actual letter contents.'
      ],
      misconceptions: [
        'Highlight that V projections are also independent for each token.'
      ],
      suggestedQuestions: [
        'How would the model change if we omitted W_v?'
      ]
    }
  },
  'scale-softmax': {
    eyebrow: 'Attention · Normalization',
    title: 'Scaling & Softmax',
    fourQuestions: {
      whatIsHappening: 'We divide the raw attention scores by the square root of the head dimension d_k, and apply the softmax function row-wise.',
      why: 'Scaling stabilizes gradients during training; softmax converts scores into positive weights between 0 and 1 that sum to 1.0, representing relevance percentages.',
      whatChanged: 'Unnormalized raw matching scores are converted into normalized attention probabilities (still shape [heads, seq_len, seq_len]).',
      whatToObserve: 'Compare Step 0 (Scaled Scores) with Step 1 (Softmax Weights). Notice how negative numbers disappear and cell values in each row sum to exactly 1.0.',
    },
    body: {
      beginner: 'To turn raw scores into final percentages, we scale them down and apply "Softmax". This ensures that for every word, the attention percentages it gives to other words are all positive and add up to exactly 100%.',
      mtech: 'The scaling step divides scores by sqrt(d_k), where d_k = 4. The softmax operator is defined as A_{i,j} = exp(S_{i,j}) / sum_k(exp(S_{i,k})), normalizing each row into a probability distribution.',
      research: 'Row-wise softmax is applied to the scaled attention scores. Softmax acts as a continuous, differentiable approximation of argmax. Scaling by 1/sqrt(d_k) is critical for training stability: with high values of d_k, dot product magnitudes push softmax inputs into regions with vanishingly small gradients.',
    },
    deepDive: {
      math: 'A = \\text{Softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right)',
      complexity: 'O(seq_len^2) exponentiations and normalization operations.',
      matrixEquivalence: 'The resulting weights represent the relative focus each token places on all other tokens. Summing along the columns for any row yields 1.0.',
      misconceptions: [
        'Softmax is applied independently to each row (token) and each head. It is not a global normalization over the entire matrix.',
      ],
      notes: 'd_k = 4. Dividing by sqrt(4) = 2.0 halves the score magnitudes. Notice the peak scores compress, leading to gentler probability slopes.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why do we need softmax?',
          body: 'We need to convert raw similarity scores into weights so we can compute a weighted average of the Value vectors. Softmax outputs sum to 1.0, which acts as a standard mathematical blending ratio.',
        },
        {
          title: 'What does softmax do to extreme values?',
          body: 'Softmax exponentiates values, which exponentiates differences: large values become very dominant, while small values are suppressed close to zero.',
        }
      ],
      example: {
        left: [2.5, 1.0, -0.5, 0.2],
        right: [0.73, 0.16, 0.04, 0.07],
        caption: 'Raw scaled scores (left) mapped through softmax into weights (right). Exponents accentuate the largest score.'
      }
    },
    beforeAfter: {
      before: { label: 'Scaled Scores S', shape: null },
      after: { label: 'Attention Weights A', shape: null },
      whatChanged: 'We normalized the raw scores into a valid probability distribution where each row sums to 1.0.',
      structured: {
        entered: 'Scaled attention scores of shape [heads, seq_len, seq_len].',
        happened: 'Applied exponentiation and normalized by the row-wise sum.',
        changed: 'All negative scores are eliminated, and values are compressed to [0, 1] range.',
        leaves: 'Attention weights matrix A of shape [heads, seq_len, seq_len].',
      }
    },
    quickCheck: {
      question: 'What is the sum of attention weights across any single row of the softmax output matrix?',
      choices: [
        'Exactly 1.0 (or 100%)',
        'It depends on the sequence length.',
        'It is equal to the scale factor sqrt(d_k).',
        '0.0, because raw scores can be negative.'
      ],
      correctIndex: 0,
      explanation: 'Correct! The softmax function divides the exponent of each score by the sum of exponents for that row, forcing the final weights along any row to sum to exactly 1.0.',
      transition: 'Now that we have the exact weight percentages, we use them to average our Value vectors. Let\'s see that in Weighted Sum.',
      distractorNotes: {
        1: 'Incorrect. The sum of softmax outputs is always 1.0 regardless of how many elements are in the row.',
        2: 'Incorrect. Sqrt(d_k) is the scaling divisor, not the sum of weights.',
        3: 'Incorrect. Exponentiation maps all inputs to positive numbers, so the sum cannot be negative or zero.'
      }
    },
    pytorch: [
      { id: 'code-scale-softmax', code: 'weights = torch.softmax(scores, dim=-1)' }
    ],
    equationTerms: [
      { id: 'eq-scale-softmax', tex: 'A = \\text{Softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right)' }
    ],
    syncMap: [
      { vizElementId: 'viz-matmul', equationTermId: 'eq-scale-softmax', codeLineId: 'code-scale-softmax' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Explain the scaling division.',
        script: 'First, we scale the dot products by dividing them by the square root of the key dimension d_k. This helps control the magnitude of values going into softmax.',
        audienceQuestion: 'Why do we scale before softmax instead of after?',
        expectedAnswer: 'Because if raw scores are too large, the softmax exponents will saturate, leading to near-zero gradients during training.',
        misconception: 'Make sure students notice that the values in this first step do not sum to 1.0 yet.',
        transition: 'Now let\'s apply softmax to normalize these scores.'
      },
      {
        duration: '~35s',
        objective: 'Explain the softmax row-wise probability distribution.',
        script: 'Next, we compute the softmax row-by-row. This exponentiates each score and divides it by the row-wise sum. Notice how all negative scores disappear, and every row now sums to exactly 1.0.',
        audienceQuestion: 'Which token gets the highest attention in this row?',
        expectedAnswer: 'The cell with the brightest orange color, representing the largest percentage.',
        misconception: 'Some think columns sum to 1.0; emphasize that normalization occurs independently along each row.',
        transition: 'Let\'s summarize the transition.'
      },
      {
        duration: '~20s',
        objective: 'Review the shape and values of the softmax weights.',
        script: 'We have converted our unnormalized matching scores into percentages. The shape remains heads by sequence length by sequence length.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Let\'s check our understanding.'
      },
      {
        duration: '~30s',
        objective: 'Reinforce the row-sum constraint.',
        script: 'Answer the quick check question. Consider what properties are required for a set of numbers to act as blending coefficients.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Great. Now we aggregate our values using these weights.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Have students hover over row cells to verify that the numbers printed in the cell box sum to 1.0 for that row.'
      ],
      misconceptions: [
        'Clarify that softmax does not change the shape of the tensor; it only scales and normalizes the scalar cell values.'
      ],
      suggestedQuestions: [
        'What would happen if one score in a row was extremely large compared to all others?'
      ]
    }
  },
  'weighted-sum': {
    eyebrow: 'Attention · Aggregation',
    title: 'Weighted Sum',
    fourQuestions: {
      whatIsHappening: 'We multiply the attention weights matrix A by the Value vectors V to produce the blended weighted sum vectors O.',
      why: 'This aggregates information from all tokens. Each token\'s output vector becomes a weighted blend of all other tokens\' values, according to their attention percentages.',
      whatChanged: 'Attention weights [seq_len, seq_len] and Value vectors [seq_len, d_k] are multiplied to produce weighted outputs [seq_len, d_k] per head, which are concatenated to [seq_len, d_model].',
      whatToObserve: 'Hover over a cell in the left grid. See how the row of attention weights multiplies the column of Value vectors to produce the cell value in the output vector.',
    },
    body: {
      beginner: 'Finally, we use our attention percentages to blend the word meanings together. If word A gives 80% attention to word B, A\'s output will take 80% of its new meaning from B\'s Value vector. This mixes context from the entire sentence.',
      mtech: 'The weighted sum aggregates the value representations V per head: O_h = A_h * V_h, where A_h is the [seq, seq] attention weights matrix and V_h is the [seq, d_k] Value matrix. The head outputs are then concatenated along the channel dimension to form the multi-head representation.',
      research: 'Multi-head aggregation aggregates Value representations. The context vector output for a single head h is computed as O_h = Softmax(Q_h K_h^T / sqrt(d_k)) * V_h. Concatenating the outputs of all h heads yields the final contextualized sequence representation of shape [seq_len, d_model], ready for output projection.',
    },
    deepDive: {
      math: 'O = A V',
      complexity: 'O(seq_len^2 × d_k) operations per attention head.',
      matrixEquivalence: 'The multiplication translates to O_{i,j} = sum_k(A_{i,k} * V_{k,j}). Each output dimension j for token i is a weighted average of that dimension across all other tokens\' Value vectors.',
      misconceptions: [
        'Value vectors (V) are what gets aggregated, not the positional encoded vectors (X_pe) directly. This allows the model to mix contextualized information rather than raw input embeddings.',
      ],
      notes: 'd_k = 4. The output shape per head is [seq_len, 4]. Since there are 4 heads, concatenating them yields a [seq_len, 16] output, which matches the model\'s d_model shape.',
    },
    whyPanel: {
      items: [
        {
          title: 'What does the weighted sum achieve?',
          body: 'This is the step where information actually mixes across tokens. Before this, all projections were purely local to individual tokens. Now, each token\'s vector collects information from other tokens based on their matching relevance scores.',
        },
        {
          title: 'Why concatenate the heads?',
          body: 'Each head represents a different focus space (e.g. tracking subjects, verbs, adjectives). Concatenating their outputs stitches these different relational summaries side-by-side, giving a rich multi-perspective context vector.',
        }
      ],
      example: {
        left: [0.8, 0.1, 0.05, 0.05],
        right: [1.5, -0.4, 3.2, 0.8],
        caption: 'Attention weight row (left) multiplying Value vectors (right) to compute the blended output.'
      }
    },
    beforeAfter: {
      before: { label: 'Attention Weights & Values', shape: null },
      after: { label: 'Weighted Sum Output', shape: null },
      whatChanged: 'We aggregated the semantic content of Value vectors using the normalized attention weights.',
      structured: {
        entered: 'Attention weights matrix A of shape [seq_len, seq_len] and Value matrix V of shape [seq_len, d_k] per head.',
        happened: 'Multiplied the attention weights matrix by the Value matrix for each head, and concatenated the head outputs.',
        changed: 'Value representations are combined and blended across sequence steps.',
        leaves: 'Concatenated weighted sum representations of shape [seq_len, d_model].',
      }
    },
    quickCheck: {
      question: 'What is the primary purpose of the Weighted Sum step in Self-Attention?',
      choices: [
        'To blend the Value vectors of all tokens together based on their attention weights.',
        'To project the inputs to a smaller vocabulary dimension.',
        'To add positional encoding to the value matrices.',
        'To calculate the relevance scores between Queries and Keys.'
      ],
      correctIndex: 0,
      explanation: 'Correct! The Weighted Sum step aggregates the Value vectors from all positions, mixing context across tokens according to their attention weights.',
      transition: 'Now that the heads are blended, we concatenate them and project them back. Let\'s see that in the registry.',
      distractorNotes: {
        1: 'Incorrect. Vocabulary mapping happens at the very end of the Transformer network.',
        2: 'Incorrect. Positional encoding was already added at the input level.',
        3: 'Incorrect. Query-Key matching is what computes the weights; this step applies those weights to Values.'
      }
    },
    pytorch: [
      { id: 'code-weighted-sum', code: 'context = torch.matmul(weights, V)' }
    ],
    equationTerms: [
      { id: 'eq-weighted-sum', tex: 'O = A V' }
    ],
    syncMap: [
      { vizElementId: 'viz-matmul', equationTermId: 'eq-weighted-sum', codeLineId: 'code-weighted-sum' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain the multiplication of weights and values.',
        script: 'Now we blend our Value vectors together. We multiply the attention weights matrix A by the Value matrix V. Each row in the output represents a weighted average of all Value vectors in the sentence.',
        audienceQuestion: 'Why do we multiply by V instead of the raw input tokens?',
        expectedAnswer: 'Because Value vectors contain contextual semantic representations optimized during training, rather than raw embeddings.',
        misconception: 'Clarify that this multiplication is done separately for each head.',
        transition: 'Let\'s check the resulting summed vectors.'
      },
      {
        duration: '~30s',
        objective: 'Demonstrate head concatenation.',
        script: 'Here is the final summed output. We concatenate the vectors from all four heads side-by-side, rebuilding a single matrix of dimension sequence length by d_model.',
        audienceQuestion: 'What is the shape of the concatenated matrix?',
        expectedAnswer: 'It is four by sixteen, matching our model\'s d_model shape.',
        misconception: 'Make sure students realize that these concatenated vectors are now ready for the output projection.',
        transition: 'Let\'s review this step.'
      },
      {
        duration: '~20s',
        objective: 'Summarize the weighted sum transition.',
        script: 'We have combined the values of all tokens according to their attention coefficients. Let\'s look at what changed.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Let\'s verify our understanding.'
      },
      {
        duration: '~30s',
        objective: 'Assess attention aggregation knowledge.',
        script: 'Take a look at the quick check question. Consider what is actually mixed in self-attention, and how it is routed.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Excellent. Now we proceed to comparing attention heads.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Help students understand that attention weights act as routing coefficients, controlling which value features flow to which output tokens.'
      ],
      misconceptions: [
        'Emphasize that concatenation does not mix values mathematically; it simply places head outputs side-by-side. Mixing happens in the subsequent Output Projection.'
      ],
      suggestedQuestions: [
        'How would the output change if a token gave 1.0 attention to itself and 0.0 to all others?'
      ]
    }
  },
  'split-heads': {
    eyebrow: 'Attention · Subspaces',
    title: 'Split into Multiple Heads',
    fourQuestions: {
      whatIsHappening: 'We reshape the Query matrix Q of shape [seq_len, d_model] into [seq_len, num_heads, d_k], and then transpose it to [num_heads, seq_len, d_k].',
      why: 'This divides the d_model dimensions into independent attention subspaces (heads). Each head can focus on different types of syntactic and semantic relationships in parallel.',
      whatChanged: 'A single sequence matrix [seq_len, d_model] is converted into multiple head matrices of shape [num_heads, seq_len, d_k].',
      whatToObserve: 'Compare Step 0 (Reshape) with Step 1 (Transpose). In Step 0, the segments are grouped by token; in Step 1, the segments are grouped by attention Head.',
    },
    body: {
      beginner: 'Instead of looking at the whole word vector at once, we split it into smaller chunks called "Heads". Think of this as putting on different colored glasses: one head might look for subjects, another for verbs, and another for adjectives. We group these segments by head so they can compute attention independently.',
      mtech: 'The multi-head attention mechanism splits the Q, K, and V matrices along the channel dimension. For a projection matrix Q [seq, d_model], it is first reshaped to [seq, num_heads, d_k] and then transposed to [num_heads, seq, d_k]. This places the head dimension first, enabling batched matrix multiplications.',
      research: 'Multi-head splitting projects representations into multiple subspaces. Formally, Q_h = Transpose(Reshape(Q, [S, H, d_k]), [1, 0, 2]) where S is seq_len, H is num_heads, and d_k = d_model / H. This transposition is essential to leverage highly optimized batch matrix multiplication (GEMM) libraries during the Q * K^T calculation.',
    },
    deepDive: {
      math: 'Q_h = \\text{Transpose}\\left(\\text{Reshape}\\left(Q, [S, H, d_k]\\right), [1, 0, 2]\\right)',
      complexity: 'O(seq_len × d_model) memory copy or metadata-only view reshape.',
      matrixEquivalence: 'The split partitions the 16-dimensional channel space into 4 independent subspaces of size 4. Cell (t, h) in Step 0 maps to row t of Head h in Step 1.',
      misconceptions: [
        'Reshaping does not copy memory in modern frameworks like PyTorch; it simply changes the tensor stride metadata. However, transposing dimensions usually requires a contiguous memory copy.',
      ],
      notes: 'd_model = 16, num_heads = 4, d_k = 4. Each token\'s vector is split into 4 equal quarters.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why do we need multiple heads?',
          body: 'Single-head attention averages all relationships, which can cause key details to blur. Multi-head attention allows different heads to simultaneously track different relationships (e.g. subject-verb, pronoun-noun).',
        },
        {
          title: 'Why transpose the dimensions?',
          body: 'For fast computing: transposing swaps the sequence and head dimensions. By placing the head dimension first (shape [heads, seq_len, d_k]), we can perform standard 2D matrix multiplications for all heads in parallel using a single batched operation.',
        }
      ],
      example: {
        left: [0.1, -0.2, 0.5, 0.3, 0.9, -0.8, 0.1, 0.4, -0.2, 0.6, 0.5, 0.1, 0.0, -0.3, 0.4, 0.8],
        right: [0.1, -0.2, 0.5, 0.3],
        caption: 'A single token vector of size 16 (left) is sliced into its Head 0 segment of size 4 (right).'
      }
    },
    beforeAfter: {
      before: { label: 'Query Projection Q', shape: null },
      after: { label: 'Split Head Subspaces Q_h', shape: null },
      whatChanged: 'We partitioned the channel dimension and transposed the tensor dimensions to separate the attention heads.',
      structured: {
        entered: 'Projection matrix Q of shape [seq_len, d_model].',
        happened: 'Reshaped the columns into [seq_len, num_heads, d_k] and transposed the sequence and head dimensions.',
        changed: 'The single representation is split into 4 independent attention subspaces.',
        leaves: 'Tensor of shape [num_heads, seq_len, d_k] representing Q for each head.',
      }
    },
    quickCheck: {
      question: 'Why do we transpose the shape from [seq_len, num_heads, d_k] to [num_heads, seq_len, d_k]?',
      choices: [
        'To group the data by attention heads, enabling parallel batch matrix multiplications.',
        'To combine all heads back into a single vector of size d_model.',
        'To add positional encoding to the head dimensions.',
        'To apply the softmax normalization function.'
      ],
      correctIndex: 0,
      explanation: 'Correct! Transposing swaps the sequence and head dimensions. Placing the head dimension first allows the model to compute attention matching (Q * K^T) for all heads in parallel using batched matrix multiplication.',
      transition: 'Now that the matrices are split and transposed, let\'s calculate the relevance scores in Q * K^T.',
      distractorNotes: {
        1: 'Incorrect. Combining heads happens during concatenation at the end of attention, not at the beginning.',
        2: 'Incorrect. Positional encoding was already added to the inputs before projections.',
        3: 'Incorrect. Softmax is a mathematical function that does not use Value vectors.'
      }
    },
    pytorch: [
      { id: 'code-split-heads', code: 'Q = Q.view(seq_len, num_heads, d_k).transpose(0, 1)' }
    ],
    equationTerms: [
      { id: 'eq-split-heads', tex: 'Q_h = \\text{Transpose}\\left(\\text{Reshape}\\left(Q, [S, H, d_k]\\right), [1, 0, 2]\\right)' }
    ],
    syncMap: [
      { vizElementId: 'viz-matmul', equationTermId: 'eq-split-heads', codeLineId: 'code-split-heads' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain the reshaping step.',
        script: 'Now we split our Query projection into multiple heads. We reshape the matrix from sequence length by d_model into sequence length by number of heads by d_k. This separates the token vectors into distinct head channels.',
        audienceQuestion: 'How many dimensions does each token segment have after reshaping?',
        expectedAnswer: 'Each token segment has size four, which is d_k.',
        misconception: 'Make sure students realize that the data is still grouped by token in this step.',
        transition: 'Now let\'s transpose the dimensions to group them by head.'
      },
      {
        duration: '~35s',
        objective: 'Explain transposition.',
        script: 'Next, we transpose the sequence and head dimensions. This groups the segment slices by attention head, placing the head dimension first. The shape is now number of heads by sequence length by d_k.',
        audienceQuestion: 'Why is this shape change beneficial for GPU computing?',
        expectedAnswer: 'Because it allows us to feed the matrices for all heads into a single batched matrix multiplication, calculating attention in parallel.',
        misconception: 'Confirm that this transposition happens similarly for Key and Value projections.',
        transition: 'Let\'s review what this accomplished.'
      },
      {
        duration: '~20s',
        objective: 'Review the split-heads shape change.',
        script: 'We have divided our representation into independent head subspaces. Let\'s see the shape change summary.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Let\'s check our understanding of transposition.'
      },
      {
        duration: '~30s',
        objective: 'Assess multi-head transposition mechanics.',
        script: 'Answer the quick check question. Think about why grouping by head allows parallel batch computing.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Excellent. Now we proceed to Q * K^T score calculation.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Use the analogy of sorting books: Reshaping is like organizing books in columns on a desk, and Transposing is like packing them into separate boxes based on their category.'
      ],
      misconceptions: [
        'Explain that the split is purely a reshape operation; no linear weights or parameter calculations occur here.'
      ],
      suggestedQuestions: [
        'How does changing the number of heads affect d_k if d_model is kept constant?'
      ]
    }
  },
  'heads-compare': {
    eyebrow: 'Attention · Visualizing Heads',
    title: 'Individual Attention Heads',
    fourQuestions: {
      whatIsHappening: 'We visualize the attention weight matrices of all 4 heads side-by-side to compare their focus patterns.',
      why: 'Multiple heads allow the model to capture different types of relationships simultaneously (e.g. subjects vs. verbs vs. direct objects).',
      whatChanged: 'We display the 3D attention tensor [num_heads, seq_len, seq_len] as 4 independent, side-by-side 2D heatmaps.',
      whatToObserve: 'Compare the attention grids for Head 0, Head 1, Head 2, and Head 3. Notice how different heads focus on different token pairs.',
    },
    body: {
      beginner: 'By looking at all attention heads side-by-side, we can see how they focus on different parts of the sentence. One head might focus heavily on connecting pronouns to their nouns, while another head focuses on connecting verbs to their direct objects. This diversity is what makes multi-head attention so powerful.',
      mtech: 'Each attention head h computes its own softmax attention matrix A_h of shape [seq_len, seq_len]. We render all A_h grids side-by-side. The distinct color patterns highlight that each head acts as an independent feature filter, learning specialized contextual relationships.',
      research: 'Multi-head attention projects Queries, Keys, and Values into H = 4 distinct parameter subspaces. We visualize A_h = Softmax(Q_h K_h^T / sqrt(d_k)) for h in [0, 3]. Because the projection parameters (W_q^h, W_k^h) are initialized independently, the heads naturally converge to complementary attention patterns (e.g., local positional matching, dependency relationships).',
    },
    deepDive: {
      math: 'A_h = \\text{Softmax}\\left(\\frac{Q_h K_h^T}{\\sqrt{d_k}}\\right)',
      complexity: 'O(heads × seq_len^2) operations to render all grids.',
      matrixEquivalence: 'The side-by-side matrices represent 4 distinct attention distributions. For any row in any head, the values sum to exactly 1.0.',
      misconceptions: [
        'Heads do not share weights during projection. If they did, they would compute identical attention maps, defeating the purpose of multi-head attention.',
      ],
      notes: 'Each head h has shape [4, 4] in this lecture mode sentence. Notice how Head 0 has strong diagonal weights (local attention), while Head 2 has vertical stripes (focusing on a specific word).',
    },
    whyPanel: {
      items: [
        {
          title: 'Why visualize heads side-by-side?',
          body: 'Comparing them side-by-side highlights the specialization. If you look closely, you\'ll see that some heads focus on adjacent words (local context), while others focus on long-range dependencies across the entire sentence.',
        },
        {
          title: 'Do heads ever focus on the same things?',
          body: 'Sometimes heads can overlap or show redundancy, especially in very deep models. However, during training, backpropagation forces them to specialize in different features to minimize loss.',
        }
      ],
      example: {
        left: [0.9, 0.05, 0.02, 0.03],
        right: [0.1, 0.25, 0.55, 0.1],
        caption: 'Head 0 (local subject focus) vs Head 1 (verb-object focus) weight rows for the same token.'
      }
    },
    beforeAfter: {
      before: { label: 'Single Head Attention Weights', shape: null },
      after: { label: 'Multi-Head Attention Weights', shape: null },
      whatChanged: 'We expanded the view from a single head to compare the attention patterns of all heads in parallel.',
      structured: {
        entered: 'A single attention head matrix of shape [seq_len, seq_len].',
        happened: 'Visualized all 4 attention head weight matrices side-by-side.',
        changed: 'Displays the complete set of attention subspaces simultaneously.',
        leaves: 'Four side-by-side grids representing the attention weights for all heads.',
      }
    },
    quickCheck: {
      question: 'What is the main benefit of having multiple attention heads rather than just one?',
      choices: [
        'It allows the model to learn multiple, complementary types of relationships simultaneously.',
        'It reduces the memory footprint of the transformer model.',
        'It forces all attention scores to sum to exactly 1.0.',
        'It eliminates the need for positional encoding.'
      ],
      correctIndex: 0,
      explanation: 'Correct! Multiple heads allow the model to focus on different syntactic and semantic relationships in parallel, capturing much richer context than a single-head representation.',
      transition: 'Now that we have compared individual heads, we concatenate them back together. Let\'s see that in Concatenation.',
      distractorNotes: {
        1: 'Incorrect. Multi-head attention actually increases parameter counts slightly due to projection weights.',
        2: 'Incorrect. Softmax is what forces the weights to sum to 1.0, not the number of heads.',
        3: 'Incorrect. Multi-head attention still relies on positional encoding to track sequence positions.'
      }
    },
    pytorch: [
      { id: 'code-heads-compare', code: 'weights_all = torch.stack([softmax(matmul(Q_h, K_h.T) / sqrt(d_k)) for h in heads])' }
    ],
    equationTerms: [
      { id: 'eq-heads-compare', tex: 'A_h = \\text{Softmax}\\left(\\frac{Q_h K_h^T}{\\sqrt{d_k}}\\right)' }
    ],
    syncMap: [
      { vizElementId: 'viz-matmul', equationTermId: 'eq-heads-compare', codeLineId: 'code-heads-compare' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain head specialization.',
        script: 'Let\'s compare all four attention heads side-by-side. By dividing our channel dimension, each head has learned to look for different grammatical and semantic relations in the sentence.',
        audienceQuestion: 'Which head shows the most active diagonal values?',
        expectedAnswer: 'Head 0, which focuses primarily on neighboring words.',
        misconception: 'Remind students that all of these grids are computed simultaneously, not sequentially.',
        transition: 'Let\'s summarize the difference between single and multi-head views.'
      },
      {
        duration: '~20s',
        objective: 'Review shape of parallel attention heads.',
        script: 'We are displaying the entire four by four by four attention weights tensor. We can see all four heads in parallel.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Let\'s check our understanding.'
      },
      {
        duration: '~30s',
        objective: 'Assess multi-head benefits.',
        script: 'Take a look at the quick check question. Consider what would be lost if we only had a single attention head for the entire model.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Perfect. Now let\'s see how we merge these heads back in the Concatenation step.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Have students look at the vertical columns in Head 2: it highlights how a single word (like "chased" or "dog") can attract attention from all other words in that subspace.'
      ],
      misconceptions: [
        'Clarify that there is no parameter sharing between the heads; their projection matrices are initialized and updated independently.'
      ],
      suggestedQuestions: [
        'What would happen if we initialized all heads with the exact same weights?'
      ]
    }
  },
  'concat': {
    eyebrow: 'Attention · Recombining',
    title: 'Concatenation',
    fourQuestions: {
      whatIsHappening: 'We concatenate the outputs of all 4 attention heads of shape [num_heads, seq_len, d_k] back into a single matrix of shape [seq_len, d_model].',
      why: 'This stitches the information captured by each independent attention subspace side-by-side, preparing it for the final output projection.',
      whatChanged: 'Multiple head tensors [num_heads, seq_len, d_k] are untransposed and concatenated back to [seq_len, d_model = 16].',
      whatToObserve: 'Hover over a head output block on the left. See how its columns correspond exactly to that head\'s segment inside the combined representations on the right.',
    },
    body: {
      beginner: 'After our different attention heads have analyzed the sentence from their unique perspectives, we stitch their findings back together. We take the 4-dimensional output vectors of each head and place them side-by-side to restore a single 16-dimensional vector for each word. This is like assembling a jigsaw puzzle to see the full picture.',
      mtech: 'Concatenation reverses the multi-head split. The independent head output vectors O_h [seq, d_k] are first transposed back to [seq, num_heads, d_k] and then reshaped to [seq, d_model] along the channel dimension. No mathematical blending occurs here; it is purely a structural concatenation.',
      research: 'Multi-head concatenation merges representation subspaces. Formally, O_{concat} = Concat(O_0, O_1, ..., O_{H-1}) where O_h has shape [seq_len, d_k]. The resulting tensor has shape [seq_len, d_model = H * d_k]. This tensor is now ready to be mixed by the linear output projection W_o.',
    },
    deepDive: {
      math: 'O_{concat} = \\text{Concat}\\left(O_0, O_1, \\dots, O_{H-1}\\right)',
      complexity: 'O(seq_len × d_model) memory layout adjustment.',
      matrixEquivalence: 'The operation groups columns by head. Segment h of token t on the right is populated directly by token t\'s output vector of Head h on the left.',
      misconceptions: [
        'Concatenation does not mix features across heads. If one head focuses on verbs and another on nouns, they remain strictly separated in different channels until the subsequent Output Projection step.',
      ],
      notes: 'H = 4, d_k = 4. The four 4-dimensional vectors from each head are concatenated side-by-side to recreate a 16-dimensional representation matching d_model.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why concatenate instead of averaging?',
          body: 'Averaging head outputs would blend their features, losing the specialized details learned by each head. Concatenating keeps their representations distinct so that the downstream layers can decide how to weight and mix them.',
        },
        {
          title: 'Is this the end of the attention layer?',
          body: 'Almost. While concatenation stitches the features together, they are still isolated in separate head channels. We need one final linear transformation (the Output Projection) to mix these concatenated features back into a unified space.',
        }
      ],
      example: {
        left: [0.5, 0.2, -0.1, 0.8],
        right: [0.5, 0.2, -0.1, 0.8, 0.9, -0.3, 0.2, 0.4, 0.0, 0.1, 0.5, -0.2, 0.1, 0.8, -0.4, 0.2],
        caption: 'The Head 0 output (left) forms the first 4 dimensions of the 16-dimensional concatenated vector (right).'
      }
    },
    beforeAfter: {
      before: { label: 'Attention Head Outputs', shape: null },
      after: { label: 'Concatenated Representation', shape: null },
      whatChanged: 'We stitched the independent attention head vectors side-by-side along the channel dimension.',
      structured: {
        entered: 'Attention head outputs of shape [num_heads, seq_len, d_k].',
        happened: 'Untransposed the sequence/head dimensions and concatenated the head outputs side-by-side.',
        changed: 'The independent attention channels are combined into a single contextual representation.',
        leaves: 'A stitched representation matrix of shape [seq_len, d_model].',
      }
    },
    quickCheck: {
      question: 'Which of the following describes the Concatenation step in multi-head attention?',
      choices: [
        'It places the output vectors of all attention heads side-by-side without mathematically mixing them.',
        'It averages the outputs of all attention heads to reduce dimensionality.',
        'It multiplies the head outputs by the positional encoding matrix.',
        'It projects the token vectors back to the vocabulary space.'
      ],
      correctIndex: 0,
      explanation: 'Correct! Concatenation simply groups the independent attention head outputs side-by-side, preserving their distinct subspace features for the downstream layers.',
      transition: 'Now that the heads are concatenated, we project them back to the original model space. Let\'s see that in Output Projection.',
      distractorNotes: {
        1: 'Incorrect. Averaging is not used because it would blur the distinct features captured by each head.',
        2: 'Incorrect. Positional encoding is only added at the input level, not here.',
        3: 'Incorrect. Vocabulary mapping happens at the very end of the entire Transformer block, not here.'
      }
    },
    pytorch: [
      { id: 'code-concat', code: 'O = torch.cat([O_h for O_h in outputs], dim=-1)' }
    ],
    equationTerms: [
      { id: 'eq-concat', tex: 'O_{concat} = \\text{Concat}\\left(O_0, O_1, \\dots, O_{H-1}\\right)' }
    ],
    syncMap: [
      { vizElementId: 'viz-matmul', equationTermId: 'eq-concat', codeLineId: 'code-concat' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain multi-head recombination.',
        script: 'After computing attention for each head, we stitch their findings back together. We take the output vectors from all four heads and concatenate them side-by-side along the channel dimension.',
        audienceQuestion: 'What is the resulting shape of the matrix after stitching?',
        expectedAnswer: 'It is sequence length by d_model, which is four by sixteen in this sentence.',
        misconception: 'Emphasize that this is purely structural; no linear multiplication occurs yet.',
        transition: 'Let\'s review this shape change.'
      },
      {
        duration: '~20s',
        objective: 'Review concatenation shapes.',
        script: 'We have recombined the attention heads. Let\'s look at what changed.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Let\'s verify our understanding.'
      },
      {
        duration: '~30s',
        objective: 'Assess concatenation understanding.',
        script: 'Look at the quick check question. Recall whether concatenation mixes features or just aligns them.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Excellent. Now we proceed to the Output Projection step.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Show how each quarter of the concatenated vector on the right matches the color of its corresponding head card on the left.'
      ],
      misconceptions: [
        'Make sure students understand that the Concatenation step itself has no trainable weights. Training weights only appear in the final Output Projection (Wo).'
      ],
      suggestedQuestions: [
        'How does concatenation preserve the independence of head features compared to summing them?'
      ]
    }
  },
  'output-proj': {
    eyebrow: 'Attention · Output Space',
    title: 'Output Projection',
    fourQuestions: {
      whatIsHappening: 'We multiply the concatenated head matrix O of shape [seq_len, d_model] by the output weight matrix Wo of shape [d_model, d_model] and add the bias bo.',
      why: 'This blends the features from all independent attention heads back into a single unified space, allowing the model to project and mix the concatenated representations.',
      whatChanged: 'The concatenated attention representations are linearly projected, yielding the attention block output Y of shape [seq_len, d_model].',
      whatToObserve: 'Hover over cells in the weight matrix. Observe how each output dimension is a linear combination of all concatenated head dimensions.',
    },
    body: {
      beginner: 'Now that the findings from all heads are stitched together side-by-side, we need to mix them. We multiply this concatenated vector by a final projection weight matrix Wo. This blends the features together, allowing the heads to communicate and synthesize their findings into a single unified meaning.',
      mtech: 'The output projection maps the concatenated head outputs O back to the hidden space: Y = O * Wo + bo, where Wo is the [d_model, d_model] projection matrix and bo is the [d_model] bias. This mixes information across different heads, resolving head-specific structures.',
      research: 'The final linear transformation of multi-head attention projects the concatenated representation back to the model space: MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * Wo. This mixes channel-wise components across different heads, recovering a unified representation of shape [seq_len, d_model].',
    },
    deepDive: {
      math: 'Y = O W_o + b_o',
      complexity: 'O(seq_len × d_model^2) parameter operations.',
      matrixEquivalence: 'The matrix multiplication projects the 16-dimensional concatenated vectors back into a new 16-dimensional space, mixing the independent attention subspaces.',
      misconceptions: [
        'Wo is not a projection to vocabulary space. It is a projection within the hidden layer d_model (size 16), which keeps the output dimension identical to the input dimension for the residual stream.',
      ],
      notes: 'd_model = 16. The projection weights Wo [16, 16] and bias bo [16] are learned parameters optimized during training.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why do we need Wo?',
          body: 'Concatenation simply groups head features side-by-side. Without Wo, the features of different heads would remain isolated in separate channels. Wo mixes these channels so the downstream layers receive unified contextual vectors.',
        },
        {
          title: 'How does it support residual streams?',
          body: 'For residual connections (X + Attention(X)) to work, the output of the attention block must match the input dimension exactly. Wo ensures that the output is projected back to d_model (size 16) dimensions.',
        }
      ],
      example: {
        left: [0.5, 0.2, -0.1, 0.8, 0.9, -0.3, 0.2, 0.4, 0.0, 0.1, 0.5, -0.2, 0.1, 0.8, -0.4, 0.2],
        right: [1.2, -0.5, 2.1, 0.6, 0.8, -0.1, 1.4, 0.3, 0.2, 0.5, -0.1, 0.0, 0.9, -0.2, 0.3, 1.1],
        caption: 'Concatenated vector of size 16 (left) projected back to 16 dimensions (right) using Wo.'
      }
    },
    beforeAfter: {
      before: { label: 'Concatenated Heads Output', shape: null },
      after: { label: 'Projected Attention Output', shape: null },
      whatChanged: 'We projected the concatenated representation back to the unified model space, mixing the attention head features.',
      structured: {
        entered: 'Concatenated multi-head output matrix O of shape [seq_len, d_model].',
        happened: 'Multiplied by the output projection matrix Wo and added the bias vector bo.',
        changed: 'Subspace features from independent heads are linearly mixed and blended.',
        leaves: 'Final attention block representation matrix Y of shape [seq_len, d_model].',
      }
    },
    quickCheck: {
      question: 'What is the primary role of the Output Projection (Wo) matrix in multi-head attention?',
      choices: [
        'To mix the independent attention head features together and restore the model\'s hidden dimension.',
        'To normalize the attention weights so they sum to 1.0.',
        'To split the model representations into multiple head subspaces.',
        'To project token embeddings back to vocabulary logits.'
      ],
      correctIndex: 0,
      explanation: 'Correct! The Output Projection matrix Wo mixes the side-by-side features of all attention heads together and projects the tensor back to the hidden d_model dimension (size 16), making it compatible with the residual stream.',
      transition: 'Now that the attention block is complete, we add its output back to the residual stream. Let\'s see that in the Residual Connection.',
      distractorNotes: {
        1: 'Incorrect. Normalization is performed by the Softmax function, not Wo.',
        2: 'Incorrect. Splitting into subspaces happens at the beginning of attention, not at the end.',
        3: 'Incorrect. Logits projection occurs at the final classification layer of the entire model.'
      }
    },
    pytorch: [
      { id: 'code-output-proj', code: 'attn_out = torch.matmul(concat_heads, Wo) + bo' }
    ],
    equationTerms: [
      { id: 'eq-output-proj', tex: 'Y = O W_o + b_o' }
    ],
    syncMap: [
      { vizElementId: 'viz-matmul', equationTermId: 'eq-output-proj', codeLineId: 'code-output-proj' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain the output projection weights matrix.',
        script: 'To conclude the attention layer, we project the concatenated heads back to the model space. We multiply by the output weights matrix Wo. This matrix acts as a cross-head mixer.',
        audienceQuestion: 'What is the dimension of the Wo weight matrix?',
        expectedAnswer: 'It is d_model by d_model, which is sixteen by sixteen.',
        misconception: 'Remind students that this weight matrix is learned and optimized during model training.',
        transition: 'Let\'s compute the projection.'
      },
      {
        duration: '~35s',
        objective: 'Explain projection matmul and bias addition.',
        script: 'We perform the matrix multiplication and add the bias vector bo. Each column in the output is a linear combination of all concatenated head dimensions, successfully blending their perspectives.',
        audienceQuestion: 'Why must the output shape match the input shape?',
        expectedAnswer: 'So that it can be added back to the input residual stream in the next step.',
        misconception: 'Make sure students notice that the individual head columns are now fully blended.',
        transition: 'Let\'s review what changed in this projection.'
      },
      {
        duration: '~20s',
        objective: 'Review output projection shapes and data shift.',
        script: 'We have projected our representation back. Let\'s look at the summary.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Let\'s verify our understanding of the output projection.'
      },
      {
        duration: '~30s',
        objective: 'Assess output projection functionality.',
        script: 'Take a look at the quick check question. Think about how the output projection relates to the residual stream.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Excellent. Now we proceed to adding this output back to the residual stream.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Emphasize that the Output Projection is the first time features from different heads actually mix mathematically.'
      ],
      misconceptions: [
        'Clarify that Wo does not change the sequence length; it only mixes and projects the channel dimension.'
      ],
      suggestedQuestions: [
        'What would happen to the residual stream if we omitted the output projection?'
      ]
    }
  },
  'residual-1': {
    eyebrow: 'Attention · Residual stream',
    title: 'Residual Connection ①',
    fourQuestions: {
      whatIsHappening: 'We add the skip connection input X_pe directly to the Attention block output A(X) cell-by-cell: Y = X_pe + A(X).',
      why: 'This allows gradients to flow directly back during training, mitigating vanishing gradients, and preserves original positional/token details.',
      whatChanged: 'The token representation matrix is updated by adding contextual adjustments from multi-head self-attention.',
      whatToObserve: 'Audience can observe that cell values on the right represent the direct sum of the left two matrices.',
    },
    body: {
      beginner: 'A residual connection acts like a highway. We take the original input vector and add the attention adjustments directly back onto it. This ensures that the model never loses the original token meanings while introducing new context.',
      mtech: 'The skip connection establishes Y = X + Attention(X), allowing feature adjustments to act additively rather than multiplicatively. This preserves identity maps and ensures clean gradient backpropagation.',
      research: 'Residual bypass is critical for training deep architectures. By summing the attention sub-layer output with its input, we create a path that avoids optimization bottlenecks, making training extremely stable.',
    },
    deepDive: {
      math: 'Y = X + \\text{Attention}(X)',
      complexity: 'O(seq\\_len \\times d\\_model) element-wise additions.',
      matrixEquivalence: 'Additive skip connection, preserving the original model dimensions [seq_len, d_model] for downstream layers.',
      misconceptions: [
        'The skip connection does not concatenate representations. It adds them cell-by-cell. Concatenation would double the dimension size.',
      ],
      notes: 'No learnable parameters are involved in the skip addition itself.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why do we add instead of multiply?',
          body: 'Multiplication causes gradients to shrink exponentially as depth increases. Addition acts as a direct highway, allowing gradients to flow back unaltered.',
        },
        {
          title: 'What does the attention block represent?',
          body: 'The attention block output represents an "update delta". Instead of rewriting the entire token representation, it calculates what adjustments should be added to the input representation.',
        }
      ],
      example: {
        left: [0.5, -0.2, 0.1, 0.4],
        right: [0.1, 0.3, -0.2, 0.2],
        caption: 'Input values (left) added to attention delta values (right) to form [0.6, 0.1, -0.1, 0.6].'
      }
    },
    beforeAfter: {
      before: { label: 'Input to Attention block', shape: null },
      after: { label: 'Residual Stream Output', shape: null },
      whatChanged: 'We added the attention output to the skip connection input.',
      structured: {
        entered: 'Input representation X_pe of shape [seq_len, d_model].',
        happened: 'Attention output A(X) is added to X_pe cell-by-cell.',
        changed: 'The representation is enriched with context while preserving identity.',
        leaves: 'Residual output representation of shape [seq_len, d_model].',
      }
    },
    quickCheck: {
      question: 'Why are residual (skip) connections mathematically critical for training deep Transformer networks?',
      choices: [
        'They provide a linear pathway for gradients to backpropagate directly without vanishing.',
        'They normalize the intermediate activations to zero mean and unit variance.',
        'They allow the attention heads to compare tokens across different sequences.',
        'They project the hidden representations back to the vocabulary size.'
      ],
      correctIndex: 0,
      explanation: 'Correct! By summing the output and input of the attention block, skip connections create a direct gradient path that bypasses the non-linear transformations, preventing vanishing gradients.',
      transition: 'Now that the residual addition is complete, we normalize the features. Let\'s see that in the Layer Normalization scene.',
      distractorNotes: {
        1: 'Incorrect. Normalization is the task of LayerNorm, not the skip connection.',
        2: 'Incorrect. Attention heads compare tokens; skip connections perform element-wise addition.',
        3: 'Incorrect. Vocabulary projection is handled by the final lm_head layer.'
      }
    },
    pytorch: [
      { id: 'code-residual-1', code: 'x = x + attention_out' }
    ],
    equationTerms: [
      { id: 'eq-residual-1', tex: 'Y = X + \\text{Attention}(X)' }
    ],
    syncMap: [
      { vizElementId: 'viz-residual', equationTermId: 'eq-residual-1', codeLineId: 'code-residual-1' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Explain the main attention output path.',
        script: 'After the Output Projection, we obtain the attention update delta. We must add this delta back to our original input.',
        audienceQuestion: 'Why do we call it a delta update?',
        expectedAnswer: 'Because it represents the contextual adjustments that refine the original representations.',
        misconception: 'Remind students that the attention output is not the final layer output; it is an additive refinement.',
        transition: 'Let\'s view the skip connection highway.'
      },
      {
        duration: '~30s',
        objective: 'Explain skip connection routing.',
        script: 'Look at the skip connection highway at the top. We route the original input directly around the attention block, bypassing all transformations.',
        audienceQuestion: 'Does the skip connection path have trainable parameters?',
        expectedAnswer: 'No, it is a direct parameter-free addition.',
        misconception: 'Ensure students understand that the skip connection behaves purely as an identity mapping.',
        transition: 'Let\'s perform the addition.'
      },
      {
        duration: '~35s',
        objective: 'Explain element-wise addition.',
        script: 'We add the original input and the attention output delta cell-by-cell. This yields the new representation matrix, mixing original features and context.',
        audienceQuestion: 'Can we perform this addition if the dimensions do not match?',
        expectedAnswer: 'No, the shapes of both matrices must be identical.',
        misconception: 'Point out that hover highlights align corresponding rows across all three matrices.',
        transition: 'Let\'s review what changed.'
      },
      {
        duration: '~20s',
        objective: 'Review residual connection shapes and data flow.',
        script: 'We have successfully merged our skip connection. Let\'s look at the summary.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Let\'s check our understanding of the residual connection.'
      },
      {
        duration: '~30s',
        objective: 'Assess skip connection functionality.',
        script: 'Take a look at the quick check question. Recall how skip connections solve vanishing gradients.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Excellent. Let\'s move on to the Layer Normalization stage.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Use the highway metaphor to explain how features can pass through the network layers unchanged.'
      ],
      misconceptions: [
        'Students often confuse addition with concatenation. Highlight the shape stays [seq, d_model] and does not double.'
      ],
      suggestedQuestions: [
        'How would removing the skip connection affect the training speed of very deep networks?'
      ]
    }
  },
  'residual-2': {
    eyebrow: 'Feed-forward · Residual stream',
    title: 'Residual Connection ②',
    fourQuestions: {
      whatIsHappening: 'We add the LayerNorm output directly to the FFN output cell-by-cell: Y = LN_1(X) + FFN(LN_1(X)).',
      why: 'This integrates feed-forward updates back into the residual stream while mitigating vanishing gradients.',
      whatChanged: 'The token representation matrix is updated by adding feed-forward adjustments.',
      whatToObserve: 'Audience can observe that cell values on the right represent the direct sum of the left two matrices.',
    },
    body: {
      beginner: 'Just like the attention block, the feed-forward network produces an update delta. We take the input to the FFN and add the FFN\'s output directly back onto it, updating our token meanings.',
      mtech: 'The second skip connection establishes Y = LN_1(X) + FFN(LN_1(X)), allowing feed-forward feature adjustments to act additively.',
      research: 'SUM FFN updates back to the primary residual highway, enabling decoupled optimization paths across layer sub-components.',
    },
    deepDive: {
      math: 'Y = X + \\text{FFN}(X)',
      complexity: 'O(seq\\_len \\times d\\_model) element-wise additions.',
      matrixEquivalence: 'Additive skip connection, preserving the original model dimensions [seq_len, d_model].',
      misconceptions: [
        'The skip connection does not use FFN weights; it acts after FFN processing is complete.',
      ],
      notes: 'No learnable parameters are involved in the skip addition itself.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why do we need a second skip connection?',
          body: 'Every sub-layer in a Transformer (Self-Attention and FFN) is wrapped in a residual connection. This ensures consistent gradient highways throughout the entire block.',
        }
      ],
      example: {
        left: [0.3, -0.1, 0.5, 0.2],
        right: [-0.2, 0.4, 0.1, -0.3],
        caption: 'FFN input values (left) added to FFN output values (right) to form [0.1, 0.3, 0.6, -0.1].'
      }
    },
    beforeAfter: {
      before: { label: 'Input to FFN block', shape: null },
      after: { label: 'Layer Residual Output', shape: null },
      whatChanged: 'We added the FFN output to the FFN input skip connection.',
      structured: {
        entered: 'Input representation of shape [seq_len, d_model].',
        happened: 'FFN output is added to FFN input cell-by-cell.',
        changed: 'The representation is enriched with non-linear feed-forward updates.',
        leaves: 'Final block residual output representation of shape [seq_len, d_model].',
      }
    },
    quickCheck: {
      question: 'What is added to the FFN output in Residual Connection ②?',
      choices: [
        'The input to the FFN layer (LayerNorm ① output).',
        'The original positional embeddings.',
        'The attention weights matrix.',
        'The output projection bias vector.'
      ],
      correctIndex: 0,
      explanation: 'Correct! The input to the FFN layer (which has undergone LayerNorm ①) is routed through the skip connection and added directly to the FFN output.',
      transition: 'Now that the second residual connection is complete, we perform final block normalization. Let\'s see Layer Normalization ②.',
      distractorNotes: {
        1: 'Incorrect. Positional embeddings are only added at the very start of the network.',
        2: 'Incorrect. Attention weights are not in the residual stream path.',
        3: 'Incorrect. The bias vector is already incorporated into the projection outputs.'
      }
    },
    pytorch: [
      { id: 'code-residual-2', code: 'x = x + ffn_out' }
    ],
    equationTerms: [
      { id: 'eq-residual-2', tex: 'Y = X + \\text{FFN}(X)' }
    ],
    syncMap: [
      { vizElementId: 'viz-residual', equationTermId: 'eq-residual-2', codeLineId: 'code-residual-2' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Explain FFN output path.',
        script: 'After the Feed Forward Network, we obtain the FFN update delta. We must add this delta back to the FFN input.',
        audienceQuestion: 'What does the FFN layer refine compared to attention?',
        expectedAnswer: 'Attention captures token-to-token relations; FFN refines individual token features.',
        misconception: 'The FFN output is added back as a delta change.',
        transition: 'Let\'s view the second skip connection highway.'
      },
      {
        duration: '~30s',
        objective: 'Explain second skip connection routing.',
        script: 'Just like before, the skip connection routes the FFN input around the FFN block, bypassing its projection layers.',
        audienceQuestion: 'Are the first and second skip connection paths identical?',
        expectedAnswer: 'They function identically, but carry different activation tensors.',
        misconception: 'The skip connection contains no learnable parameters.',
        transition: 'Let\'s perform the addition.'
      },
      {
        duration: '~35s',
        objective: 'Explain FFN element-wise addition.',
        script: 'We sum the FFN input and FFN output cell-by-cell. The resulting matrix represents the fully processed transformer block outputs.',
        audienceQuestion: 'Does this addition alter the dimension shapes?',
        expectedAnswer: 'No, shapes remain d_model.',
        misconception: 'Point out the clean hover alignment across all three matrices.',
        transition: 'Let\'s review what changed.'
      },
      {
        duration: '~20s',
        objective: 'Review second residual connection shapes and data flow.',
        script: 'Let\'s look at the summary of the block update.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Let\'s check our understanding.'
      },
      {
        duration: '~30s',
        objective: 'Assess second skip connection functionality.',
        script: 'Take a look at the quick check question. Think about which activations are being added.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Excellent. Let\'s proceed to the final Layer Normalization stage.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Emphasize that the Transformer block alternates Attention (relational mixing) and FFN (independent processing), each wrapped in skip connections.'
      ],
      misconceptions: [
        'Ensure students see that each skip connection wraps exactly one processing sub-layer.'
      ],
      suggestedQuestions: [
        'What would happen if we only had one skip connection around both layers together?'
      ]
    }
  },
  'layer-norm-1': {
    eyebrow: 'Attention · Normalization',
    title: 'Layer Normalization ①',
    fourQuestions: {
      whatIsHappening: 'We normalize the activations of each token row independently across the d_model dimensions, then apply scaling parameters (gamma) and shifting parameters (beta).',
      why: 'Normalization keeps the scale of activations consistent across layers, preventing training instability and accelerating learning convergence.',
      whatChanged: 'Each row is transformed to have mean 0.0 and variance 1.0 (pre-scaling), without changing the overall tensor shape.',
      whatToObserve: 'Audience can observe how calculations occur independently for each token row (channel-wise), unlike Batch Normalization which operates column-wise.',
    },
    body: {
      beginner: 'Layer Normalization makes sure the numbers in our vectors do not get too big or too small. We look at each token\'s vector, compute its average and variability, center its values around zero, and scale them to a standard size. Finally, we adjust them slightly using two learned tuning dials (gamma and beta).',
      mtech: 'Layer Normalization centers and scales the features row-by-row: X_norm = (X - mean) / sqrt(var + eps). The normalized representations are then linearly scaled and shifted: Y = gamma * X_norm + beta. This stabilizes the hidden representations before they enter subsequent layers.',
      research: 'Layer normalization operates channel-wise over each individual token: LN(x) = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta. This stabilizes training dynamics and provides invariance to shifts in input scaling, crucial for optimization in deep multi-head architectures.',
    },
    deepDive: {
      math: 'Y = \\gamma \\odot \\hat{X} + \\beta',
      complexity: 'O(seq\\_len \\times d\\_model) row-wise standard normalizations.',
      matrixEquivalence: 'Row-wise standardization keeping shape [seq_len, d_model] identical. Parameters gamma [d_model] and beta [d_model] are learned during training.',
      misconceptions: [
        'Layer Normalization does not mix information between different tokens (rows). It normalizes each token\'s representation completely independently.',
      ],
      notes: 'Typically epsilon = 1e-5 to avoid division by zero.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why normalize row-by-row instead of column-by-column?',
          body: 'Column-wise normalization (Batch Normalization) depends on other sentences in the training batch, which is problematic for variable sequence lengths in transformers. LayerNorm operates purely within one token row, making it completely independent of batch size or context size.',
        },
        {
          title: 'What do gamma and beta do?',
          body: 'If we only normalized, we would force every token to have a fixed distribution. Gamma (scale) and beta (shift) let the model learn to restore whatever distribution shape is best for representing features.',
        }
      ],
      example: {
        left: [1.2, 0.4, -0.8, 2.0],
        right: [0.3, -0.6, -1.2, 1.5],
        caption: 'Row values with mean = 0.7, std = 1.1 normalized to [0.45, -0.27, -1.36, 1.18].'
      }
    },
    beforeAfter: {
      before: { label: 'Residual Output ①', shape: null },
      after: { label: 'LayerNorm Output ①', shape: null },
      whatChanged: 'We standardized the residual representations row-by-row and applied scale/shift parameters.',
      structured: {
        entered: 'Residual Output matrix X of shape [seq_len, d_model].',
        happened: 'Computed row-wise mean and variance, normalized values, and scaled/shifted with gamma and beta.',
        changed: 'Activations are standardized to zero mean and unit variance per row, then scaled.',
        leaves: 'Normalized representations Y of shape [seq_len, d_model].',
      }
    },
    quickCheck: {
      question: 'Unlike Batch Normalization, Layer Normalization computes mean and variance across which dimension?',
      choices: [
        'Across the channel/embedding dimension (row-wise for each token independently).',
        'Across all batch samples and sequences simultaneously.',
        'Across the sequence length dimension (column-wise for each feature).',
        'Across multiple heads inside the attention block.'
      ],
      correctIndex: 0,
      explanation: 'Correct! Layer Normalization normalizes the features of each token independently across the embedding (channel) dimension, making it robust against batch changes and sequence length variations.',
      transition: 'Now that the first normalization step is complete, the features enter the Feed Forward Network. Let\'s proceed to the FFN scene.',
      distractorNotes: {
        1: 'Incorrect. Batch Normalization normalizes across batch samples and sequence indices.',
        2: 'Incorrect. Normalizing column-wise across sequences is not LayerNorm.',
        3: 'Incorrect. Normalization does not occur across heads; it occurs across the full model dimension.'
      }
    },
    pytorch: [
      { id: 'code-layernorm-1', code: 'x = torch.nn.functional.layer_norm(x, x.shape[-1:])' }
    ],
    equationTerms: [
      { id: 'eq-layernorm-1', tex: 'Y = \\gamma \\odot \\left( \\frac{X - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\right) + \\beta' }
    ],
    syncMap: [
      { vizElementId: 'viz-layernorm', equationTermId: 'eq-layernorm-1', codeLineId: 'code-layernorm-1' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain LayerNorm row-wise statistics.',
        script: 'After adding features in the skip connection, we must normalize them. We start by calculating the mean mu and variance sigma squared for each token row independently.',
        audienceQuestion: 'Does LayerNorm share statistics across different tokens?',
        expectedAnswer: 'No, each token row is normalized completely independently of the other tokens.',
        misconception: 'Make sure students notice that the statistics are computed across the columns for each row.',
        transition: 'Let\'s normalize the activations.'
      },
      {
        duration: '~35s',
        objective: 'Explain row-wise standardization.',
        script: 'Next, we standardize each row value. We subtract the mean and divide by the standard deviation. This yields intermediate values with zero mean and unit variance.',
        audienceQuestion: 'Why do we add a small epsilon term to the variance?',
        expectedAnswer: 'To prevent division by zero in case the variance is exactly zero.',
        misconception: 'Point out how the intermediate values are now nicely bounded.',
        transition: 'Let\'s scale and shift using gamma and beta.'
      },
      {
        duration: '~35s',
        objective: 'Explain scale and shift step.',
        script: 'Finally, we apply scale factor gamma and shift factor beta. These learned parameters allow the layer to restore representation capacities.',
        audienceQuestion: 'What are the default values of gamma and beta?',
        expectedAnswer: 'Gamma is initialized to all ones, and beta is initialized to all zeros.',
        misconception: 'Remind students that gamma and beta are optimized during training.',
        transition: 'Let\'s review what changed in Layer Normalization.'
      },
      {
        duration: '~20s',
        objective: 'Review LayerNorm shapes and data flow.',
        script: 'We have normalized our representations. Let\'s look at the summary.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Let\'s check our understanding of Layer Normalization.'
      },
      {
        duration: '~30s',
        objective: 'Assess LayerNorm functionality.',
        script: 'Take a look at the quick check question. Think about how LayerNorm differs from Batch Normalization.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Excellent. Now we are ready to enter the Feed Forward Network.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Contrast LayerNorm and BatchNorm visually using a table. BatchNorm normalizes columns, LayerNorm normalizes rows.'
      ],
      misconceptions: [
        'Clarify that LayerNorm does not change the sequence length or the d_model dimensions; it is a value-scaling operation.'
      ],
      suggestedQuestions: [
        'How would Layer Normalization help prevent representation drift during training?'
      ]
    }
  },
  'ffn': {
    eyebrow: 'Feed-forward · Non-linear projection',
    title: 'Feed Forward Network (FFN)',
    fourQuestions: {
      whatIsHappening: 'Each token representation independently passes through a two-layer multi-layer perceptron (MLP) with a non-linear activation (ReLU in this implementation).',
      why: 'While attention handles communication between tokens, FFN handles token-specific feature transformation and introduces non-linearities essential for complex representation learning.',
      whatChanged: 'Tokens are projected to a higher-dimensional space (d_ff = 32) where non-linear activation is applied, then projected back to model dimension (d_model = 16).',
      whatToObserve: 'Audience can observe how negative values in the first linear layer output are completely suppressed to zero during the activation step.',
    },
    body: {
      beginner: 'The Feed Forward Network is like an individual thinking cap for each token. Once attention is done letting tokens chat and share context, each token goes through this FFN alone to process its newly gained information. It expands the vector into a larger size, filters out negative values, and compresses it back to model size.',
      mtech: 'The FFN is a position-wise network consisting of two linear transformations with a non-linear activation in between: FFN(x) = max(0, x W1 + b1) W2 + b2. It processes each position independently and identically, transforming channels locally.',
      research: 'The FFN sub-layer consists of a position-wise feed-forward network applied to each token vector independently. This MLP expands the vector to a higher capacity intermediate dimension (d_ff) to disentangle features, applies a non-linear activation (typically ReLU or GELU), and projects it back to the residual stream space.',
    },
    deepDive: {
      math: '\\text{FFN}(X) = \\text{ReLU}(X W_1 + b_1) W_2 + b_2',
      complexity: 'O(seq\\_len \\times d\\_model \\times d\\_ff) parameter operations.',
      matrixEquivalence: 'Two linear projections. First projects from 16 to 32 dimensions, second projects from 32 back to 16 dimensions.',
      misconceptions: [
        'There is absolutely no token-to-token communication in the FFN layer. The sequence length dimension remains completely isolated; only feature channels mix.',
      ],
      notes: 'd_model = 16, d_ff = 32. All weights (W1, W2) and biases (b1, b2) are learned parameters.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why do we need non-linear activations?',
          body: 'Without non-linear activations like ReLU, multiple linear layers would collapse mathematically into a single linear projection. Non-linearities allow the network to learn arbitrarily complex mapping functions.',
        },
        {
          title: 'Why project to a higher dimension and back?',
          body: 'Projecting to a higher capacity intermediate dimension (d_ff) allows features to be linearly separable, making it easier for the non-linear activation to filter out noise before the representation is compressed back.',
        }
      ],
      example: {
        left: [0.8, -0.4, 1.2, -0.9],
        right: [0.8, 0.0, 1.2, 0.0],
        caption: 'Negative values in the first linear output (left) are zeroed out by the ReLU activation (right).'
      }
    },
    beforeAfter: {
      before: { label: 'LayerNorm Output ①', shape: null },
      after: { label: 'FFN Block Output', shape: null },
      whatChanged: 'We transformed features at each token position independently using a non-linear projection.',
      structured: {
        entered: 'LayerNorm Output matrix of shape [seq_len, d_model].',
        happened: 'Each row multiplied by W1 + b1, activated via ReLU, and multiplied by W2 + b2.',
        changed: 'Feature channels are mixed and filtered non-linearly per position.',
        leaves: 'FFN output matrix of shape [seq_len, d_model].',
      }
    },
    quickCheck: {
      question: 'Which of the following statements is true regarding the Feed Forward Network (FFN) in a Transformer block?',
      choices: [
        'It processes each token position completely independently, mixing channel features without token-to-token communication.',
        'It calculates dot-product attention scores between different tokens in the sequence.',
        'It normalizes sequence channels to have zero mean and unit variance.',
        'It injects relative positional information into the sequence representation.'
      ],
      correctIndex: 0,
      explanation: 'Correct! The FFN is position-wise, meaning it processes each token independently and identically, mixing feature channels locally without any cross-token communication.',
      transition: 'Now that the FFN is complete, we add its output back to the residual stream. Let\'s see that in the second Residual Connection.',
      distractorNotes: {
        1: 'Incorrect. Attention scores are calculated in the self-attention layer, not the FFN.',
        2: 'Incorrect. Normalization is the job of the LayerNorm layers.',
        3: 'Incorrect. Positional information is only injected at the inputs.'
      }
    },
    pytorch: [
      { id: 'code-ffn', code: 'x = torch.matmul(torch.relu(torch.matmul(x, W1) + b1), W2) + b2' }
    ],
    equationTerms: [
      { id: 'eq-ffn', tex: 'Y = \\text{ReLU}(X W_1 + b_1) W_2 + b_2' }
    ],
    syncMap: [
      { vizElementId: 'viz-ffn', equationTermId: 'eq-ffn', codeLineId: 'code-ffn' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain the first FFN linear layer.',
        script: 'Following normalization, features enter the Feed Forward Network. Every token is processed completely independently. In the first linear layer, we project each token\'s vector to a higher-dimensional space of size thirty-two.',
        audienceQuestion: 'Is there any communication between tokens in this layer?',
        expectedAnswer: 'No, every token position is processed completely independently and in parallel.',
        misconception: 'Remind students that only channels mix here, not tokens.',
        transition: 'Let\'s activate these projections.'
      },
      {
        duration: '~35s',
        objective: 'Explain non-linear activation.',
        script: 'Next, we apply the non-linear activation function, ReLU. Watch how any negative values in our expanded vectors are completely suppressed to zero, while positive values pass through unchanged.',
        audienceQuestion: 'Why is it important to filter out negative values?',
        expectedAnswer: 'It introduces a non-linearity, allowing the model to learn complex, non-linear relationships.',
        misconception: 'Make sure students notice the red highlighted zeros in the active column.',
        transition: 'Let\'s project back to the model dimension.'
      },
      {
        duration: '~35s',
        objective: 'Explain the second FFN linear layer.',
        script: 'In the final step of the FFN, we project the activated representations back to the original model size of sixteen. This compresses the filtered information, ready to be added back to the main stream.',
        audienceQuestion: 'Why must the output size match the hidden dimension?',
        expectedAnswer: 'So that it can be added back to the residual stream via the skip connection.',
        misconception: 'The FFN weights (W1 and W2) are separate learned parameters.',
        transition: 'Let\'s review what changed in the FFN.'
      },
      {
        duration: '~20s',
        objective: 'Review FFN shapes and data flow.',
        script: 'Our Feed Forward processing is complete. Let\'s look at the summary.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Let\'s check our understanding of the Feed Forward Network.'
      },
      {
        duration: '~30s',
        objective: 'Assess FFN functionality.',
        script: 'Take a look at the quick check question. Think about how the FFN behaves across different positions.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Excellent. Now we proceed to adding the FFN output back to the residual stream.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Emphasize the division of labor in a Transformer block: Attention mixes tokens, FFN refines features.'
      ],
      misconceptions: [
        'Clarify that the FFN operates identically at every position, using the exact same weight matrices W1 and W2.'
      ],
      suggestedQuestions: [
        'How would the network behavior change if we replaced the non-linear activation with a linear mapping?'
      ]
    }
  },
  'layer-norm-2': {
    eyebrow: 'Feed-forward · Normalization',
    title: 'Layer Normalization ②',
    fourQuestions: {
      whatIsHappening: 'We normalize the activations of each token row independently across the d_model dimensions after the second skip connection, then scale with gamma and shift with beta.',
      why: 'This normalizes features after Feed Forward processing, stabilizing representations before they propagate to the next encoder block or the output layer.',
      whatChanged: 'Each row is standardized to zero mean and unit variance, preparing the representation for downstream stages.',
      whatToObserve: 'Audience can observe how calculations occur independently for each token row.',
    },
    body: {
      beginner: 'This is the final checkpoint of our transformer block. We take the output of our second skip connection and run it through Layer Normalization again to keep all numbers stable before sending them forward.',
      mtech: 'The second LayerNorm centers and scales features row-by-row: Y = gamma2 * X_norm + beta2, stabilizing final block outputs.',
      research: 'Normalizes block representations prior to block transitions, ensuring uniform activation boundaries across layers.',
    },
    deepDive: {
      math: 'Y = \\gamma_2 \\odot \\hat{X} + \\beta_2',
      complexity: 'O(seq\\_len \\times d\\_model) row-wise standard normalizations.',
      matrixEquivalence: 'Row-wise standardization keeping shape [seq_len, d_model] identical.',
      misconceptions: [
        'This LayerNorm uses its own separate scale (gamma2) and shift (beta2) parameters, distinct from LayerNorm ①.',
      ],
      notes: 'No dimension transformations occur.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why normalize at the end of the block?',
          body: 'Normalization stabilizes activations. In a deep network, stacking multiple layers causes representations to drift. Final normalization ensures consistent scaling at block boundaries.',
        }
      ],
      example: {
        left: [0.5, -0.4, 0.9, -1.0],
        right: [0.1, -0.8, 1.2, -1.5],
        caption: 'Values stabilized and scaled using LayerNorm ② weights.'
      }
    },
    beforeAfter: {
      before: { label: 'Residual Output ②', shape: null },
      after: { label: 'Block Normalized Output', shape: null },
      whatChanged: 'We standardized the second residual block output row-by-row.',
      structured: {
        entered: 'Residual Output ② matrix X of shape [seq_len, d_model].',
        happened: 'Normalized row-wise and scale-shifted with gamma2 and beta2.',
        changed: 'Output values are standardized to standard scale boundaries.',
        leaves: 'Final block output matrix Y of shape [seq_len, d_model].',
      }
    },
    quickCheck: {
      question: 'What is the purpose of the second Layer Normalization step at the block boundary?',
      choices: [
        'To stabilize features after FFN processing and ensure uniform scale before block output.',
        'To compute new attention scores for the next sequence block.',
        'To map token representations directly to vocabulary probability scores.',
        'To insert positional details into the FFN outputs.'
      ],
      correctIndex: 0,
      explanation: 'Correct! The second Layer Normalization step stabilizes and normalizes features after the FFN block skip connection, ensuring clean block boundaries.',
      transition: 'Now that the encoder block is complete, let\'s review the final block representation.',
      distractorNotes: {
        1: 'Incorrect. Attention scores are calculated inside the self-attention sub-layers, not LayerNorm.',
        2: 'Incorrect. Vocabulary projection is handled by the model classification head.',
        3: 'Incorrect. Positional information is only injected at the network inputs.'
      }
    },
    pytorch: [
      { id: 'code-layernorm-2', code: 'x = torch.nn.functional.layer_norm(x, x.shape[-1:])' }
    ],
    equationTerms: [
      { id: 'eq-layernorm-2', tex: 'Y = \\gamma_2 \\odot \\left( \\frac{X - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\right) + \\beta_2' }
    ],
    syncMap: [
      { vizElementId: 'viz-layernorm', equationTermId: 'eq-layernorm-2', codeLineId: 'code-layernorm-2' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain LayerNorm ② statistics.',
        script: 'After the second skip connection, we perform our final block normalization. We compute the row-wise mean and variance.',
        audienceQuestion: 'What does this second normalization prepare features for?',
        expectedAnswer: 'For entry into the next transformer block or classification layer.',
        misconception: 'Each LayerNorm has its own independent weights.',
        transition: 'Let\'s normalize the activations.'
      },
      {
        duration: '~35s',
        objective: 'Explain row-wise standardization ②.',
        script: 'We standardize each row value, centering features around zero mean and standard scale.',
        audienceQuestion: 'Does sequence length affect these statistics?',
        expectedAnswer: 'No, statistics are independent of sequence length.',
        misconception: 'Standardization limits unbounded feature drift.',
        transition: 'Let\'s scale and shift using gamma and beta.'
      },
      {
        duration: '~35s',
        objective: 'Explain scale and shift step ②.',
        script: 'We apply gamma two and beta two parameters, completing the encoder block calculations.',
        audienceQuestion: 'Are these parameters learned during training?',
        expectedAnswer: 'Yes, they are optimized to find the ideal feature distribution.',
        misconception: 'The output maintains shape d_model.',
        transition: 'Let\'s review what changed.'
      },
      {
        duration: '~20s',
        objective: 'Review second LayerNorm shapes and data flow.',
        script: 'We have normalized our final block outputs. Let\'s look at the summary.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Let\'s check our understanding.'
      },
      {
        duration: '~30s',
        objective: 'Assess second LayerNorm functionality.',
        script: 'Take a look at the quick check question. Think about the block boundary.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Excellent. This completes the encoder block normalization.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Explain that the second LayerNorm acts as a boundary guard, keeping values clean before they enter next encoder blocks.'
      ],
      misconceptions: [
        'Emphasize that gamma2 and beta2 are separate variables from gamma1 and beta1.'
      ],
      suggestedQuestions: [
        'How does pre-layer normalization differ from post-layer normalization?'
      ]
    }
  },
  'encoder-output': {
    eyebrow: 'Summary · Contextual Embeddings',
    title: 'Encoder Output',
    fourQuestions: {
      whatIsHappening: 'We take the final contextualized token representations produced after LayerNorm-2 and present them as the final output of the Encoder block.',
      why: 'This block transforms standard input embeddings into deep contextual representations that capture word meanings relative to the entire sentence structure.',
      whatChanged: 'Input tokens entered as isolated representations; they leave as contextual vectors where each row integrates details from the entire sentence.',
      whatToObserve: 'Audience can observe how changing the sentence or configuration alters all embedding values, reflecting the global context built by the Attention head.',
    },
    body: {
      beginner: 'The Encoder Output is the final report card of the encoder block. Instead of just knowing what each word means in a dictionary, the encoder has looked at the whole sentence. Every word vector has now been adjusted to include the context of the words around it, making it ready to be used for translation or sorting tasks.',
      mtech: 'The encoder output represents the final contextual embedding matrix of shape [seq_len, d_model] outputted by LayerNorm-2. These outputs are sent to the decoder stack or classification heads.',
      research: 'The encoder output corresponds to the final representations of the encoder block. It captures deep contextual interactions via self-attention followed by position-wise feed-forward processing.',
    },
    deepDive: {
      math: 'H_{\\text{out}} = \\text{LayerNorm}_2(\\text{FFN}(X_{LN1}) + X_{LN1})',
      complexity: 'Identity transformation. Consumes the precomputed output of LayerNorm-2.',
      matrixEquivalence: 'No mathematical transformations occur. Shape remains [seq_len, d_model].',
      misconceptions: [
        'The output is not a single vector, but a full matrix where each row represents a token in the sequence with its surrounding context.',
      ],
      notes: 'This output acts as key/value matrices for cross-attention in the decoder.',
    },
    whyPanel: {
      items: [
        {
          title: 'What does "contextual representation" mean?',
          body: 'In static embeddings, the word "bank" has the same vector whether it means a river bank or a money bank. After passing through the encoder block, self-attention alters the vector of "bank" based on other tokens in the sentence like "river" or "deposit".',
        },
        {
          title: 'Where do these outputs go next?',
          body: 'They can go to the next encoder block in a stack, to the cross-attention layer of a decoder block for translation, or to a classification head (like pooling the representations) to classify sentence sentiment.',
        }
      ],
      example: {
        left: [0.12, 0.85, -0.42, 1.15],
        right: [0.95, -0.12, 0.44, 0.82],
        caption: 'Encoder output representations for "chased" incorporating context from both "cat" and "dog".'
      }
    },
    beforeAfter: {
      before: { label: 'Input Embeddings', shape: null },
      after: { label: 'Final Contextual Embeddings', shape: null },
      whatChanged: 'Tokens now contain deep bidirectional context.',
      structured: {
        entered: 'Isolated word/positional embeddings X.',
        happened: 'Self-Attention mixed token features, followed by non-linear Feed Forward transformation.',
        changed: 'Activations represent the whole sentence context per token.',
        leaves: 'Deep contextual representations Y of shape [seq_len, d_model].',
      }
    },
    quickCheck: {
      question: 'What characterizes the vectors in the final Encoder Output matrix compared to the input embeddings?',
      choices: [
        'Each token vector now includes contextual information gathered from the entire sentence.',
        'They are larger in dimension than the input embeddings.',
        'They are converted into probability values summing to 1.',
        'They contain only positional information without semantic word data.'
      ],
      correctIndex: 0,
      explanation: 'Correct! The self-attention mechanism mixes token features across the sequence dimension, meaning each output vector represents a token in the context of the entire sentence.',
      transition: 'This completes the Transformer Encoder block visualization! We are now ready to summarize the entire block flow.',
      distractorNotes: {
        1: 'Incorrect. The model dimension remains unchanged.',
        2: 'Incorrect. These are hidden representations, not probabilities.',
        3: 'Incorrect. Position and semantic data are both preserved and mixed.'
      }
    },
    pytorch: [
      { id: 'code-encoder-output', code: 'outputs = encoder(input_ids) # outputs.shape: [batch, seq_len, d_model]' }
    ],
    equationTerms: [
      { id: 'eq-encoder-output', tex: 'X_{\\text{context}} = \\text{LayerNorm}_2(X_{\\text{residual2}})' }
    ],
    syncMap: [
      { vizElementId: 'viz-encoder-output', equationTermId: 'eq-encoder-output', codeLineId: 'code-encoder-output' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Introduce final encoder representations.',
        script: 'We have reached the end of the encoder block. The final matrix represents our tokens after processing through attention, FFN, and normalization.',
        audienceQuestion: 'Is the sequence length or dimension modified here?',
        expectedAnswer: 'No, the shape is preserved, but values now reflect contextual relationships.',
        misconception: 'Make sure students see that every token has updated values.',
        transition: 'Let\'s see how context is integrated.'
      },
      {
        duration: '~35s',
        objective: 'Explain contextual embedding.',
        script: 'Look at the highlighted row. Unlike the static input embeddings we started with, these vectors have gathered contextual clues from every other token in the sentence.',
        audienceQuestion: 'Why is bidirectional context important?',
        expectedAnswer: 'It helps the model resolve ambiguous words and understand syntax correctly.',
        misconception: 'Point out how self-attention weights drove this blending.',
        transition: 'Let\'s review where this representation is used.'
      },
      {
        duration: '~35s',
        objective: 'Explain applications.',
        script: 'This contextual matrix can now be fed into stacked encoder blocks, passed to a decoder for machine translation, or sent to classification heads.',
        audienceQuestion: 'Can we stack multiple encoder blocks?',
        expectedAnswer: 'Yes, modern transformers stack many blocks to learn increasingly abstract representations.',
        misconception: 'The format remains fully compatible across all blocks.',
        transition: 'Let\'s review what changed in the complete encoder block.'
      },
      {
        duration: '~20s',
        objective: 'Review block shapes and data flow.',
        script: 'The complete encoder block transformations are summarized here.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Let\'s verify our understanding.'
      },
      {
        duration: '~30s',
        objective: 'Assess encoder output functionality.',
        script: 'Take a look at the quick check question. Think about how the representations changed.',
        audienceQuestion: null,
        expectedAnswer: null,
        misconception: null,
        transition: 'Excellent. This completes the Transformer Encoder block.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Use this scene to summarize the entire block, tracing the journey of a token from static embedding to contextual output.'
      ],
      misconceptions: [
        'Ensure students understand that the encoder output is ready to be consumed directly by other layers.'
      ],
      suggestedQuestions: [
        'How does this contextual representation benefit downstream classification tasks?'
      ]
    }
  },
  'dec-embedding': {
    eyebrow: 'Decoder Setup · Word Embedding',
    title: 'Decoder Word Embedding',
    fourQuestions: {
      whatIsHappening: 'We look up vector representations for target language tokens in the decoder vocabulary.',
      why: 'The decoder requires high-dimensional vector representations of the target tokens (shifted right) to compute auto-regressive or sequence predictions.',
      whatChanged: 'Target token IDs are mapped to dense d_model vectors.',
      whatToObserve: 'Audience can observe how target token lookup functions identically to encoder embedding lookup.',
    },
    body: {
      beginner: 'Just like the encoder converted input words into numerical vectors, the decoder converts target words (or start-of-sequence tokens) into vectors so it can begin generating translations or predictions.',
      mtech: 'Target tokens Y are mapped to continuous embedding vectors E_dec[Y] of shape [target_seq_len, d_model]. During training, these are shifted right by one position.',
      research: 'The decoder embedding matrix maps target token indices to d_model dimensional space. Often, the encoder and decoder share the same tied weight embedding matrix.',
    },
    deepDive: {
      math: 'X_{\\text{dec}} = \\text{Embedding}(Y)',
      complexity: 'O(target\\_seq\\_len \\times d\\_model) index lookups.',
      matrixEquivalence: 'Row lookup from target embedding matrix.',
      misconceptions: [
        'The decoder input during inference is built auto-regressively token-by-token, whereas during training it is fed as a shifted target sequence.',
      ],
      notes: 'd_model = 16.',
    },
    whyPanel: {
      items: [
        {
          title: 'Why does the decoder need its own input embeddings?',
          body: 'The decoder processes target tokens (or previous output tokens) to generate the next word in the sequence, requiring dense representations in the same model dimension.',
        }
      ],
      example: {
        left: [1, 5, 8],
        right: [[0.1, -0.4, 0.2], [0.8, 0.2, -0.1], [-0.5, 0.9, 0.3]],
        caption: 'Target token IDs converted into vector rows.'
      }
    },
    beforeAfter: {
      before: { label: 'Target Token IDs', shape: null },
      after: { label: 'Decoder Embeddings', shape: null },
      whatChanged: 'Target token IDs converted to dense vectors.',
      structured: {
        entered: 'Target token IDs of shape [target_seq_len].',
        happened: 'Matrix lookup from target embedding table.',
        changed: 'Discrete IDs converted to d_model dimensional space.',
        leaves: 'Target embedding matrix of shape [target_seq_len, d_model].',
      }
    },
    quickCheck: {
      question: 'In sequence-to-sequence Transformers, how are target tokens fed to the decoder during training?',
      choices: [
        'As a target sequence shifted right by one position (with a start-of-sequence token).',
        'Directly mixed with the encoder input tokens in the same matrix.',
        'As one-hot encoded scalar values without embedding lookup.',
        'Across the batch dimension as fixed positional indices.'
      ],
      correctIndex: 0,
      explanation: 'Correct! During training, target tokens are shifted right by one position (prepended with <sos>) so the decoder learns to predict the next token.',
      transition: 'Next, let\'s add positional information to the decoder embeddings.',
      distractorNotes: {
        1: 'Incorrect. Decoder inputs are processed in their own sub-block.',
        2: 'Incorrect. Embedding lookups convert discrete IDs into continuous vectors.',
        3: 'Incorrect. Tokens are embedded into the model dimension.'
      }
    },
    pytorch: [
      { id: 'code-dec-embedding', code: 'dec_embeds = self.dec_embedding(target_ids)' }
    ],
    equationTerms: [
      { id: 'eq-dec-embedding', tex: 'X_{\\text{dec}} = E_{\\text{dec}}[Y]' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-embedding', equationTermId: 'eq-dec-embedding', codeLineId: 'code-dec-embedding' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Introduce decoder word embeddings.',
        script: 'Welcome to the Transformer Decoder. Just like the encoder, the decoder starts by looking up vector representations for its target tokens.',
        audienceQuestion: 'Are decoder embeddings formatted the same as encoder embeddings?',
        expectedAnswer: 'Yes, both produce vectors in the model dimension d_model.',
        misconception: 'Remind students that target sequences are shifted during training.',
        transition: "Let's proceed to positional encoding."
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Explain that the decoder takes target tokens as input to learn auto-regressive generation.'
      ],
      misconceptions: [
        'Target tokens are independent vectors before positional encodings are added.'
      ],
      suggestedQuestions: [
        'Why do translation models shift target tokens to the right?'
      ]
    }
  },
  'dec-positional-enc': {
    eyebrow: 'Decoder Setup · Positional Encoding',
    title: 'Decoder Positional Encoding',
    fourQuestions: {
      whatIsHappening: 'We add sinusoidal positional encodings to the target token embeddings.',
      why: 'Without positional encodings, the decoder would have no sense of word order in the target sequence.',
      whatChanged: 'Target embeddings now carry position information along the sequence.',
      whatToObserve: 'Audience can observe how sinusoidal position vectors are added element-wise to target embeddings.',
    },
    body: {
      beginner: 'Just as in the encoder, words in the target sentence need order. We add a unique positional pattern to each target word vector so the decoder knows which word comes first, second, and so on.',
      mtech: 'Decoder positional encodings PE_dec are added to target embeddings: X_pe = X_dec + PE_dec. This informs masked self-attention of temporal sequence order.',
      research: 'Applies fixed sinusoidal or learned positional encodings to target representations, establishing temporal topology for auto-regressive predictions.',
    },
    deepDive: {
      math: 'X_{\\text{dec\\_pe}} = X_{\\text{dec}} + PE',
      complexity: 'O(target\\_seq\\_len \\times d\\_model) element-wise additions.',
      matrixEquivalence: 'Element-wise addition of target PE matrix to target embedding matrix.',
      misconceptions: [
        'Decoder positional encodings use the exact same formula and dimensions as encoder positional encodings.',
      ],
      notes: 'Preserves shape [target_seq_len, d_model].',
    },
    whyPanel: {
      items: [
        {
          title: 'Why does the decoder need positional encoding?',
          body: 'Order matters in output generation. Adding positional encodings ensures that target word positions are distinguished during self-attention and cross-attention.',
        }
      ],
      example: {
        left: [0.2, -0.1, 0.4],
        right: [0.0, 1.0, 0.0],
        caption: 'Positional encoding vector added element-wise to target word embedding.'
      }
    },
    beforeAfter: {
      before: { label: 'Decoder Embeddings', shape: null },
      after: { label: 'Decoder Input + PE', shape: null },
      whatChanged: 'Position signals combined with target word representations.',
      structured: {
        entered: 'Target embedding matrix of shape [target_seq_len, d_model].',
        happened: 'Element-wise addition of sinusoidal positional table.',
        changed: 'Target vectors now encode both word identity and sequence order.',
        leaves: 'Decoder input matrix X_pe of shape [target_seq_len, d_model].',
      }
    },
    quickCheck: {
      question: 'What is the mathematical shape of the decoder representation after adding positional encoding?',
      choices: [
        '[target_seq_len, d_model] — the shape remains unchanged after element-wise addition.',
        '[target_seq_len, 2 * d_model] — the dimension doubles to store position.',
        '[d_model, d_model] — a square representation matrix.',
        '[batch_size, target_seq_len] — scalar sequence indices.'
      ],
      correctIndex: 0,
      explanation: 'Correct! Adding positional encodings is an element-wise addition, so the shape remains [target_seq_len, d_model].',
      transition: 'Now the decoder input is ready to enter Masked Self-Attention.',
      distractorNotes: {
        1: 'Incorrect. Positional encoding is added, not concatenated.',
        2: 'Incorrect. The sequence length is preserved.',
        3: 'Incorrect. Vectors are dense embeddings.'
      }
    },
    pytorch: [
      { id: 'code-dec-positional-enc', code: 'dec_x = dec_embeds + self.pos_encoder(dec_embeds)' }
    ],
    equationTerms: [
      { id: 'eq-dec-positional-enc', tex: 'X_{\\text{dec\\_pe}} = X_{\\text{dec}} + PE' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-positional-enc', equationTermId: 'eq-dec-positional-enc', codeLineId: 'code-dec-positional-enc' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain decoder positional encoding.',
        script: 'We now add positional encoding vectors to our target word embeddings. This gives the decoder the position information it needs to predict words in order.',
        audienceQuestion: 'Is the positional encoding formula different for the decoder?',
        expectedAnswer: 'No, it uses the exact same sinusoidal formula as the encoder.',
        misconception: 'Make sure students see that position vectors are added element-wise.',
        transition: 'Decoder input setup is complete.'
      }
    ],
    speakerNotes: {
      teachingTips: [
        'Emphasize that positional encoding ensures sequence order is preserved across target tokens.'
      ],
      misconceptions: [
        'Positional encoding alters vector values without changing tensor dimensions.'
      ]
    }
  },
  'dec-proj-q': {
    eyebrow: 'Decoder Self-Attention · Query Projection',
    title: 'Decoder Projection to Q',
    fourQuestions: {
      whatIsHappening: 'The decoder positional representations X_dec_pe are multiplied by weight matrix W_q_dec to produce Query vectors for the target sequence.',
      why: 'Query vectors represent what each target token is asking or searching for within the target sequence context.',
      whatChanged: 'Input matrix [target_seq_len, d_model] is linearly projected to Query matrix Q_dec of shape [target_seq_len, d_model].',
      whatToObserve: 'Observe how each target token is projected through W_q_dec into a specialized Query vector.'
    },
    body: {
      beginner: 'To let words in the target sentence talk to each other, each word first creates a "Query" vector — a list of questions that the word is asking about surrounding target words.',
      mtech: 'The decoder input X_dec_pe ∈ R^(L_dec × d_model) is multiplied by learned weight matrix W_(q,dec) ∈ R^(d_model × d_model), producing Query matrix Q_dec ∈ R^(L_dec × d_model).',
      research: 'Linear projection into Query space allows the network to learn a parameter-driven subspace for target self-attention matching, parameterized by W_q ∈ R^(d_model × d_model).'
    },
    deepDive: {
      math: 'Q_{\\text{dec}} = X_{\\text{dec\\_pe}} W_{q,\\text{dec}} + b_{q,\\text{dec}}',
      complexity: 'O(L_{\\text{dec}} \\cdot d_{\\text{model}}^2) floating point operations.',
      matrixEquivalence: 'Linear projection maps each token vector row independently from model space to query space.',
      misconceptions: [
        'Query projection in decoder self-attention uses target tokens, unlike cross-attention queries which use decoder outputs and keys from the encoder.'
      ],
      notes: 'd_model = 16 in this visualization; production LLMs use 4096-12288.'
    },
    whyPanel: {
      items: [
        {
          title: 'Why project to Query space instead of using raw embeddings?',
          body: 'Raw embeddings carry general word identity. Projecting through W_q creates a specialized vector representation focused specifically on what information this token needs from target context.'
        }
      ],
      example: {
        left: [0.3, -0.1, 0.5],
        right: [0.8, 0.2, -0.4],
        caption: 'Target embedding transformed by W_q into a Query vector.'
      }
    },
    beforeAfter: {
      before: { label: 'Decoder Input X_dec_pe', shape: null },
      after: { label: 'Decoder Queries Q_dec', shape: null },
      whatChanged: 'Target representations mapped into the Query subspace.',
      structured: {
        entered: 'Decoder input matrix of shape [target_seq_len, d_model].',
        happened: 'Matrix multiplication with W_q_dec plus bias b_q.',
        changed: 'Dense input vectors transformed into Query space vectors.',
        leaves: 'Query matrix Q_dec of shape [target_seq_len, d_model].'
      }
    },
    quickCheck: {
      question: 'Which weight matrix is used to compute Decoder Self-Attention Queries?',
      choices: [
        'W_q_dec of shape [d_model, d_model]',
        'W_k_dec of shape [d_model, d_model]',
        'W_v_dec of shape [d_model, d_model]',
        'Wo_dec of shape [d_model, d_model]'
      ],
      correctIndex: 0,
      explanation: 'W_q_dec maps input token vectors to Query vectors.',
      transition: 'Now let\'s compute the Key vectors for the decoder sequence.',
      distractorNotes: {
        1: 'Incorrect. W_k_dec creates Key vectors.',
        2: 'Incorrect. W_v_dec creates Value vectors.',
        3: 'Incorrect. Wo_dec projects the final concatenated attention output.'
      }
    },
    pytorch: [
      { id: 'code-dec-proj-q', code: 'Q_dec = self.W_q(dec_input)' }
    ],
    equationTerms: [
      { id: 'eq-dec-proj-q', tex: 'Q_{\\text{dec}} = X_{\\text{dec\\_pe}} W_{q,\\text{dec}}' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-proj-q', equationTermId: 'eq-dec-proj-q', codeLineId: 'code-dec-proj-q' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Explain Decoder Query projection.',
        script: 'We multiply the decoder input vectors by W_q_dec to produce Query vectors Q_dec. These vectors represent what each target token needs from previous tokens.',
        audienceQuestion: 'Are decoder queries computed for all target tokens at once during training?',
        expectedAnswer: 'Yes, during training all target token queries are computed in parallel.',
        misconception: 'Clarify that decoder self-attention queries operate on target sequence tokens.',
        transition: 'Next, let\'s project target tokens to Keys.'
      }
    ],
    speakerNotes: {
      teachingTips: ['Point out that Q_dec is the first of three linear projections in self-attention.'],
      misconceptions: ['Decoder self-attention uses target sequence tokens for Q, K, and V.']
    }
  },
  'dec-proj-k': {
    eyebrow: 'Decoder Self-Attention · Key Projection',
    title: 'Decoder Projection to K',
    fourQuestions: {
      whatIsHappening: 'The decoder input X_dec_pe is multiplied by weight matrix W_k_dec to produce Key vectors K_dec for the target sequence.',
      why: 'Key vectors represent the information or attributes that each target token offers to answer incoming queries.',
      whatChanged: 'Input matrix [target_seq_len, d_model] is linearly projected to Key matrix K_dec of shape [target_seq_len, d_model].',
      whatToObserve: 'Watch each target token pass through W_k_dec to materialize its corresponding Key vector.'
    },
    body: {
      beginner: 'Every word in the target sentence also creates a "Key" vector — the answers or index tags that other words check against.',
      mtech: 'K_dec = X_dec_pe W_(k,dec) + b_(k,dec) ∈ R^(L_dec × d_model). Key vectors encode feature keys used during dot-product attention.',
      research: 'Key projection maps target representations into a matching space compatible with Query vectors, enabling dot-product affinity scoring.'
    },
    deepDive: {
      math: 'K_{\\text{dec}} = X_{\\text{dec\\_pe}} W_{k,\\text{dec}} + b_{k,\\text{dec}}',
      complexity: 'O(L_{\\text{dec}} \\cdot d_{\\text{model}}^2) FLOPs.',
      matrixEquivalence: 'Linear transformation of input rows by W_k_dec.',
      misconceptions: [
        'Decoder self-attention Keys come from the target sequence, whereas cross-attention Keys come from the encoder.'
      ],
      notes: 'Shape: [target_seq_len, d_model].'
    },
    whyPanel: {
      items: [
        {
          title: 'Why do we need separate Query and Key matrices?',
          body: 'If Q and K were the same, attention would be symmetric (word A attending to B would equal B attending to A). Separate Q and K projections allow asymmetric relationships, where A can look for B without B needing to look for A.'
        }
      ],
      example: {
        left: [0.1, 0.4, -0.2],
        right: [-0.3, 0.6, 0.1],
        caption: 'Target input converted by W_k into a Key vector.'
      }
    },
    beforeAfter: {
      before: { label: 'Decoder Input X_dec_pe', shape: null },
      after: { label: 'Decoder Keys K_dec', shape: null },
      whatChanged: 'Target representations projected to Key space.',
      structured: {
        entered: 'Decoder input matrix of shape [target_seq_len, d_model].',
        happened: 'Matrix multiplication with W_k_dec.',
        changed: 'Dense input vectors transformed into Key vectors.',
        leaves: 'Key matrix K_dec of shape [target_seq_len, d_model].'
      }
    },
    quickCheck: {
      question: 'What role do Key vectors play in Decoder Self-Attention?',
      choices: [
        'They act as tags/labels that Queries are matched against via dot product.',
        'They contain the final output text probabilities.',
        'They normalize activation variances.',
        'They compress sequence length to 1.'
      ],
      correctIndex: 0,
      explanation: 'Queries are multiplied with Keys to calculate dot-product attention scores.',
      transition: 'Now let\'s project the Value vectors for the decoder sequence.',
      distractorNotes: {
        1: 'Incorrect. Softmax projection computes output probabilities.',
        2: 'Incorrect. LayerNorm normalizes variances.',
        3: 'Incorrect. Sequence length is preserved.'
      }
    },
    pytorch: [
      { id: 'code-dec-proj-k', code: 'K_dec = self.W_k(dec_input)' }
    ],
    equationTerms: [
      { id: 'eq-dec-proj-k', tex: 'K_{\\text{dec}} = X_{\\text{dec\\_pe}} W_{k,\\text{dec}}' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-proj-k', equationTermId: 'eq-dec-proj-k', codeLineId: 'code-dec-proj-k' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Explain Decoder Key projection.',
        script: 'We project target tokens through W_k_dec into Key vectors K_dec. Keys act as index labels for matching against Queries.',
        audienceQuestion: 'Are Key vectors the same size as Query vectors?',
        expectedAnswer: 'Yes, both have dimension d_model.',
        misconception: 'Remind students that Q and K must have matching feature dimensions for dot products.',
        transition: 'Next, let\'s create the Value vectors.'
      }
    ],
    speakerNotes: {
      teachingTips: ['Emphasize that Q and K live in compatible spaces so their dot product is meaningful.'],
      misconceptions: ['Decoder self-attention Keys are distinct from encoder cross-attention Keys.']
    }
  },
  'dec-proj-v': {
    eyebrow: 'Decoder Self-Attention · Value Projection',
    title: 'Decoder Projection to V',
    fourQuestions: {
      whatIsHappening: 'The decoder input X_dec_pe is multiplied by weight matrix W_v_dec to produce Value vectors V_dec for the target sequence.',
      why: 'Value vectors contain the actual semantic content that will be extracted and weighted during attention blending.',
      whatChanged: 'Input matrix [target_seq_len, d_model] is linearly projected to Value matrix V_dec of shape [target_seq_len, d_model].',
      whatToObserve: 'Observe each target token transform through W_v_dec into a Value vector.'
    },
    body: {
      beginner: 'Finally, each target word creates a "Value" vector — the actual content that gets passed along after attention weights decide how much focus to give each word.',
      mtech: 'V_dec = X_dec_pe W_(v,dec) + b_(v,dec) ∈ R^(L_dec × d_model). Value vectors hold the contextual payload.',
      research: 'Value projection decouples the attention matching mechanism (Q, K) from the information payload (V), enabling flexible feature routing.'
    },
    deepDive: {
      math: 'V_{\\text{dec}} = X_{\\text{dec\\_pe}} W_{v,\\text{dec}} + b_{v,\\text{dec}}',
      complexity: 'O(L_{\\text{dec}} \\cdot d_{\\text{model}}^2) FLOPs.',
      matrixEquivalence: 'Linear transformation of input rows by W_v_dec.',
      misconceptions: [
        'Value vectors are not attention weights — attention weights multiply Value vectors.'
      ],
      notes: 'Shape: [target_seq_len, d_model].'
    },
    whyPanel: {
      items: [
        {
          title: 'Why separate Value vectors from Key vectors?',
          body: 'Keys are optimized for matching (answering "should I pay attention to this?"), whereas Values are optimized for representation (answering "what information should I send?"). Separating them gives the model much greater expressive power.'
        }
      ],
      example: {
        left: [-0.2, 0.5, 0.1],
        right: [0.4, -0.3, 0.7],
        caption: 'Target input transformed by W_v into a Value vector.'
      }
    },
    beforeAfter: {
      before: { label: 'Decoder Input X_dec_pe', shape: null },
      after: { label: 'Decoder Values V_dec', shape: null },
      whatChanged: 'Target representations projected to Value space.',
      structured: {
        entered: 'Decoder input matrix of shape [target_seq_len, d_model].',
        happened: 'Matrix multiplication with W_v_dec.',
        changed: 'Dense input vectors transformed into Value vectors.',
        leaves: 'Value matrix V_dec of shape [target_seq_len, d_model].'
      }
    },
    quickCheck: {
      question: 'What is the primary function of Value vectors in Decoder Self-Attention?',
      choices: [
        'They hold the content that gets weighted and summed to form the attention output.',
        'They determine the causal mask pattern.',
        'They normalize token features across sequence length.',
        'They compute vocabulary logits.'
      ],
      correctIndex: 0,
      explanation: 'Attention probabilities multiply Value vectors to produce the weighted sum output.',
      transition: 'Now we have Q, K, and V. Let\'s split them into multiple heads!',
      distractorNotes: {
        1: 'Incorrect. Causal masks are fixed lower-triangular matrices.',
        2: 'Incorrect. LayerNorm normalizes features.',
        3: 'Incorrect. Vocabulary projection computes logits.'
      }
    },
    pytorch: [
      { id: 'code-dec-proj-v', code: 'V_dec = self.W_v(dec_input)' }
    ],
    equationTerms: [
      { id: 'eq-dec-proj-v', tex: 'V_{\\text{dec}} = X_{\\text{dec\\_pe}} W_{v,\\text{dec}}' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-proj-v', equationTermId: 'eq-dec-proj-v', codeLineId: 'code-dec-proj-v' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Explain Decoder Value projection.',
        script: 'We project target tokens through W_v_dec into Value vectors V_dec. These vectors hold the semantic content to be retrieved.',
        audienceQuestion: 'Are Q, K, and V projected using three separate weight matrices?',
        expectedAnswer: 'Yes: W_q, W_k, and W_v.',
        misconception: 'Ensure students understand that Q, K, and V are distinct projections from the same input.',
        transition: 'Next, let\'s split Q, K, and V into multiple attention heads.'
      }
    ],
    speakerNotes: {
      teachingTips: ['Reiterate the Q/K/V analogy: Q = query, K = key/index, V = value/content.'],
      misconceptions: ['Value vectors hold representations, not attention scores.']
    }
  },
  'dec-split-heads': {
    eyebrow: 'Decoder Self-Attention · Head Splitting',
    title: 'Decoder Split Heads',
    fourQuestions: {
      whatIsHappening: 'The Query, Key, and Value matrices of shape [target_seq_len, d_model] are reshaped and transposed into num_heads separate tensors of shape [target_seq_len, d_k].',
      why: 'Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.',
      whatChanged: 'Tensors go from [target_seq_len, d_model] to [num_heads, target_seq_len, d_k], where d_k = d_model / num_heads.',
      whatToObserve: 'Watch the single 16-dimensional matrix split into 4 parallel 4-dimensional head slices.'
    },
    body: {
      beginner: 'Instead of using one big 16-number vector for attention, we split it into 4 smaller 4-number heads so each head can focus on a different type of relationship in the sentence.',
      mtech: 'Q_dec, K_dec, V_dec ∈ R^(L × d_model) → reshape → R^(L × h × d_k) → transpose → R^(h × L × d_k).',
      research: 'Head splitting partitions d_model into h parallel d_k subspaces. Vaswani et al. (2017) set d_k = d_v = d_model / h = 64 for 8 heads in Transformer-base.'
    },
    deepDive: {
      math: 'd_k = d_{\\text{model}} / h, \\quad Q_h \\in \\mathbb{R}^{h \\times L_{\\text{dec}} \\times d_k}',
      complexity: 'O(1) memory view / zero-copy transpose in modern frameworks.',
      matrixEquivalence: 'Reshaping and axis swapping without arithmetic operations.',
      misconceptions: [
        'Head splitting does not alter any numerical values; it simply changes tensor view dimensions.'
      ],
      notes: 'In this demo: h = 4 heads, d_k = 16 / 4 = 4.'
    },
    whyPanel: {
      items: [
        {
          title: 'Why use multiple smaller heads instead of one large head?',
          body: 'A single attention head can only produce one attention pattern per token. Multi-head attention allows Head 1 to track subject-verb agreements while Head 2 tracks noun-modifier relationships, all in parallel.'
        }
      ],
      example: {
        left: [1, 16],
        right: [4, 1, 4],
        caption: '16-dim vector reshaped into 4 heads of size 4.'
      }
    },
    beforeAfter: {
      before: { label: 'Un-split Matrix Q_dec', shape: null },
      after: { label: 'Split Head Tensors Q_h', shape: null },
      whatChanged: 'Reshaped and transposed into num_heads parallel head slices.',
      structured: {
        entered: 'Q, K, V matrices of shape [target_seq_len, d_model].',
        happened: 'Reshape to [target_seq_len, num_heads, d_k] and transpose dimensions.',
        changed: 'Single wide representation split into parallel head channels.',
        leaves: 'Head tensors of shape [num_heads, target_seq_len, d_k].'
      }
    },
    quickCheck: {
      question: 'If d_model = 16 and num_heads = 4, what is the head dimension d_k?',
      choices: [
        'd_k = 4 (because 16 / 4 = 4)',
        'd_k = 16 (remains unchanged)',
        'd_k = 64 (16 * 4)',
        'd_k = 8 (16 / 2)'
      ],
      correctIndex: 0,
      explanation: 'd_k = d_model / num_heads = 16 / 4 = 4.',
      transition: 'Now let\'s compute the raw attention scores Q × Kᵀ for each head.',
      distractorNotes: {
        1: 'Incorrect. Un-split dimension is 16.',
        2: 'Incorrect. Dimension is divided, not multiplied.',
        3: 'Incorrect. 16 divided by 4 is 4, not 8.'
      }
    },
    pytorch: [
      { id: 'code-dec-split-heads', code: 'Q_h = Q.view(B, L, h, d_k).transpose(1, 2)' }
    ],
    equationTerms: [
      { id: 'eq-dec-split-heads', tex: 'Q_h \\in \\mathbb{R}^{h \\times L_{\\text{dec}} \\times d_k}' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-split-heads', equationTermId: 'eq-dec-split-heads', codeLineId: 'code-dec-split-heads' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Explain decoder head splitting.',
        script: 'We reshape Q, K, and V from d_model into num_heads slices of dimension d_k. This lets 4 heads operate in parallel.',
        audienceQuestion: 'Does head splitting require any mathematical multiplication?',
        expectedAnswer: 'No, it is just a tensor reshape and transpose operation.',
        misconception: 'Clarify that head splitting reorganizes tensor memory without changing values.',
        transition: 'Next, we compute raw attention scores Q × Kᵀ.'
      }
    ],
    speakerNotes: {
      teachingTips: ['Show how the 16 columns divide neatly into 4 groups of 4.'],
      misconceptions: ['Splitting heads does not add parameters or compute; it reorganizes existing parameters.']
    }
  },
  'dec-qk-matmul': {
    eyebrow: 'Decoder Self-Attention · Raw Scores',
    title: 'Decoder Q × Kᵀ (Raw Scores)',
    fourQuestions: {
      whatIsHappening: 'For each head, we multiply Query matrix Q_h by transposed Key matrix K_hᵀ and scale by 1/√d_k to get raw attention scores S.',
      why: 'Dot products measure the similarity/affinity between each Query vector and every Key vector in the sequence.',
      whatChanged: 'Tensors Q_h [L, d_k] and K_h [L, d_k] multiply to form a square score grid S of shape [target_seq_len, target_seq_len].',
      whatToObserve: 'Observe the square matrix of raw dot product scores before any masking or softmax is applied.'
    },
    body: {
      beginner: 'We multiply every Query by every Key to see how strongly each target word relates to every other target word.',
      mtech: 'S_dec = (Q_h K_h^T) / sqrt(d_k) ∈ R^(L_dec × L_dec). Scaling by 1/sqrt(d_k) prevents dot products from growing excessively large for larger d_k.',
      research: 'Scaled dot-product attention score computation. For large d_k, dot products grow in magnitude, pushing softmax into regions with extremely small gradients. Scaling counteracts this variance scaling.'
    },
    deepDive: {
      math: 'S_{\\text{raw}} = \\frac{Q_h K_h^T}{\\sqrt{d_k}} \\in \\mathbb{R}^{L_{\\text{dec}} \\times L_{\\text{dec}}}',
      complexity: 'O(h \\cdot L_{\\text{dec}}^2 \\cdot d_k) FLOPs.',
      matrixEquivalence: 'Batch matrix multiplication of [L, d_k] by [d_k, L].',
      misconceptions: [
        'These are raw scores — notice that at this step, future tokens CAN still see past tokens in the grid until causal masking is applied next!'
      ],
      notes: 'Scaling factor = 1 / sqrt(4) = 0.5 in our d_k=4 setup.'
    },
    whyPanel: {
      items: [
        {
          title: 'Why scale dot products by 1 / sqrt(d_k)?',
          body: 'Assuming Q and K have mean 0 and variance 1, their dot product has mean 0 and variance d_k. For large d_k, dot products become large in magnitude, causing softmax to saturate with near-zero gradients. Dividing by sqrt(d_k) pulls variance back to 1.'
        }
      ],
      example: {
        left: [2.0, -1.0],
        right: [1.5, 3.0],
        caption: 'Dot product = 0.0, scaled by 1/sqrt(4) = 0.5.'
      }
    },
    beforeAfter: {
      before: { label: 'Query & Key Head Tensors', shape: null },
      after: { label: 'Raw Score Grid S', shape: null },
      whatChanged: 'Matrix multiplication produced an L × L grid of raw attention scores.',
      structured: {
        entered: 'Q_h and K_h of shape [L_dec, d_k].',
        happened: 'Matmul Q_h * K_h^T divided by sqrt(d_k).',
        changed: 'Vector pairs converted to pairwise scalar affinity scores.',
        leaves: 'Square raw score matrix of shape [target_seq_len, target_seq_len].'
      }
    },
    quickCheck: {
      question: 'Why do we scale dot product scores by 1 / sqrt(d_k)?',
      choices: [
        'To prevent large values from saturating the Softmax and causing vanishing gradients.',
        'To apply causal masking to future tokens.',
        'To convert negative numbers to positive numbers.',
        'To reduce the sequence length.'
      ],
      correctIndex: 0,
      explanation: 'Scaling by 1 / sqrt(d_k) keeps score variance at 1, preventing Softmax saturation.',
      transition: 'Now comes the critical decoder step: Causal Masking!',
      distractorNotes: {
        1: 'Incorrect. Causal masking is a separate step added next.',
        2: 'Incorrect. Softmax exponentiation handles negative numbers.',
        3: 'Incorrect. Sequence length is unchanged.'
      }
    },
    pytorch: [
      { id: 'code-dec-qk-matmul', code: 'scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(d_k)' }
    ],
    equationTerms: [
      { id: 'eq-dec-qk-matmul', tex: 'S_{\\text{dec}} = \\frac{Q_h K_h^T}{\\sqrt{d_k}}' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-qk-matmul', equationTermId: 'eq-dec-qk-matmul', codeLineId: 'code-dec-qk-matmul' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain raw Q × Kᵀ matmul in decoder.',
        script: 'We multiply Q_h by K_h transposed and divide by sqrt(d_k). This creates a square grid of raw similarity scores between all target tokens.',
        audienceQuestion: 'Can tokens see future tokens in this raw score matrix?',
        expectedAnswer: 'Yes! In this raw grid, upper-triangle scores still exist until we apply causal masking next.',
        misconception: 'Highlight that raw matmul has not masked future tokens yet.',
        transition: 'Next, let\'s apply Causal Masking to enforce auto-regressive generation.'
      }
    ],
    speakerNotes: {
      teachingTips: ['Point to the upper-right triangle values in the heatmap before masking.'],
      misconceptions: ['Causal masking is NOT applied inside the matmul itself; it is applied immediately after matmul.']
    }
  },
  'dec-causal-mask': {
    eyebrow: 'Decoder Self-Attention · Causal Masking',
    title: 'Causal Masking (Lower Δ)',
    fourQuestions: {
      whatIsHappening: 'We add -infinity (or -1e9) to all upper-triangular elements of the score matrix (where col > row).',
      why: 'To prevent target token i from attending to future target tokens j > i, preserving auto-regressive causality during generation and training.',
      whatChanged: 'All entries in the upper triangle (col > row) become -inf; lower triangle (col <= row) remains unchanged.',
      whatToObserve: 'Watch the upper triangle of the heatmap turn dark/masked as -inf values are injected.'
    },
    body: {
      beginner: 'When predicting the next word, you cannot cheat by looking into the future! Causal masking blocks all future words by setting their score to negative infinity (-∞), which Softmax turns into 0% attention.',
      mtech: 'S_masked[i, j] = S[i, j] if j ≤ i else -∞. In practice, an upper-triangular mask matrix M with -∞ is added element-wise: S_masked = S + M.',
      research: 'Causal (autoregressive) masking enforces the conditional independence property P(y_t | y_<t, X). In PyTorch, nn.TransformerDecoder uses torch.triu(..., diagonal=1) filled with float("-inf").'
    },
    deepDive: {
      math: 'M_{i,j} = \\begin{cases} 0 & \\text{if } j \\le i \\\\ -\\infty & \\text{if } j > i \\end{cases}, \\quad S_{\\text{masked}} = S + M',
      complexity: 'O(L_{\\text{dec}}^2) element-wise mask addition.',
      matrixEquivalence: 'Lower-triangular mask addition forcing e^{-\\infty} = 0 in Softmax.',
      misconceptions: [
        'Causal masking is only used in decoder self-attention. Encoder self-attention and decoder cross-attention do NOT use causal masking.'
      ],
      notes: 'Lower triangular matrix keeps entries on and below the main diagonal.'
    },
    whyPanel: {
      items: [
        {
          title: 'Why use -infinity instead of 0 for masking?',
          body: 'Because Softmax exponentiates its inputs (e^x). e^0 = 1, which would still give future tokens attention weight! But e^(-inf) = 0, which guarantees future tokens get exactly 0.0 attention weight after Softmax.'
        }
      ],
      example: {
        left: [[1.2, 0.8], [0.5, 1.1]],
        right: [[1.2, -1e9], [0.5, 1.1]],
        caption: 'Upper-triangle entry (row 0, col 1) set to -inf.'
      }
    },
    beforeAfter: {
      before: { label: 'Raw Unmasked Scores S', shape: null },
      after: { label: 'Causally Masked Scores S_masked', shape: null },
      whatChanged: 'Upper triangle (future tokens) set to -infinity.',
      structured: {
        entered: 'Square score matrix of shape [target_seq_len, target_seq_len].',
        happened: 'Upper-triangular mask M added (-inf where col > row).',
        changed: 'Future token positions blocked from receiving attention.',
        leaves: 'Causally masked score matrix with lower triangular non-inf values.'
      }
    },
    quickCheck: {
      question: 'Why is -infinity used for causal masking instead of 0?',
      choices: [
        'Because e^(-inf) = 0 in Softmax, ensuring future tokens get exactly 0% attention weight.',
        'Because 0 is reserved for diagonal elements.',
        'Because matrix multiplication requires negative numbers.',
        'To reduce floating point precision requirements.'
      ],
      correctIndex: 0,
      explanation: 'Softmax exponentiates inputs: e^(-inf) = 0, guaranteeing zero attention weight for future tokens.',
      transition: 'Now let\'s apply Softmax to convert masked scores into probabilities.',
      distractorNotes: {
        1: 'Incorrect. Diagonal elements contain real dot products.',
        2: 'Incorrect. Matmul works with any real numbers.',
        3: 'Incorrect. Precision is unaffected.'
      }
    },
    pytorch: [
      { id: 'code-dec-causal-mask', code: 'mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)\nscores = scores + mask' }
    ],
    equationTerms: [
      { id: 'eq-dec-causal-mask', tex: 'S_{\\text{masked}} = S + M, \\quad M_{i,j} = \\begin{cases} 0 & j \\le i \\\\ -\\infty & j > i \\end{cases}' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-causal-mask', equationTermId: 'eq-dec-causal-mask', codeLineId: 'code-dec-causal-mask' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain Causal Masking in the decoder.',
        script: 'We add -infinity to all upper-triangle positions where col > row. This prevents any token from looking at future words during training or generation.',
        audienceQuestion: 'Does the first token ("<sos>") see any other target tokens?',
        expectedAnswer: 'No! The first token can only see itself.',
        misconception: 'Highlight that row 0 has all future columns masked to -inf.',
        transition: 'Next, we run Softmax to turn these scores into attention weights.'
      }
    ],
    speakerNotes: {
      teachingTips: ['Point to row 0: only col 0 is unmasked. Point to row 1: col 0 and col 1 are unmasked.'],
      misconceptions: ['Causal masking is NOT used in Encoder self-attention or Cross-attention.']
    }
  },
  'dec-scale-softmax': {
    eyebrow: 'Decoder Self-Attention · Masked Softmax',
    title: 'Decoder Masked Softmax',
    fourQuestions: {
      whatIsHappening: 'Softmax is applied row-wise to the causally masked score matrix: A_dec[i] = softmax(S_masked[i]).',
      why: 'Softmax converts raw scores into a valid probability distribution where each row sums to 1.0, with future tokens receiving 0.0 weight.',
      whatChanged: 'Masked score grid S_masked is transformed into attention weight matrix A_dec of shape [target_seq_len, target_seq_len].',
      whatToObserve: 'Observe how all upper-triangle entries become 0.0%, while valid past positions sum to 100% (1.0) per row.'
    },
    body: {
      beginner: 'Softmax turns scores into percentages. Because future words had a score of -∞, their percentage is exactly 0%. Each row now adds up to 100% across available past words!',
      mtech: 'A_dec = softmax(S_masked, dim=-1). For row i, ∑_(j=0)^(i) A[i,j] = 1.0, and A[i,j] = 0 ∀ j > i.',
      research: 'Row-wise Softmax over causally masked dot-product scores. Exp-normalize formulation e^(S_(i,j)) / ∑_k e^(S_(i,k)) naturally zeroes out masked positions due to e^(-∞) = 0.'
    },
    deepDive: {
      math: 'A_{\\text{dec}}[i, j] = \\frac{e^{S_{\\text{masked}}[i, j]}}{\\sum_{k=0}^{L-1} e^{S_{\\text{masked}}[i, k]}}',
      complexity: 'O(h \\cdot L_{\\text{dec}}^2) exponentiations and divisions.',
      matrixEquivalence: 'Row-wise normalization producing lower-triangular stochastic matrix.',
      misconceptions: [
        'Each row sums to 1.0 independently — token 0 has 100% on itself, token 1 splits 100% between token 0 and token 1.'
      ],
      notes: 'Lower triangular probability matrix.'
    },
    whyPanel: {
      items: [
        {
          title: 'Why must attention weights sum to 1.0 for each row?',
          body: 'Summing to 1.0 ensures that the resulting weighted average of Value vectors preserves scale. If weights summed to more or less than 1.0, vector magnitudes would explode or shrink as signals pass through layers.'
        }
      ],
      example: {
        left: [1.2, -1e9],
        right: [1.0, 0.0],
        caption: 'Softmax converts [1.2, -inf] to [1.0, 0.0].'
      }
    },
    beforeAfter: {
      before: { label: 'Causally Masked Scores S_masked', shape: null },
      after: { label: 'Softmax Attention Weights A_dec', shape: null },
      whatChanged: 'Raw scores converted to row-wise probabilities summing to 1.0.',
      structured: {
        entered: 'Causally masked score matrix.',
        happened: 'Row-wise exponentiation and sum normalization.',
        changed: 'Negative infinity entries become 0.0; valid past entries sum to 1.0 per row.',
        leaves: 'Lower-triangular attention probability matrix A_dec.'
      }
    },
    quickCheck: {
      question: 'What is the attention weight for a future token (where col > row) after Masked Softmax?',
      choices: [
        'Exactly 0.0 (because e^-inf = 0)',
        '0.5 (equal probability)',
        '1.0 (maximum weight)',
        '-1.0 (negative weight)'
      ],
      correctIndex: 0,
      explanation: 'Exponentiating -infinity yields e^-inf = 0, giving future tokens exactly 0.0 weight.',
      transition: 'Now we use these attention weights to blend the Value vectors!',
      distractorNotes: {
        1: 'Incorrect. Future tokens receive no weight.',
        2: 'Incorrect. Max weight goes to valid tokens.',
        3: 'Incorrect. Softmax outputs are strictly non-negative probabilities [0, 1].'
      }
    },
    pytorch: [
      { id: 'code-dec-scale-softmax', code: 'attn_weights = F.softmax(scores, dim=-1)' }
    ],
    equationTerms: [
      { id: 'eq-dec-scale-softmax', tex: 'A_{\\text{dec}} = \\text{softmax}(S_{\\text{masked}})' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-scale-softmax', equationTermId: 'eq-dec-scale-softmax', codeLineId: 'code-dec-scale-softmax' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain decoder masked Softmax.',
        script: 'Softmax normalizes each row into a probability distribution. Future tokens receive 0% weight, while past tokens split 100% of the attention.',
        audienceQuestion: 'What is the sum of attention weights in each row?',
        expectedAnswer: 'Exactly 1.0 (100%).',
        misconception: 'Confirm that row 0 puts 100% weight on itself because all other columns are masked.',
        transition: 'Next, we compute the Weighted Sum with Value vectors.'
      }
    ],
    speakerNotes: {
      teachingTips: ['Highlight row 0: 100% on col 0. Row 1: splits 100% between col 0 and col 1.'],
      misconceptions: ['Softmax probabilities are always between 0.0 and 1.0.']
    }
  },
  'dec-weighted-sum': {
    eyebrow: 'Decoder Self-Attention · Weighted Sum',
    title: 'Decoder Weighted Sum',
    fourQuestions: {
      whatIsHappening: 'For each head, we multiply the attention probability matrix A_dec by the Value matrix V_h to compute head output O_h = A_dec * V_h.',
      why: 'This blends past target token Value vectors according to their attention weights, creating context-enriched representations.',
      whatChanged: 'Attention matrix [L, L] multiplies Value matrix [L, d_k] to produce head output O_h of shape [target_seq_len, d_k].',
      whatToObserve: 'Watch each target token collect a weighted combination of Value vectors from itself and prior target tokens.'
    },
    body: {
      beginner: 'We use the attention percentages to mix together the Value vectors of past words. Each target word receives a custom cocktail of information from preceding words!',
      mtech: 'O_h = A_dec V_h ∈ R^(L_dec × d_k). For position i, O_h[i] = ∑_(j=0)^(i) A[i,j] V_h[j].',
      research: 'Linear combination of Value vectors weighted by causally masked self-attention probabilities. Produces context-aware target representations restricted to historical tokens.'
    },
    deepDive: {
      math: 'O_h = A_{\\text{dec}} V_h \\in \\mathbb{R}^{L_{\\text{dec}} \\times d_k}',
      complexity: 'O(h \\cdot L_{\\text{dec}}^2 \\cdot d_k) FLOPs.',
      matrixEquivalence: 'Matrix multiplication contracting the sequence dimension L.',
      misconceptions: [
        'Each target token vector only contains information from current and past target tokens — never future tokens.'
      ],
      notes: 'Head output shape: [target_seq_len, d_k].'
    },
    whyPanel: {
      items: [
        {
          title: 'Why is the output vector for token 0 identical to Value vector 0?',
          body: 'Because token 0 can only attend to itself (100% weight on position 0), its weighted sum is 1.0 * V[0] = V[0]. Token 1 can mix V[0] and V[1], and so on!'
        }
      ],
      example: {
        left: [1.0, 0.0],
        right: [[0.4, 0.2], [0.8, -0.1]],
        caption: 'Row 0 weights [1.0, 0.0] pick out row 0 of V.'
      }
    },
    beforeAfter: {
      before: { label: 'Attention Weights A & Values V', shape: null },
      after: { label: 'Contextual Head Output O_h', shape: null },
      whatChanged: 'Value vectors blended according to causal attention weights.',
      structured: {
        entered: 'Attention matrix [L, L] and Value matrix [L, d_k].',
        happened: 'Matrix multiplication O_h = A * V_h.',
        changed: 'Un-contextualized Values blended into causally context-aware vectors.',
        leaves: 'Head output matrix O_h of shape [target_seq_len, d_k].'
      }
    },
    quickCheck: {
      question: 'Which vectors are multiplied by the decoder self-attention weights A_dec?',
      choices: [
        'Decoder Value vectors V_dec',
        'Decoder Query vectors Q_dec',
        'Encoder Key vectors K_enc',
        'Positional Encoding vectors'
      ],
      correctIndex: 0,
      explanation: 'Attention weights A_dec multiply Decoder Value vectors V_dec to form the weighted sum.',
      transition: 'Now let\'s compare all 4 decoder self-attention heads!',
      distractorNotes: {
        1: 'Incorrect. Queries compute dot products with Keys.',
        2: 'Incorrect. Encoder keys belong to cross-attention.',
        3: 'Incorrect. Positional encodings are added at the input.'
      }
    },
    pytorch: [
      { id: 'code-dec-weighted-sum', code: 'head_out = torch.matmul(attn_weights, V_h)' }
    ],
    equationTerms: [
      { id: 'eq-dec-weighted-sum', tex: 'O_h = A_{\\text{dec}} V_h' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-weighted-sum', equationTermId: 'eq-dec-weighted-sum', codeLineId: 'code-dec-weighted-sum' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain decoder weighted sum.',
        script: 'We multiply attention weights A_dec by Value vectors V_h. Each token position receives a weighted mix of past Value vectors.',
        audienceQuestion: 'Does token 0 receive information from token 1?',
        expectedAnswer: 'No, token 0 only receives information from itself.',
        misconception: 'Confirm that information flows only from past positions to current positions.',
        transition: 'Next, let\'s compare all attention heads side-by-side.'
      }
    ],
    speakerNotes: {
      teachingTips: ['Point out that row 0 output equals V[0], while row 1 output mixes V[0] and V[1].'],
      misconceptions: ['Weighted sum contracts sequence columns, resulting in shape [L, d_k].']
    }
  },
  'dec-heads-compare': {
    eyebrow: 'Decoder Self-Attention · Head Comparison',
    title: 'Decoder Attention Heads',
    fourQuestions: {
      whatIsHappening: 'We compare the causally masked attention heatmaps across all num_heads parallel attention heads.',
      why: 'Each head learns distinct causal relationships (e.g., immediate previous word, start token anchor, grammatical dependencies).',
      whatChanged: 'Allows side-by-side inspection of num_heads distinct lower-triangular attention distributions.',
      whatToObserve: 'Observe how all heads respect the causal mask (upper triangle is 0), but place high attention on different past tokens.'
    },
    body: {
      beginner: 'Comparing the 4 decoder heads shows how each head pays attention to different past words. Head 1 might focus on the word right before it, while Head 2 focuses on the start of the sentence!',
      mtech: 'Multi-head decoder self-attention provides h independent causal attention distributions A^(1), ..., A^(h).',
      research: 'Empirical analysis of decoder self-attention heads reveals specialized roles: positional offset heads (attending to t-1), global anchor heads (attending to <sos>), and semantic agreement heads.'
    },
    deepDive: {
      math: 'A^{(h)} = \\text{softmax}\\left(\\frac{Q^{(h)} K^{(h)T}}{\\sqrt{d_k}} + M\\right)',
      complexity: 'O(h \\cdot L_{\\text{dec}}^2) parallel attention patterns.',
      matrixEquivalence: 'Parallel evaluation of h lower-triangular stochastic matrices.',
      misconceptions: [
        'All heads enforce causal masking (upper triangle zero), but their non-zero lower triangle weights differ.'
      ],
      notes: '4 heads visualized simultaneously.'
    },
    whyPanel: {
      items: [
        {
          title: 'Why do different heads learn different attention patterns?',
          body: 'Random initialization causes each head to project into different d_k subspaces. During backpropagation, gradients drive each head to specialize in capturing distinct syntactic or semantic dependencies.'
        }
      ],
      example: {
        left: 'Head 1: Focuses on t-1',
        right: 'Head 2: Focuses on <sos>',
        caption: 'Different decoder heads capture distinct temporal patterns.'
      }
    },
    beforeAfter: {
      before: { label: 'Single Head View', shape: null },
      after: { label: 'All 4 Decoder Heads View', shape: null },
      whatChanged: 'Side-by-side comparison of 4 parallel decoder attention heads.',
      structured: {
        entered: 'Single head attention matrix.',
        happened: 'Parallel rendering of all head attention heatmaps.',
        changed: 'Multiple head specializations made visible simultaneously.',
        leaves: '4 lower-triangular attention probability heatmaps.'
      }
    },
    quickCheck: {
      question: 'Do all decoder self-attention heads enforce causal masking?',
      choices: [
        'Yes, all decoder self-attention heads mask future tokens (upper triangle = 0).',
        'No, only Head 0 is masked.',
        'No, heads are unmasked during training.',
        'Only heads 1 and 2 are masked.'
      ],
      correctIndex: 0,
      explanation: 'Every decoder self-attention head applies the exact same causal mask to prevent looking into the future.',
      transition: 'Now let\'s concatenate the outputs of all 4 heads back together!',
      distractorNotes: {
        1: 'Incorrect. All heads must enforce causality.',
        2: 'Incorrect. Causal masking is active during both training and inference.',
        3: 'Incorrect. Masking applies across all heads.'
      }
    },
    pytorch: [
      { id: 'code-dec-heads-compare', code: '# Multi-head decoder self-attention weight comparison' }
    ],
    equationTerms: [
      { id: 'eq-dec-heads-compare', tex: 'A^{(h)} = \\text{softmax}\\left(\\frac{Q^{(h)} K^{(h)T}}{\\sqrt{d_k}} + M\\right)' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-heads-compare', equationTermId: 'eq-dec-heads-compare', codeLineId: 'code-dec-heads-compare' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Compare decoder self-attention heads.',
        script: 'Look at all 4 decoder attention heatmaps side-by-side. Notice how every head has its upper triangle zeroed out, but each head focuses on different past words.',
        audienceQuestion: 'Do any heads look into the future?',
        expectedAnswer: 'No, every head has the upper triangle masked to 0.',
        misconception: 'Highlight that head diversity exists within the lower triangle.',
        transition: 'Next, we concatenate the head outputs back together.'
      }
    ],
    speakerNotes: {
      teachingTips: ['Point out how every heatmap has a clear diagonal edge due to causal masking.'],
      misconceptions: ['All heads enforce causal masking equally.']
    }
  },
  'dec-concat': {
    eyebrow: 'Decoder Self-Attention · Concatenation',
    title: 'Decoder Concatenation',
    fourQuestions: {
      whatIsHappening: 'The num_heads head outputs O_1, ..., O_h of shape [target_seq_len, d_k] are concatenated side-by-side along the feature dimension back into shape [target_seq_len, d_model].',
      why: 'To combine the diverse contextual features extracted by all parallel attention heads into a single matrix of dimension d_model.',
      whatChanged: '4 matrices of shape [target_seq_len, 4] are stitched together into one matrix of shape [target_seq_len, 16].',
      whatToObserve: 'Watch the 4 separate 4-column head blocks merge side-by-side into a single 16-column matrix.'
    },
    body: {
      beginner: 'We take the 4 smaller head outputs and stitch them back together side-by-side. 4 heads of size 4 become one full 16-number vector for each target word!',
      mtech: 'Concat(O_1, ..., O_h) ∈ R^(L_dec × d_model), where h · d_k = d_model.',
      research: 'Head concatenation restores the model dimension d_model prior to linear output projection Wo. Recombines multi-head subspace features into a unified tensor.'
    },
    deepDive: {
      math: '\\text{Concat}(O_1, \\dots, O_h) \\in \\mathbb{R}^{L_{\\text{dec}} \\times d_{\\text{model}}}',
      complexity: 'O(1) memory view concatenation.',
      matrixEquivalence: 'Stitching columns along dim=-1.',
      misconceptions: [
        'Concatenation places head outputs side-by-side; it does NOT sum them together.'
      ],
      notes: 'Shape: [target_seq_len, 16].'
    },
    whyPanel: {
      items: [
        {
          title: 'Why concatenate instead of averaging the head outputs?',
          body: 'Concatenation preserves all information from every head independently without destroying feature channels. Averaging would blend all head channels together, losing the distinct insights each head learned.'
        }
      ],
      example: {
        left: '4 heads x 4 cols',
        right: '1 matrix x 16 cols',
        caption: 'Stitching head columns side-by-side.'
      }
    },
    beforeAfter: {
      before: { label: 'Split Head Outputs O_1..O_4', shape: null },
      after: { label: 'Concatenated Matrix', shape: null },
      whatChanged: 'Head output matrices concatenated side-by-side into d_model columns.',
      structured: {
        entered: '4 head matrices of shape [target_seq_len, 4].',
        happened: 'Concatenation along feature dimension dim=-1.',
        changed: 'Parallel head channels combined into a single matrix.',
        leaves: 'Concatenated matrix of shape [target_seq_len, 16].'
      }
    },
    quickCheck: {
      question: 'What is the tensor shape after concatenating 4 heads of dimension d_k = 4?',
      choices: [
        '[target_seq_len, 16] (because 4 * 4 = 16 = d_model)',
        '[target_seq_len, 4]',
        '[4, 4]',
        '[16, 16]'
      ],
      correctIndex: 0,
      explanation: 'Stitching 4 heads of width 4 produces a matrix of width 16 (d_model).',
      transition: 'Now we pass this concatenated matrix through the Output Projection Wo_dec!',
      distractorNotes: {
        1: 'Incorrect. 4 is the single head dimension d_k.',
        2: 'Incorrect. Head count and d_k pair.',
        3: 'Incorrect. Square dimension.'
      }
    },
    pytorch: [
      { id: 'code-dec-concat', code: 'concat_out = head_outputs.transpose(1, 2).contiguous().view(B, L, d_model)' }
    ],
    equationTerms: [
      { id: 'eq-dec-concat', tex: '\\text{Concat}(O_1, \\dots, O_h) \\in \\mathbb{R}^{L_{\\text{dec}} \\times d_{\\text{model}}}' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-concat', equationTermId: 'eq-dec-concat', codeLineId: 'code-dec-concat' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Explain decoder head concatenation.',
        script: 'We concatenate the outputs from all 4 attention heads side-by-side. 4 heads of width 4 form a full 16-dimensional vector for each word.',
        audienceQuestion: 'Does concatenation add the numbers together?',
        expectedAnswer: 'No, it places them side-by-side in columns.',
        misconception: 'Confirm that concatenation joins columns without arithmetic addition.',
        transition: 'Finally, we apply the linear Output Projection Wo_dec.'
      }
    ],
    speakerNotes: {
      teachingTips: ['Show how columns 0-3 come from Head 1, 4-7 from Head 2, 8-11 from Head 3, 12-15 from Head 4.'],
      misconceptions: ['Concatenation is a reshape/stitch operation, not an addition.']
    }
  },
  'dec-output-proj': {
    eyebrow: 'Decoder Self-Attention · Output Projection',
    title: 'Decoder Output Projection',
    fourQuestions: {
      whatIsHappening: 'The concatenated multi-head output matrix is multiplied by weight matrix Wo_dec of shape [d_model, d_model] plus bias bo_dec.',
      why: 'Wo_dec mixes the features from all attention heads together and projects them back into the decoder residual stream space.',
      whatChanged: 'Concatenated matrix [target_seq_len, d_model] is linearly projected to final self-attention output O_dec of shape [target_seq_len, d_model].',
      whatToObserve: 'Watch the concatenated head features pass through Wo_dec to form the final output of Decoder Masked Self-Attention.'
    },
    body: {
      beginner: 'The final step in self-attention is multiplying by an Output Projection matrix (Wo). This blends the information from all 4 heads together into a final, polished vector for each target word.',
      mtech: 'O_dec = Concat(O_1, ..., O_h) W_(o,dec) + b_(o,dec) ∈ R^(L_dec × d_model).',
      research: 'Multi-head self-attention output projection: MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O. Maps multi-head concatenated features into the residual stream.'
    },
    deepDive: {
      math: 'O_{\\text{dec}} = \\text{Concat}(O_1, \\dots, O_h) W_{o,\\text{dec}} + b_{o,\\text{dec}}',
      complexity: 'O(L_{\\text{dec}} \\cdot d_{\\text{model}}^2) FLOPs.',
      matrixEquivalence: 'Linear transformation mixing concatenated head channels.',
      misconceptions: [
        'Wo_dec is an essential linear transformation that allows heads to interact — without Wo, head outputs would remain isolated channels.'
      ],
      notes: 'Final output shape: [target_seq_len, d_model].'
    },
    whyPanel: {
      items: [
        {
          title: 'Why is the output projection Wo necessary after concatenation?',
          body: 'Concatenation simply places head outputs side-by-side in separate columns. Wo linearly combines features across all heads, allowing the network to learn interactions between what Head 1 discovered and what Head 2 discovered.'
        }
      ],
      example: {
        left: [1, 16],
        right: [1, 16],
        caption: 'Concatenated vector multiplied by Wo matrix.'
      }
    },
    beforeAfter: {
      before: { label: 'Concatenated Head Matrix', shape: null },
      after: { label: 'Decoder Self-Attention Output O_dec', shape: null },
      whatChanged: 'Multi-head features linearly projected into final output vectors.',
      structured: {
        entered: 'Concatenated matrix of shape [target_seq_len, d_model].',
        happened: 'Matrix multiplication with Wo_dec plus bias bo_dec.',
        changed: 'Multi-head features mixed across channel dimensions.',
        leaves: 'Final Masked Self-Attention output tensor of shape [target_seq_len, d_model].'
      }
    },
    quickCheck: {
      question: 'What is the purpose of the Output Projection matrix Wo_dec?',
      choices: [
        'To linearly mix features across all attention heads before adding to the residual stream.',
        'To mask future target tokens.',
        'To calculate token embeddings.',
        'To compute Softmax probabilities.'
      ],
      correctIndex: 0,
      explanation: 'Wo_dec allows features from different attention heads to interact and mix together.',
      transition: 'Decoder Masked Self-Attention is complete! Next comes Decoder Residual Connection ①.',
      distractorNotes: {
        1: 'Incorrect. Causal masking masks future tokens.',
        2: 'Incorrect. Embeddings convert token IDs.',
        3: 'Incorrect. Softmax computes probabilities.'
      }
    },
    pytorch: [
      { id: 'code-dec-output-proj', code: 'self_attn_out = self.Wo(concat_out)' }
    ],
    equationTerms: [
      { id: 'eq-dec-output-proj', tex: 'O_{\\text{dec}} = \\text{Concat}(O_1, \\dots, O_h) W_{o,\\text{dec}}' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-output-proj', equationTermId: 'eq-dec-output-proj', codeLineId: 'code-dec-output-proj' }
    ],
    narration: [
      {
        duration: '~35s',
        objective: 'Explain decoder output projection.',
        script: 'We multiply the concatenated head outputs by Wo_dec. This mixes the features from all 4 heads together, producing the final output of Decoder Masked Self-Attention.',
        audienceQuestion: 'Does the output matrix have the same shape as the decoder input?',
        expectedAnswer: 'Yes! [target_seq_len, d_model].',
        misconception: 'Highlight that output shape matches input shape, ready for residual addition.',
        transition: 'Decoder Masked Self-Attention is complete! Next, we add the residual connection.'
      }
    ],
    speakerNotes: {
      teachingTips: ['Emphasize that output shape matches input shape [L_dec, d_model] so it can be added to the residual stream.'],
      misconceptions: ['Wo_dec is a learned weight matrix of shape [d_model, d_model].']
    }
  },
  'dec-residual-1': {
    eyebrow: 'DECODER OPERATOR 3: SKIP CONNECTION',
    title: 'Decoder Residual Connection ①',
    fourQuestions: {
      q1: 'What problem does the decoder residual connection solve?',
      a1: 'It prevents vanishing gradients and preserves target token identity across self-attention sublayers.',
      q2: 'What are the two input paths being added?',
      a2: 'The skip connection from decoder positional encoding (X_dec_pe) and the masked self-attention output projection (O_dec).',
      q3: 'Why must both tensors have the exact same shape?',
      a3: 'Element-wise addition requires identical dimensions [target_seq_len, d_model].',
      q4: 'How does this impact backpropagation?',
      a4: 'Gradients flow directly through the skip connection without passing through projection weights.'
    },
    body: {
      beginner: 'The residual connection adds the original decoder input vectors back to the result of masked self-attention.',
      mtech: 'The skip connection forms X_dec + MultiHeadSelfAttention(X_dec), stabilizing gradient flow during training.',
      research: 'Residual pathways allow deep decoder architectures (e.g. 96-layer models) to train stably without optimization collapse.'
    },
    deepDive: {
      title: 'Residual Stream in Decoder Architectures',
      content: 'In auto-regressive decoders, the residual stream carries target token representations forward while each sublayer writes additive updates.'
    },
    whyPanel: {
      summary: 'Without residual connections, deep decoders suffer from severe gradient attenuation.'
    },
    beforeAfter: {
      before: 'Unstabilized sublayer outputs',
      after: 'Summed residual representation X_dec + O_dec'
    },
    quickCheck: {
      question: 'What is the output of the first decoder residual connection?',
      options: [
        'X_dec_pe + Output_Proj',
        'X_dec_pe * Output_Proj',
        'LayerNorm(X_dec_pe)',
        'Softmax(Output_Proj)'
      ],
      correctIndex: 0,
      explanation: 'The first decoder residual connection computes element-wise addition X_dec_pe + Output_Proj.',
      distractorNotes: {
        1: 'Incorrect. Residual connections use addition, not multiplication.',
        2: 'Incorrect. LayerNorm is applied after the residual connection.',
        3: 'Incorrect. Softmax occurs inside self-attention.'
      }
    },
    pytorch: [
      { id: 'code-dec-residual-1', code: 'x = x + self.self_attn(x)' }
    ],
    equationTerms: [
      { id: 'eq-dec-residual-1', tex: 'X_{\\text{res1}} = X_{\\text{dec\\_pe}} + \\text{OutputProj}' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-residual-1', equationTermId: 'eq-dec-residual-1', codeLineId: 'code-dec-residual-1' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Explain the first decoder residual connection.',
        script: 'We add the incoming target token representations back to the masked self-attention output.',
        audienceQuestion: 'Why is this addition important for auto-regressive generation?',
        expectedAnswer: 'It ensures target token identity is preserved alongside contextual updates.',
        misconception: 'Make sure students see that both tensors have shape [seq_len, d_model].',
        transition: 'Next, we normalize the combined representation with Layer Normalization.'
      }
    ],
    speakerNotes: {
      teachingTips: ['Highlight how residual paths act as gradient highways during decoder training.'],
      misconceptions: ['Residual addition does not alter tensor shapes.'],
      suggestedQuestions: ['What would happen to target token identity without the skip connection?']
    }
  },
  'dec-layer-norm-1': {
    eyebrow: 'DECODER OPERATOR 4: NORMALIZATION',
    title: 'Decoder Layer Normalization ①',
    fourQuestions: {
      q1: 'What is the goal of Decoder Layer Normalization?',
      a1: 'To standardize feature distributions across target token vectors to mean 0 and variance 1.',
      q2: 'Which dimension is normalized?',
      a2: 'Each target token vector is normalized independently across its d_model feature dimension.',
      q3: 'Why do we use learned parameters γ (gamma) and β (beta)?',
      a3: 'To allow the network to re-scale and shift normalized representations back to optimal operational ranges.',
      q4: 'How does LayerNorm differ from BatchNorm?',
      a4: 'LayerNorm operates per token across features, making it independent of batch size and ideal for sequential decoding.'
    },
    body: {
      beginner: 'Layer Normalization standardizes each target word representation so vector values remain balanced.',
      mtech: 'LayerNorm computes mean and variance across d_model, normalizes, and applies element-wise affine transform γ * x_hat + β.',
      research: 'Post-LN / Pre-LN placement controls signal dynamics in auto-regressive Transformer decoders.'
    },
    deepDive: {
      title: 'Per-Token Normalization Invariance',
      content: 'Because LayerNorm normalizes each target token independently, generation behavior remains invariant across varying batch sizes during inference.'
    },
    whyPanel: {
      summary: 'Normalization prevents exploding / vanishing activations across decoder layers.'
    },
    beforeAfter: {
      before: 'Unnormalized residual sum',
      after: 'Normalized & affine-scaled output'
    },
    quickCheck: {
      question: 'What are the target statistics after Layer Normalization (before γ and β scaling)?',
      options: [
        'Mean = 0, Variance = 1',
        'Mean = 1, Variance = 0',
        'Mean = 0.5, Variance = 0.5',
        'Mean = -1, Variance = 1'
      ],
      correctIndex: 0,
      explanation: 'Standard Layer Normalization centers feature values to mean 0 and scales variance to 1.',
      distractorNotes: {
        1: 'Incorrect. Zero variance would mean all values are identical.',
        2: 'Incorrect. Mean is 0, not 0.5.',
        3: 'Incorrect. Mean is centered at 0.'
      }
    },
    pytorch: [
      { id: 'code-dec-layer-norm-1', code: 'x = self.norm1(x)' }
    ],
    equationTerms: [
      { id: 'eq-dec-layer-norm-1', tex: '\\text{LN}(x) = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\cdot \\gamma + \\beta' }
    ],
    syncMap: [
      { vizElementId: 'viz-dec-layer-norm-1', equationTermId: 'eq-dec-layer-norm-1', codeLineId: 'code-dec-layer-norm-1' }
    ],
    narration: [
      {
        duration: '~30s',
        objective: 'Explain decoder Layer Normalization.',
        script: 'Now we normalize each target token representation across its feature dimension.',
        audienceQuestion: 'Does LayerNorm look at future target tokens?',
        expectedAnswer: 'No, LayerNorm operates per token across feature channels independently.',
        misconception: 'Emphasize that normalization happens per row, independently across sequence tokens.',
        transition: 'The self-attention block of the decoder is now complete and ready for cross-attention.'
      }
    ],
    speakerNotes: {
      teachingTips: ['Demonstrate how mean and variance are calculated across the d_model columns of each row.'],
      misconceptions: ['LayerNorm does not mix information across target tokens.'],
      suggestedQuestions: ['Why is per-token LayerNorm ideal for auto-regressive decoding?']
    }
  },
  'dec-cross-proj-q': {
    eyebrow: 'CROSS ATTENTION: QUERY PROJECTION',
    title: 'Cross Projection Q (Decoder)',
    fourQuestions: {
      q1: 'Where do Cross-Attention Queries come from?',
      a1: 'From the normalized decoder representations (decoder.ln1_outputs).',
      q2: 'What is the role of Query vectors in Cross-Attention?',
      a2: 'They represent what information each target token is looking for in the source sentence.',
      q3: 'Which weight matrix projects Query vectors?',
      a3: 'W_q_cross of shape [d_model, d_model].',
      q4: 'Is Cross-Attention Query computation masked?',
      a4: 'No, Query vectors are projected per token independently.'
    },
    body: {
      beginner: 'The decoder projects its target tokens into Query vectors to ask the encoder questions.',
      mtech: 'Q_cross = X_dec_ln1 * W_q_cross produces Query vectors of shape [target_seq_len, d_model].',
      research: 'Cross-attention queries act as contextual probes into the encoder representations.'
    },
    deepDive: { title: 'Asymmetric Attention Roles', content: 'Cross-attention uses queries from the decoder stream and keys/values from the encoder stream.' },
    whyPanel: { summary: 'Projecting queries allows target tokens to query source tokens in a learned subspace.' },
    beforeAfter: { before: 'Decoder LN1 representation', after: 'Cross Query matrix Q_cross' },
    quickCheck: {
      question: 'Which tensor is projected to produce Cross-Attention Queries?',
      options: ['decoder.ln1_outputs', 'encoder.ln2_outputs', 'decoder.xPe', 'encoder.embeddings'],
      correctIndex: 0,
      explanation: 'Cross-attention queries are projected from decoder.ln1_outputs.',
      distractorNotes: { 1: 'Incorrect. encoder.ln2_outputs provides Keys and Values.', 2: 'Incorrect. decoder.xPe is input before self-attention.', 3: 'Incorrect. Encoder embeddings are un-contextualized.' }
    },
    pytorch: [{ id: 'code-dec-cross-proj-q', code: 'Q_cross = self.W_q_cross(dec_ln1)' }],
    equationTerms: [{ id: 'eq-dec-cross-proj-q', tex: 'Q_{\\text{cross}} = X_{\\text{dec\\_ln1}} W_{q,\\text{cross}}' }],
    syncMap: [{ vizElementId: 'viz-dec-cross-proj-q', equationTermId: 'eq-dec-cross-proj-q', codeLineId: 'code-dec-cross-proj-q' }],
    narration: [{ duration: '~30s', objective: 'Explain Cross-Attention Query projection.', script: 'We project normalized decoder tokens into Query space using W_q_cross.', audienceQuestion: 'Where do these queries come from?', expectedAnswer: 'From decoder.ln1_outputs.', misconception: 'Clarify that queries come from decoder, not encoder.', transition: 'Now let\'s project Encoder Keys.' }],
    speakerNotes: { teachingTips: ['Emphasize that queries originate in the decoder.'], misconceptions: ['Keys and Queries do not come from the same sequence in cross attention.'], suggestedQuestions: ['Why do queries come from the decoder?'] }
  },
  'dec-cross-proj-k': {
    eyebrow: 'CROSS ATTENTION: KEY PROJECTION',
    title: 'Cross Projection K (Encoder)',
    fourQuestions: {
      q1: 'Where do Cross-Attention Keys come from?',
      a1: 'From the final contextual output of the Encoder (encoder.ln2_outputs).',
      q2: 'What do Key vectors represent?',
      a2: 'They index source token features for matching against decoder queries.',
      q3: 'Which weight matrix is used?',
      a3: 'W_k_cross of shape [d_model, d_model].',
      q4: 'Do Keys change as target tokens are generated?',
      a4: 'No, encoder keys remain constant throughout sequence generation.'
    },
    body: {
      beginner: 'The encoder output is projected into Key vectors so the decoder can search source words.',
      mtech: 'K_cross = X_enc_ln2 * W_k_cross yields Key vectors of shape [source_seq_len, d_model].',
      research: 'Encoder key caching avoids redundant key projections during auto-regressive decoding.'
    },
    deepDive: { title: 'Static Encoder Key Caching', content: 'Because encoder outputs are fixed for a source sentence, K_cross can be computed once and reused for all decoding steps.' },
    whyPanel: { summary: 'Key projections map source token representations into the cross-attention matching space.' },
    beforeAfter: { before: 'Encoder LN2 output', after: 'Cross Key matrix K_cross' },
    quickCheck: {
      question: 'Where do Cross-Attention Keys originate?',
      options: ['encoder.ln2_outputs', 'decoder.ln1_outputs', 'decoder.xPe', 'decoder.Q'],
      correctIndex: 0,
      explanation: 'Keys in cross-attention come from encoder.ln2_outputs.',
      distractorNotes: { 1: 'Incorrect. decoder.ln1_outputs provides Queries.', 2: 'Incorrect. decoder.xPe is decoder input.', 3: 'Incorrect. decoder.Q is self-attention query.' }
    },
    pytorch: [{ id: 'code-dec-cross-proj-k', code: 'K_cross = self.W_k_cross(enc_ln2)' }],
    equationTerms: [{ id: 'eq-dec-cross-proj-k', tex: 'K_{\\text{cross}} = X_{\\text{enc\\_ln2}} W_{k,\\text{cross}}' }],
    syncMap: [{ vizElementId: 'viz-dec-cross-proj-k', equationTermId: 'eq-dec-cross-proj-k', codeLineId: 'code-dec-cross-proj-k' }],
    narration: [{ duration: '~30s', objective: 'Explain Cross-Attention Key projection.', script: 'We project the final encoder representations into Key vectors using W_k_cross.', audienceQuestion: 'Are these keys recomputed for every target token?', expectedAnswer: 'No, encoder keys can be cached.', misconception: 'Remind students that source keys come from the encoder.', transition: 'Next, let\'s project Encoder Values.' }],
    speakerNotes: { teachingTips: ['Point out key caching optimizations during inference.'], misconceptions: ['Keys do not come from target tokens.'], suggestedQuestions: ['Why can encoder keys be cached?'] }
  },
  'dec-cross-proj-v': {
    eyebrow: 'CROSS ATTENTION: VALUE PROJECTION',
    title: 'Cross Projection V (Encoder)',
    fourQuestions: {
      q1: 'Where do Cross-Attention Values come from?',
      a1: 'From the final contextual output of the Encoder (encoder.ln2_outputs).',
      q2: 'What do Value vectors contain?',
      a2: 'The source information that will be retrieved and blended into decoder tokens.',
      q3: 'Which weight matrix projects Value vectors?',
      a3: 'W_v_cross of shape [d_model, d_model].',
      q4: 'Can Value vectors also be cached?',
      a4: 'Yes, like Keys, Value vectors remain constant for a given source sentence.'
    },
    body: {
      beginner: 'Value vectors hold the actual information that the decoder extracts from the source sentence.',
      mtech: 'V_cross = X_enc_ln2 * W_v_cross of shape [source_seq_len, d_model].',
      research: 'Value representations encode deep contextualized source features for translation transfer.'
    },
    deepDive: { title: 'Value Vector Caching', content: 'Caching V_cross prevents re-running encoder projections during auto-regressive token generation.' },
    whyPanel: { summary: 'Value projections prepare source content vectors for weighted blending.' },
    beforeAfter: { before: 'Encoder LN2 output', after: 'Cross Value matrix V_cross' },
    quickCheck: {
      question: 'Which tensor is projected to produce Cross-Attention Values?',
      options: ['encoder.ln2_outputs', 'decoder.ln1_outputs', 'decoder.scores', 'decoder.residual1'],
      correctIndex: 0,
      explanation: 'Cross-attention values are projected from encoder.ln2_outputs.',
      distractorNotes: { 1: 'Incorrect. decoder.ln1_outputs provides Queries.', 2: 'Incorrect. Scores are intermediate matrices.', 3: 'Incorrect. Residual1 is decoder state.' }
    },
    pytorch: [{ id: 'code-dec-cross-proj-v', code: 'V_cross = self.W_v_cross(enc_ln2)' }],
    equationTerms: [{ id: 'eq-dec-cross-proj-v', tex: 'V_{\\text{cross}} = X_{\\text{enc\\_ln2}} W_{v,\\text{cross}}' }],
    syncMap: [{ vizElementId: 'viz-dec-cross-proj-v', equationTermId: 'eq-dec-cross-proj-v', codeLineId: 'code-dec-cross-proj-v' }],
    narration: [{ duration: '~30s', objective: 'Explain Cross-Attention Value projection.', script: 'We project the encoder output into Value vectors V_cross.', audienceQuestion: 'Do Values come from the encoder or decoder?', expectedAnswer: 'From the encoder.', misconception: 'Confirm that Values represent source sentence content.', transition: 'Now let\'s split all three projections into heads.' }],
    speakerNotes: { teachingTips: ['Emphasize that Q comes from decoder, K and V come from encoder.'], misconceptions: ['Values are source vectors, not target vectors.'], suggestedQuestions: ['How do Value vectors transfer source content to target tokens?'] }
  },
  'dec-cross-split-heads': {
    eyebrow: 'CROSS ATTENTION: HEAD SPLITTING',
    title: 'Cross Split Heads',
    fourQuestions: {
      q1: 'How are Cross-Attention heads split?',
      a1: 'Q_cross [L_dec, d_model] is split into [h, L_dec, d_k], while K_cross and V_cross [L_enc, d_model] are split into [h, L_enc, d_k].',
      q2: 'Why can source and target sequence lengths differ?',
      a2: 'Because matrix multiplication [L_dec, d_k] x [d_k, L_enc] produces a grid of shape [L_dec, L_enc].',
      q3: 'What is d_k?',
      a3: 'd_model / num_heads.',
      q4: 'Does head splitting alter tensor values?',
      a4: 'No, it reshapes and transposes dimensions for parallel multi-head attention.'
    },
    body: { beginner: 'We divide the Query, Key, and Value matrices into multiple attention heads.', mtech: 'Reshapes Q_cross to [h, L_dec, d_k] and K_cross, V_cross to [h, L_enc, d_k].', research: 'Multi-head cross-attention allows simultaneous alignment to multiple source concepts.' },
    deepDive: { title: 'Asymmetric Sequence Grid Dimensions', content: 'Cross-attention matrices have rows equal to target sequence length and columns equal to source sequence length.' },
    whyPanel: { summary: 'Splitting heads enables diverse cross-attention pathways across different representation subspaces.' },
    beforeAfter: { before: 'Combined projection matrices', after: 'Split head tensors Q_h, K_h, V_h' },
    quickCheck: {
      question: 'What is the shape of a split Cross Key head K_h?',
      options: ['[source_seq_len, d_k]', '[target_seq_len, d_k]', '[d_model, d_model]', '[source_seq_len, target_seq_len]'],
      correctIndex: 0,
      explanation: 'Key heads have shape [source_seq_len, d_k].',
      distractorNotes: { 1: 'Incorrect. Query heads have shape [target_seq_len, d_k].', 2: 'Incorrect. d_model is the un-split dimension.', 3: 'Incorrect. Scores have shape [target_seq_len, source_seq_len].' }
    },
    pytorch: [{ id: 'code-dec-cross-split-heads', code: 'Q_h = split_heads(Q_cross, num_heads)' }],
    equationTerms: [{ id: 'eq-dec-cross-split-heads', tex: 'Q_h \\in \\mathbb{R}^{L_{\\text{dec}} \\times d_k}, K_h \\in \\mathbb{R}^{L_{\\text{enc}} \\times d_k}' }],
    syncMap: [{ vizElementId: 'viz-dec-cross-split-heads', equationTermId: 'eq-dec-cross-split-heads', codeLineId: 'code-dec-cross-split-heads' }],
    narration: [{ duration: '~30s', objective: 'Explain cross-attention head splitting.', script: 'We split Q_cross, K_cross, and V_cross into multiple heads.', audienceQuestion: 'Why can source and target lengths differ?', expectedAnswer: 'Because Q_h and K_h contract over d_k.', misconception: 'Show that sequence lengths L_dec and L_enc do not have to be equal.', transition: 'Next, we compute raw cross-attention scores.' }],
    speakerNotes: { teachingTips: ['Stress that L_dec and L_enc can be completely different lengths.'], misconceptions: ['Head splitting does not change numerical values, only tensor shapes.'], suggestedQuestions: ['What determines the dimensions of the cross-attention score grid?'] }
  },
  'dec-cross-qk-matmul': {
    eyebrow: 'CROSS ATTENTION: RAW SCORES',
    title: 'Cross Q × Kᵀ (Raw Scores)',
    fourQuestions: {
      q1: 'How are raw cross-attention scores computed?',
      a1: 'S = (Q_h * K_h^T) / sqrt(d_k) of shape [target_seq_len, source_seq_len].',
      q2: 'What does score S[i][j] represent?',
      a2: 'The raw alignment strength between target token i and source token j.',
      q3: 'Is causal masking applied to cross-attention?',
      a3: 'No! Cross-attention is unmasked; every target token can attend to all source tokens.',
      q4: 'Why is causal masking absent here?',
      a4: 'Because the entire source sentence is already fully known and encoded.',
    },
    body: { beginner: 'We compute dot products between target queries and source keys to find which source words match each target word.', mtech: 'S_cross = (Q_cross * K_cross^T) / sqrt(d_k) resulting in grid of shape [L_dec, L_enc].', research: 'Unmasked cross-attention provides complete bidirectional access to the encoded source context.' },
    deepDive: { title: 'Unmasked Cross-Attention Matrix', content: 'Unlike masked self-attention, cross-attention matrix has no lower-triangular restrictions because source context is fully available.' },
    whyPanel: { summary: 'Raw cross-scores measure semantic affinity between target queries and source keys.' },
    beforeAfter: { before: 'Split Q_h and K_h heads', after: 'Raw score matrix S of shape [L_dec, L_enc]' },
    quickCheck: {
      question: 'Is causal masking applied during Cross-Attention?',
      options: ['No, all source tokens are accessible.', 'Yes, future source tokens are masked.', 'Only during training.', 'Only for the first head.'],
      correctIndex: 0,
      explanation: 'Cross-attention is unmasked because the entire source sentence is available to the decoder.',
      distractorNotes: { 1: 'Incorrect. Source tokens are not auto-regressively masked.', 2: 'Incorrect. Unmasked in both training and inference.', 3: 'Incorrect. Unmasked across all heads.' }
    },
    pytorch: [{ id: 'code-dec-cross-qk-matmul', code: 'scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(d_k)' }],
    equationTerms: [{ id: 'eq-dec-cross-qk-matmul', tex: 'S_{\\text{cross}} = \\frac{Q_h K_h^T}{\\sqrt{d_k}} \\in \\mathbb{R}^{L_{\\text{dec}} \\times L_{\\text{enc}}}' }],
    syncMap: [{ vizElementId: 'viz-dec-cross-qk-matmul', equationTermId: 'eq-dec-cross-qk-matmul', codeLineId: 'code-dec-cross-qk-matmul' }],
    narration: [{ duration: '~35s', objective: 'Explain raw cross-attention score computation.', script: 'We multiply Q_h by K_h transposed to get raw alignment scores between target and source words.', audienceQuestion: 'Why is there no causal mask here?', expectedAnswer: 'Because the entire source sentence is already encoded.', misconception: 'Make sure students notice that rows are target tokens and columns are source tokens.', transition: 'Next, we apply Softmax to convert scores into attention weights.' }],
    speakerNotes: { teachingTips: ['Highlight that rows = target tokens and columns = source tokens.'], misconceptions: ['Cross-attention does NOT use causal masking.'], suggestedQuestions: ['Why does cross-attention allow full access to all source tokens?'] }
  },
  'dec-cross-scale-softmax': {
    eyebrow: 'CROSS ATTENTION: SOFTMAX WEIGHTS',
    title: 'Cross Softmax Weights',
    fourQuestions: {
      q1: 'How are cross-attention weights normalized?',
      a1: 'Softmax is applied across columns for each target token row: A[i] = softmax(S[i]).',
      q2: 'What is the sum of weights in each row?',
      a2: 'Exactly 1.0.',
      q3: 'What does weight A[i][j] represent?',
      a3: 'The percentage of attention target token i pays to source token j.',
      q4: 'What shape does the weight matrix have?',
      a4: '[target_seq_len, source_seq_len].'
    },
    body: { beginner: 'Softmax turns raw alignment scores into probability percentages that sum to 100% for each target word.', mtech: 'A_cross = softmax(S_cross, dim=-1) produces probability distribution over source sequence tokens.', research: 'Cross-attention weights visualize word alignment maps between target and source languages.' },
    deepDive: { title: 'Neural Machine Translation Alignment Maps', content: 'Visualizing A_cross reveals learned word-level and phrase-level translation alignments.' },
    whyPanel: { summary: 'Softmax normalization creates a valid probability distribution over source token features.' },
    beforeAfter: { before: 'Raw score matrix S', after: 'Normalized probability matrix A' },
    quickCheck: {
      question: 'What is the sum of attention weights in each row of A_cross?',
      options: ['1.0', '0.0', 'd_model', 'num_heads'],
      correctIndex: 0,
      explanation: 'Softmax normalizes each row so the weights sum to 1.0.',
      distractorNotes: { 1: 'Incorrect. Sum is 1.0, not 0.', 2: 'Incorrect. d_model is feature dimension.', 3: 'Incorrect. num_heads is number of heads.' }
    },
    pytorch: [{ id: 'code-dec-cross-scale-softmax', code: 'weights = F.softmax(scores, dim=-1)' }],
    equationTerms: [{ id: 'eq-dec-cross-scale-softmax', tex: 'A_{\\text{cross}} = \\text{softmax}(S_{\\text{cross}})' }],
    syncMap: [{ vizElementId: 'viz-dec-cross-scale-softmax', equationTermId: 'eq-dec-cross-scale-softmax', codeLineId: 'code-dec-cross-scale-softmax' }],
    narration: [{ duration: '~30s', objective: 'Explain cross-attention Softmax weights.', script: 'Softmax converts raw alignment scores into probability weights that sum to 1.0 for each target word.', audienceQuestion: 'What does a high weight mean?', expectedAnswer: 'The target word strongly focuses on that source word.', misconception: 'Confirm that each row sums to 1.0 independently.', transition: 'Now we use these weights to blend source Value vectors.' }],
    speakerNotes: { teachingTips: ['Show how cross-attention weights act as soft alignment matrices in translation.'], misconceptions: ['Each row sums to 1.0 across source columns.'], suggestedQuestions: ['How do cross-attention heatmaps relate to traditional alignment tables?'] }
  },
  'dec-cross-weighted-sum': {
    eyebrow: 'CROSS ATTENTION: WEIGHTED SUM',
    title: 'Cross Weighted Sum',
    fourQuestions: {
      q1: 'How are source Values blended for each target token?',
      a1: 'O_h = A_cross * V_h_cross of shape [target_seq_len, d_k].',
      q2: 'Where do the Value vectors V_h come from?',
      a2: 'From the encoder output (encoder.ln2_outputs).',
      q3: 'What does the resulting vector contain?',
      a3: 'A weighted combination of source token features relevant to each target token.',
      q4: 'What is the output shape for each head?',
      a4: '[target_seq_len, d_k].'
    },
    body: { beginner: 'We use the attention probabilities to multiply and sum source Value vectors into contextual source representations.', mtech: 'O_h = matmul(A_cross, V_cross_h) blends encoder value vectors into target token positions.', research: 'Cross-attention weighted sums inject source context directly into the target decoding stream.' },
    deepDive: { title: 'Information Transfer from Source to Target', content: 'The weighted sum contracts the source sequence dimension, producing target-aligned source features.' },
    whyPanel: { summary: 'Weighted summation pulls relevant source content into each target token position.' },
    beforeAfter: { before: 'Attention weights A and Encoder Values V', after: 'Blended head output O_h' },
    quickCheck: {
      question: 'Which vectors are multiplied by cross-attention weights A_cross?',
      options: ['Encoder Value vectors (V_cross)', 'Decoder Query vectors (Q_cross)', 'Decoder Key vectors', 'Positional encodings'],
      correctIndex: 0,
      explanation: 'Cross-attention weights multiply Encoder Value vectors V_cross.',
      distractorNotes: { 1: 'Incorrect. Queries are used to compute scores.', 2: 'Incorrect. Decoder keys belong to self-attention.', 3: 'Incorrect. Positional encodings are added earlier.' }
    },
    pytorch: [{ id: 'code-dec-cross-weighted-sum', code: 'head_out = torch.matmul(weights, V_cross_h)' }],
    equationTerms: [{ id: 'eq-dec-cross-weighted-sum', tex: 'O_h = A_{\\text{cross}} V_{h,\\text{cross}} \\in \\mathbb{R}^{L_{\\text{dec}} \\times d_k}' }],
    syncMap: [{ vizElementId: 'viz-dec-cross-weighted-sum', equationTermId: 'eq-dec-cross-weighted-sum', codeLineId: 'code-dec-cross-weighted-sum' }],
    narration: [{ duration: '~35s', objective: 'Explain cross-attention weighted summation.', script: 'We multiply cross-attention weights by Encoder Value vectors to extract source context for each target word.', audienceQuestion: 'Where do the Value vectors come from?', expectedAnswer: 'From the encoder.', misconception: 'Reinforce that source values are being blended into target positions.', transition: 'Let\'s inspect all cross-attention heads simultaneously.' }],
    speakerNotes: { teachingTips: ['Show how source information flows into target vectors via weighted averaging.'], misconceptions: ['Value vectors come from the encoder, not the decoder.'], suggestedQuestions: ['Why are source values blended rather than copied directly?'] }
  },
  'dec-cross-heads-compare': {
    eyebrow: 'CROSS ATTENTION: HEAD COMPARISON',
    title: 'Cross Attention Heads',
    fourQuestions: {
      q1: 'Why do we use multiple cross-attention heads?',
      a1: 'Different heads learn different alignment relationships (e.g., noun-noun, verb-object, syntactic structure).',
      q2: 'Are all cross-attention head heatmaps identical?',
      a2: 'No, each head focuses on different source tokens for the same target token.',
      q3: 'How many heads are active?',
      a3: 'num_heads (typically 4 or 8).',
      q4: 'What shape does each head heatmap have?',
      a4: '[target_seq_len, source_seq_len].'
    },
    body: { beginner: 'Comparing cross-attention heads shows how different heads focus on different parts of the source sentence.', mtech: 'Multi-head cross-attention provides multiple parallel alignment distributions over source tokens.', research: 'Analysis of cross-attention heads reveals specialized linguistic and syntactic translators.' },
    deepDive: { title: 'Head Specialization in Cross Attention', content: 'Some heads perform strict 1-to-1 word translations while others capture multi-word phrase alignments.' },
    whyPanel: { summary: 'Parallel heads capture diverse source-target alignment patterns.' },
    beforeAfter: { before: 'Single head view', after: 'Side-by-side comparison of all cross-attention heads' },
    quickCheck: {
      question: 'What do different cross-attention heads learn?',
      options: ['Different alignment patterns between target and source tokens.', 'Identical duplicate matrices.', 'Random noise.', 'Only punctuation alignment.'],
      correctIndex: 0,
      explanation: 'Multi-head cross-attention allows different heads to learn distinct alignment patterns.',
      distractorNotes: { 1: 'Incorrect. Heads specialize in different relationships.', 2: 'Incorrect. Learned weights differ across heads.', 3: 'Incorrect. Punctuation is only one aspect.' }
    },
    pytorch: [{ id: 'code-dec-cross-heads-compare', code: '# Compare attention weights across all heads' }],
    equationTerms: [{ id: 'eq-dec-cross-heads-compare', tex: 'A^{(h)} = \\text{softmax}\\left(\\frac{Q^{(h)} K^{(h)T}}{\\sqrt{d_k}}\\right)' }],
    syncMap: [{ vizElementId: 'viz-dec-cross-heads-compare', equationTermId: 'eq-dec-cross-heads-compare', codeLineId: 'code-dec-cross-heads-compare' }],
    narration: [{ duration: '~30s', objective: 'Compare all cross-attention heads.', script: 'Observe how each cross-attention head attends to different source words for the same target sentence.', audienceQuestion: 'Do all heads focus on the same source words?', expectedAnswer: 'No, different heads specialize in different relationships.', misconception: 'Highlight head diversity.', transition: 'Now we concatenate all head outputs.' }],
    speakerNotes: { teachingTips: ['Compare heatmaps across heads to demonstrate functional specialization.'], misconceptions: ['Heads are not redundant.'], suggestedQuestions: ['How does multi-head cross-attention improve translation accuracy?'] }
  },
  'dec-cross-concat': {
    eyebrow: 'CROSS ATTENTION: CONCATENATION',
    title: 'Cross Concatenation',
    fourQuestions: {
      q1: 'How are cross-attention head outputs concatenated?',
      a1: 'The num_heads vectors of size d_k are stitched side-by-side into a vector of size d_model.',
      q2: 'What is the resulting tensor shape?',
      a2: '[target_seq_len, d_model].',
      q3: 'Does concatenation change numerical values?',
      a3: 'No, it combines head columns along the d_model dimension.',
      q4: 'Why do we concatenate before output projection?',
      a4: 'To merge feature channels from all heads into a unified vector for Wo_cross projection.'
    },
    body: { beginner: 'We stitch the output vectors from all cross-attention heads back together side-by-side.', mtech: 'Concat(O_1, ..., O_h) recombines split heads into shape [L_dec, d_model].', research: 'Concatenation restores the model dimension d_model prior to linear mixing.' },
    deepDive: { title: 'Head Recombination', content: 'Concatenation joins all head features back into a single [L_dec, d_model] matrix.' },
    whyPanel: { summary: 'Concatenation merges multi-head cross-attention features into a unified matrix.' },
    beforeAfter: { before: 'Split head outputs O_1..O_h', after: 'Concatenated matrix [L_dec, d_model]' },
    quickCheck: {
      question: 'What is the tensor shape after cross-attention concatenation?',
      options: ['[target_seq_len, d_model]', '[source_seq_len, d_model]', '[num_heads, d_k]', '[d_model, d_model]'],
      correctIndex: 0,
      explanation: 'Concatenating num_heads vectors of size d_k yields shape [target_seq_len, d_model].',
      distractorNotes: { 1: 'Incorrect. Rows correspond to target sequence length.', 2: 'Incorrect. Un-split shape is d_model.', 3: 'Incorrect. Weight matrix shape.' }
    },
    pytorch: [{ id: 'code-dec-cross-concat', code: 'concat_out = torch.cat(head_outputs, dim=-1)' }],
    equationTerms: [{ id: 'eq-dec-cross-concat', tex: '\\text{Concat}(O_1, \\dots, O_h) \\in \\mathbb{R}^{L_{\\text{dec}} \\times d_{\\text{model}}}' }],
    syncMap: [{ vizElementId: 'viz-dec-cross-concat', equationTermId: 'eq-dec-cross-concat', codeLineId: 'code-dec-cross-concat' }],
    narration: [{ duration: '~30s', objective: 'Explain cross-attention concatenation.', script: 'We concatenate the outputs of all cross-attention heads side-by-side.', audienceQuestion: 'What is the shape after concatenation?', expectedAnswer: 'Target sequence length by d_model.', misconception: 'Show that head outputs are concatenated along columns.', transition: 'Now we apply the final cross-attention output projection Wo_cross.' }],
    speakerNotes: { teachingTips: ['Demonstrate how column sections from each head form the full d_model vector.'], misconceptions: ['Concatenation does not add values together; it joins them along the column dimension.'], suggestedQuestions: ['Why is concatenation necessary before output projection?'] }
  },
  'dec-cross-output-proj': {
    eyebrow: 'CROSS ATTENTION: OUTPUT PROJECTION',
    title: 'Cross Output Projection',
    fourQuestions: {
      q1: 'Which weight matrix projects concatenated cross-attention outputs?',
      a1: 'Wo_cross of shape [d_model, d_model].',
      q2: 'What is the goal of Wo_cross?',
      a2: 'To linearly mix features extracted from all cross-attention heads.',
      q3: 'What is the output tensor shape?',
      a3: '[target_seq_len, d_model].',
      q4: 'Is bias bo_cross added?',
      a4: 'Yes, element-wise bias bo_cross is added after matrix multiplication.'
    },
    body: { beginner: 'The output projection matrix Wo_cross mixes the features from all cross-attention heads together.', mtech: 'OutputProj_cross = Concat * Wo_cross + bo_cross of shape [L_dec, d_model].', research: 'Wo_cross projects multi-head cross-attention features into the decoder residual stream space.' },
    deepDive: { title: 'Cross Subspace Blending', content: 'Wo_cross allows inter-head feature interactions before adding to the decoder residual stream.' },
    whyPanel: { summary: 'Output projection maps multi-head cross-attention outputs back into the residual stream space.' },
    beforeAfter: { before: 'Concatenated head matrix', after: 'Projected cross-attention output' },
    quickCheck: {
      question: 'What is the role of Wo_cross in Cross-Attention?',
      options: ['To mix features from all cross-attention heads.', 'To normalize token vectors.', 'To calculate Softmax probabilities.', 'To mask future tokens.'],
      correctIndex: 0,
      explanation: 'Wo_cross projects and mixes features from all cross-attention heads.',
      distractorNotes: { 1: 'Incorrect. LayerNorm normalizes vectors.', 2: 'Incorrect. Softmax calculates probabilities.', 3: 'Incorrect. Causal mask applies to self-attention.' }
    },
    pytorch: [{ id: 'code-dec-cross-output-proj', code: 'cross_out = self.Wo_cross(concat_out)' }],
    equationTerms: [{ id: 'eq-dec-cross-output-proj', tex: '\\text{OutputProj}_{\\text{cross}} = \\text{Concat} \\cdot W_{o,\\text{cross}} + b_{o,\\text{cross}}' }],
    syncMap: [{ vizElementId: 'viz-dec-cross-output-proj', equationTermId: 'eq-dec-cross-output-proj', codeLineId: 'code-dec-cross-output-proj' }],
    narration: [{ duration: '~30s', objective: 'Explain cross-attention output projection.', script: 'We multiply concatenated head outputs by Wo_cross to mix features across heads.', audienceQuestion: 'What dimension does Wo_cross have?', expectedAnswer: 'd_model by d_model.', misconception: 'Confirm that Wo_cross maps features back to d_model.', transition: 'Next, we add this output to the second decoder residual connection.' }],
    speakerNotes: { teachingTips: ['Explain that Wo_cross mixes information across all head channels.'], misconceptions: ['Wo_cross is separate from self-attention Wo.'], suggestedQuestions: ['Why does multi-head attention need an output projection?'] }
  },
  'dec-cross-residual-2': {
    eyebrow: 'DECODER OPERATOR 5: SKIP CONNECTION 2',
    title: 'Decoder Residual Connection ②',
    fourQuestions: {
      q1: 'What two inputs are added in Decoder Residual Connection ②?',
      a1: 'decoder.ln1_outputs (input to cross-attention) and decoder.crossAttention.outputProj (output of cross-attention).',
      q2: 'What is the tensor shape?',
      a2: '[target_seq_len, d_model].',
      q3: 'Why is this residual connection crucial?',
      a3: 'It allows target token features from self-attention to flow around cross-attention into subsequent decoder sublayers.',
      q4: 'Does addition change tensor dimensions?',
      a4: 'No, element-wise addition preserves shape [L_dec, d_model].'
    },
    body: { beginner: 'The second residual connection adds the self-attention output back to the cross-attention output.', mtech: 'residual2 = decoder.ln1_outputs + crossAttention.outputProj.', research: 'Residual stream ② accumulates cross-attention source context on top of self-attention target context.' },
    deepDive: { title: 'Dual Residual Streams in Transformer Decoders', content: 'Decoder blocks feature two residual connections: Residual ① around Self-Attention and Residual ② around Cross-Attention.' },
    whyPanel: { summary: 'Residual Connection ② preserves decoder self-attention features alongside cross-attention source updates.' },
    beforeAfter: { before: 'Decoder LN1 state & Cross Output', after: 'Summed Residual ② representation' },
    quickCheck: {
      question: 'What is the input to the second decoder residual skip connection?',
      options: ['decoder.ln1_outputs', 'decoder.xPe', 'encoder.ln2_outputs', 'decoder.embeds'],
      correctIndex: 0,
      explanation: 'The skip connection for Residual ② comes from decoder.ln1_outputs.',
      distractorNotes: { 1: 'Incorrect. decoder.xPe is skip connection for Residual ①.', 2: 'Incorrect. encoder.ln2_outputs is encoder output.', 3: 'Incorrect. decoder.embeds is unencoded input.' }
    },
    pytorch: [{ id: 'code-dec-cross-residual-2', code: 'x = x + self.cross_attn(x, enc_out)' }],
    equationTerms: [{ id: 'eq-dec-cross-residual-2', tex: 'X_{\\text{res2}} = X_{\\text{dec\\_ln1}} + \\text{OutputProj}_{\\text{cross}}' }],
    syncMap: [{ vizElementId: 'viz-dec-cross-residual-2', equationTermId: 'eq-dec-cross-residual-2', codeLineId: 'code-dec-cross-residual-2' }],
    narration: [{ duration: '~30s', objective: 'Explain Decoder Residual Connection ②.', script: 'We add decoder.ln1_outputs back to the cross-attention output projection.', audienceQuestion: 'Where does the skip path come from?', expectedAnswer: 'From decoder.ln1_outputs.', misconception: 'Confirm that both branches have shape [L_dec, d_model].', transition: 'Finally, we normalize with Layer Normalization ②.' }],
    speakerNotes: { teachingTips: ['Trace the two residual paths in a decoder block: Residual 1 (self-attn) and Residual 2 (cross-attn).'], misconceptions: ['Residual 2 uses decoder.ln1_outputs as its skip input, not decoder.xPe.'], suggestedQuestions: ['How do the two residual connections work together in a decoder block?'] }
  },
  'dec-cross-layernorm-2': {
    eyebrow: 'DECODER OPERATOR 6: NORMALIZATION 2',
    title: 'Decoder Layer Normalization ②',
    fourQuestions: {
      q1: 'What is normalized in Decoder Layer Normalization ②?',
      a1: 'The sum from Residual Connection ② (decoder.crossAttention.residual2).',
      q2: 'What are the target statistics before affine scaling?',
      a2: 'Mean = 0.0, Variance = 1.0 for each target token row.',
      q3: 'Which parameters perform affine scaling?',
      a3: 'ln2_gamma_dec and ln2_beta_dec of shape [d_model].',
      q4: 'What is the final output shape?',
      a4: '[target_seq_len, d_model].'
    },
    body: { beginner: 'Layer Normalization ② standardizes target token representations after cross-attention.', mtech: 'ln2_outputs = LayerNorm(residual2) of shape [L_dec, d_model].', research: 'LN2 prepares combined self-attention and cross-attention representations for the Feed Forward Network.' },
    deepDive: { title: 'Decoder Sublayer Normalization Invariants', content: 'LN2 ensures stable activation scaling before vectors enter the Decoder Feed Forward Network.' },
    whyPanel: { summary: 'Layer Normalization ② stabilizes cross-attention augmented representations.' },
    beforeAfter: { before: 'Unnormalized Residual ② sum', after: 'Normalized & scaled LN2 output' },
    quickCheck: {
      question: 'What is the output shape of Decoder Layer Normalization ②?',
      options: ['[target_seq_len, d_model]', '[source_seq_len, d_model]', '[num_heads, d_k]', '[d_model, d_model]'],
      correctIndex: 0,
      explanation: 'Decoder LayerNorm ② outputs shape [target_seq_len, d_model].',
      distractorNotes: { 1: 'Incorrect. Source sequence length belongs to encoder.', 2: 'Incorrect. Head shape.', 3: 'Incorrect. Weight matrix shape.' }
    },
    pytorch: [{ id: 'code-dec-cross-layernorm-2', code: 'x = self.norm2(x)' }],
    equationTerms: [{ id: 'eq-dec-cross-layernorm-2', tex: '\\text{LN}_2(x) = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\cdot \\gamma + \\beta' }],
    syncMap: [{ vizElementId: 'viz-dec-cross-layernorm-2', equationTermId: 'eq-dec-cross-layernorm-2', codeLineId: 'code-dec-cross-layernorm-2' }],
    narration: [{ duration: '~30s', objective: 'Explain Decoder Layer Normalization ②.', script: 'We normalize Residual ② outputs across d_model columns to mean 0 and variance 1.', audienceQuestion: 'What sublayer follows LN2 in the decoder?', expectedAnswer: 'The Decoder Feed Forward Network.', misconception: 'Confirm that LN2 outputs are ready for FFN processing.', transition: 'Stage 4 Decoder Cross-Attention is now complete!' }],
    speakerNotes: { teachingTips: ['Summarize the entire Cross-Attention sublayer block.'], misconceptions: ['LN2 operates per token row.'], suggestedQuestions: ['Why is LN2 applied before the Feed Forward Network?'] }
  },
  'dec-ffn': {
    eyebrow: 'DECODER OPERATOR 7: FEED FORWARD NETWORK',
    title: 'Decoder Feed Forward Network',
    fourQuestions: {
      q1: 'What input enters the Decoder Feed Forward Network?',
      a1: 'The normalized cross-attention output (decoder.crossAttention.ln2_outputs).',
      q2: 'What operations make up the Decoder FFN?',
      a2: 'Linear expansion (W1_dec to d_ff), non-linear activation (ReLU), and Linear projection (W2_dec back to d_model).',
      q3: 'Why do we expand features to d_ff?',
      a3: 'To process and transform token representations in a higher-dimensional feature space.',
      q4: 'Is the FFN applied per token independently?',
      a4: 'Yes, the FFN operates position-wise across each target token identically.'
    },
    body: { beginner: 'The Position-wise Feed Forward Network transforms each target token vector independently through two linear layers and a non-linear activation.', mtech: 'FFN(x) = max(0, x W1_dec + b1_dec) W2_dec + b2_dec of shape [target_seq_len, d_model].', research: 'The FFN sublayer acts as key-value memory storing factual and linguistic knowledge in transformer decoders.' },
    deepDive: { title: 'Position-wise Token Transformations', content: 'Each target token is processed by the exact same FFN weights, allowing independent per-token feature refinement.' },
    whyPanel: { summary: 'The FFN applies non-linear transformations to store and retrieve contextual features.' },
    beforeAfter: { before: 'Decoder LN2 output matrix [L_dec, d_model]', after: 'FFN processed output matrix [L_dec, d_model]' },
    quickCheck: {
      question: 'What is the expansion dimension d_ff in the Decoder Feed Forward Network?',
      options: ['2 * d_model', 'd_model', 'num_heads', 'd_k'],
      correctIndex: 0,
      explanation: 'In this implementation, d_ff is set to 2 * d_model.',
      distractorNotes: { 1: 'Incorrect. d_model is un-expanded dimension.', 2: 'Incorrect. num_heads is number of attention heads.', 3: 'Incorrect. d_k is head size.' }
    },
    pytorch: [{ id: 'code-dec-ffn', code: 'ffn_out = self.w_2(F.relu(self.w_1(dec_ln2)))' }],
    equationTerms: [{ id: 'eq-dec-ffn', tex: '\\text{FFN}(x) = \\max(0, x W_{1,\\text{dec}} + b_{1,\\text{dec}}) W_{2,\\text{dec}} + b_{2,\\text{dec}}' }],
    syncMap: [{ vizElementId: 'viz-dec-ffn', equationTermId: 'eq-dec-ffn', codeLineId: 'code-dec-ffn' }],
    narration: [{ duration: '~30s', objective: 'Explain Decoder Feed Forward Network.', script: 'We pass the cross-attention LN2 representation through two linear transformations with a ReLU activation in between.', audienceQuestion: 'Does the FFN mix information across tokens?', expectedAnswer: 'No, FFN operates position-wise on each token independently.', misconception: 'Clarify that FFN processes rows independently.', transition: 'Next, we apply the third decoder residual connection.' }],
    speakerNotes: { teachingTips: ['Emphasize that the FFN operates position-wise (per token).'], misconceptions: ['FFN does not perform token-to-token attention.'], suggestedQuestions: ['Why does the FFN expand features to d_ff before projecting back to d_model?'] }
  },
  'dec-residual-3': {
    eyebrow: 'DECODER OPERATOR 8: SKIP CONNECTION 3',
    title: 'Decoder Residual Connection ③',
    fourQuestions: {
      q1: 'What two inputs are added in Decoder Residual Connection ③?',
      a1: 'decoder.crossAttention.ln2_outputs (input to FFN) and decoder.ffn.outputs (output of FFN).',
      q2: 'What is the tensor shape?',
      a2: '[target_seq_len, d_model].',
      q3: 'Why is Residual Connection ③ essential?',
      a3: 'It prevents gradient vanishing and preserves cross-attention representation alongside FFN transformations.',
      q4: 'Does addition preserve tensor dimensions?',
      a4: 'Yes, element-wise addition preserves shape [L_dec, d_model].'
    },
    body: { beginner: 'The third residual connection adds the cross-attention representation back to the FFN output.', mtech: 'residual3 = decoder.crossAttention.ln2_outputs + decoder.ffn.outputs.', research: 'Residual stream ③ accumulates FFN non-linear updates onto the combined self/cross-attention decoder state.' },
    deepDive: { title: 'Complete Decoder Residual Highway', content: 'A complete decoder block has 3 residual connections: Residual ① (Self-Attn), Residual ② (Cross-Attn), and Residual ③ (FFN).' },
    whyPanel: { summary: 'Residual Connection ③ integrates FFN outputs into the decoder main highway.' },
    beforeAfter: { before: 'Decoder LN2 state & FFN Output', after: 'Summed Residual ③ representation' },
    quickCheck: {
      question: 'What is the skip connection input to Decoder Residual Connection ③?',
      options: ['decoder.crossAttention.ln2_outputs', 'decoder.ln1_outputs', 'decoder.xPe', 'encoder.ln2_outputs'],
      correctIndex: 0,
      explanation: 'The skip connection for Residual ③ comes from decoder.crossAttention.ln2_outputs.',
      distractorNotes: { 1: 'Incorrect. decoder.ln1_outputs is skip connection for Residual ②.', 2: 'Incorrect. decoder.xPe is skip connection for Residual ①.', 3: 'Incorrect. encoder.ln2_outputs is encoder output.' }
    },
    pytorch: [{ id: 'code-dec-residual-3', code: 'x = x + self.ffn(x)' }],
    equationTerms: [{ id: 'eq-dec-residual-3', tex: 'X_{\\text{res3}} = X_{\\text{cross\\_ln2}} + \\text{FFN}_{\\text{dec}}' }],
    syncMap: [{ vizElementId: 'viz-dec-residual-3', equationTermId: 'eq-dec-residual-3', codeLineId: 'code-dec-residual-3' }],
    narration: [{ duration: '~30s', objective: 'Explain Decoder Residual Connection ③.', script: 'We add decoder.crossAttention.ln2_outputs back to the FFN output.', audienceQuestion: 'How many residual connections exist in a full decoder block?', expectedAnswer: 'Three residual connections.', misconception: 'Confirm that all 3 sublayers have skip connections.', transition: 'Finally, we apply Layer Normalization ③.' }],
    speakerNotes: { teachingTips: ['Review the 3 residual connections of the decoder block.'], misconceptions: ['Each sublayer (self-attn, cross-attn, FFN) has its own residual connection.'], suggestedQuestions: ['Why does the transformer decoder need 3 residual connections while the encoder needs only 2?'] }
  },
  'dec-layernorm-3': {
    eyebrow: 'DECODER OPERATOR 9: NORMALIZATION 3',
    title: 'Decoder Layer Normalization ③',
    fourQuestions: {
      q1: 'What tensor is normalized in Decoder Layer Normalization ③?',
      a1: 'The sum from Residual Connection ③ (decoder.ffn.residual3).',
      q2: 'What are the target statistics before affine scaling?',
      a2: 'Mean = 0.0, Variance = 1.0 for each target token row.',
      q3: 'Which parameters perform affine scaling?',
      a3: 'ln3_gamma_dec and ln3_beta_dec of shape [d_model].',
      q4: 'What is the final output of the Decoder block?',
      a4: 'ln3_outputs of shape [target_seq_len, d_model].'
    },
    body: { beginner: 'Layer Normalization ③ standardizes the final output vectors of the Transformer Decoder block.', mtech: 'ln3_outputs = LayerNorm(residual3) of shape [L_dec, d_model].', research: 'LN3 completes the Transformer Decoder block, producing refined target vectors ready for final linear projection to vocabulary logits.' },
    deepDive: { title: 'Decoder Block Completion', content: 'LN3 ensures stable scaling of the complete decoder block output vectors.' },
    whyPanel: { summary: 'Layer Normalization ③ stabilizes the complete Transformer Decoder block representations.' },
    beforeAfter: { before: 'Unnormalized Residual ③ sum', after: 'Final normalized Decoder block output' },
    quickCheck: {
      question: 'What is the output shape of a complete Transformer Decoder block after LayerNorm ③?',
      options: ['[target_seq_len, d_model]', '[source_seq_len, d_model]', '[vocab_size, d_model]', '[d_model, d_model]'],
      correctIndex: 0,
      explanation: 'The decoder block output preserves shape [target_seq_len, d_model].',
      distractorNotes: { 1: 'Incorrect. Source sequence length belongs to encoder.', 2: 'Incorrect. Vocabulary projection occurs in the linear head.', 3: 'Incorrect. Weight matrix shape.' }
    },
    pytorch: [{ id: 'code-dec-layernorm-3', code: 'x = self.norm3(x)' }],
    equationTerms: [{ id: 'eq-dec-layernorm-3', tex: '\\text{LN}_3(x) = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\cdot \\gamma + \\beta' }],
    syncMap: [{ vizElementId: 'viz-dec-layernorm-3', equationTermId: 'eq-dec-layernorm-3', codeLineId: 'code-dec-layernorm-3' }],
    narration: [{ duration: '~30s', objective: 'Explain Decoder Layer Normalization ③.', script: 'We normalize Residual ③ outputs across d_model columns to mean 0 and variance 1.', audienceQuestion: 'What does this complete?', expectedAnswer: 'A complete Transformer Decoder block!', misconception: 'Highlight that the full decoder block is now mathematically complete.', transition: 'Stage 5 Transformer Decoder block is complete!' }],
    speakerNotes: { teachingTips: ['Celebrate completing the entire Transformer Decoder block.'], misconceptions: ['LN3 output is ready for stacking additional decoder blocks or linear classification head.'], suggestedQuestions: ['What comes after LayerNorm 3 in a full translation model?'] }
  },
  'input-sentence': {
    eyebrow: 'TRANSFORMER SETUP: DUAL INPUT SEQUENCES',
    title: 'Input Sentence Setup',
    fourQuestions: {
      q1: 'What are the two input streams in a sequence-to-sequence Transformer?',
      a1: 'The source sentence fed to the Encoder, and the target sentence fed to the Decoder.',
      q2: 'How are source tokens processed in the Encoder?',
      a2: 'All source tokens are processed simultaneously in parallel.',
      q3: 'How are target tokens processed in the Decoder during inference?',
      a3: 'Sequentially, one token at a time in an auto-regressive loop.',
      q4: 'Can users interactively modify these input sentences?',
      a4: 'Yes, adding or removing words updates all downstream matrices in real time.'
    },
    body: { beginner: 'Transformers take a source sentence for the Encoder and a target sentence for the Decoder to learn translations or sequence tasks.', mtech: 'X_{\\text{enc}} \\in \\mathbb{R}^{L_{\\text{enc}} \\times d_{\\text{model}}}, X_{\\text{dec}} \\in \\mathbb{R}^{L_{\\text{dec}} \\times d_{\\text{model}}}.', research: 'Bilingual tokenization maps source and target text into shared or distinct vocabulary subwords.' },
    deepDive: { title: 'Bilingual Sequence Pair Setup', content: 'Encoder receives unmasked source context while Decoder receives causally masked target context.' },
    whyPanel: { summary: 'Sequence initialization defines the primary tensor dimensions for the entire attention pipeline.' },
    beforeAfter: { before: 'Raw text strings', after: 'Structured source and target token sequences' },
    quickCheck: {
      question: 'Which component processes all input tokens simultaneously in parallel?',
      options: ['The Encoder', 'The Decoder during auto-regressive generation', 'The Softmax classification head', 'The greedy argmax operator'],
      correctIndex: 0,
      explanation: 'The Encoder processes the complete source sequence in parallel.',
      distractorNotes: { 1: 'Incorrect. Decoder inference is sequential.', 2: 'Incorrect. Softmax operates per position.', 3: 'Incorrect. Argmax is a point-wise operator.' }
    },
    pytorch: [{ id: 'code-input-sentence', code: 'src = ["cat", "chased", "dog"]\ntgt = ["the", "dog", "ran"]' }],
    equationTerms: [{ id: 'eq-input-sentence', tex: 'X_{\\text{src}} \\in \\mathbb{R}^{L_{\\text{enc}}}, X_{\\text{tgt}} \\in \\mathbb{R}^{L_{\\text{dec}}}' }],
    syncMap: [{ vizElementId: 'viz-input-sentence', equationTermId: 'eq-input-sentence', codeLineId: 'code-input-sentence' }],
    narration: [{ duration: '~30s', objective: 'Explain dual sequence inputs.', script: 'We initialize the Encoder source sentence and Decoder target sentence.', audienceQuestion: 'Why does the Transformer need two inputs?', expectedAnswer: 'One for the source context, one for the target sequence.', misconception: 'Confirm that source and target can have different sequence lengths.', transition: 'Next, let\'s convert raw words into vocabulary tokens and embeddings.' }],
    speakerNotes: { teachingTips: ['Demonstrate changing words in Interactive Mode.'], misconceptions: ['Source and target sequence lengths do not need to be equal.'], suggestedQuestions: ['Why is the encoder unmasked while the decoder is masked?'] }
  },
  'tokenize': {
    eyebrow: 'TRANSFORMER SETUP: TOKENIZATION & EMBEDDINGS',
    title: 'Tokenization & Vector Lookup',
    fourQuestions: {
      q1: 'What is tokenization?',
      a1: 'Breaking raw text into discrete vocabulary tokens and assigning integer IDs.',
      q2: 'How are vocabulary IDs converted to continuous vectors?',
      a2: 'By looking up rows in an embedding matrix W_embed of shape [vocab_size, d_model].',
      q3: 'What is the dimension of the embedding vectors?',
      a3: 'd_model channels (e.g. 16, 64, 512).',
      q4: 'Why scale embeddings by sqrt(d_model)?',
      a4: 'To balance embedding magnitudes with positional encodings.'
    },
    body: { beginner: 'Words are split into tokens, mapped to dictionary IDs, and converted into d_model dimensional numerical vectors.', mtech: 'E = \\text{Embedding}(X_{\\text{ids}}) \\cdot \\sqrt{d_{\\text{model}}} \\in \\mathbb{R}^{L \\times d_{\\text{model}}}.', research: 'Subword tokenization algorithms like BPE and WordPiece handle out-of-vocabulary words smoothly.' },
    deepDive: { title: 'Embedding Scale Factor', content: 'Vaswani et al. multiply embedding vectors by sqrt(d_model) prior to adding positional encodings.' },
    whyPanel: { summary: 'Token embeddings transform discrete words into dense continuous vector spaces for neural network processing.' },
    beforeAfter: { before: 'Discrete word text', after: 'Continuous d_model dimensional embedding vectors' },
    quickCheck: {
      question: 'What is the shape of an embedding table matrix W_embed?',
      options: ['[vocab_size, d_model]', '[seq_len, d_model]', '[d_model, d_model]', '[vocab_size, seq_len]'],
      correctIndex: 0,
      explanation: 'The embedding table has one d_model row for every token in the vocabulary.',
      distractorNotes: { 1: 'Incorrect. seq_len is sequence length.', 2: 'Incorrect. Square model matrix.', 3: 'Incorrect. Invalid dimensions.' }
    },
    pytorch: [{ id: 'code-tokenize', code: 'embeds = self.embedding(token_ids) * math.sqrt(d_model)' }],
    equationTerms: [{ id: 'eq-tokenize', tex: 'E = W_{\\text{embed}}[X_{\\text{ids}}] \\cdot \\sqrt{d_{\\text{model}}}' }],
    syncMap: [{ vizElementId: 'viz-tokenize', equationTermId: 'eq-tokenize', codeLineId: 'code-tokenize' }],
    narration: [{ duration: '~30s', objective: 'Explain tokenization and embedding lookup.', script: 'Tokens are mapped to vocabulary IDs and retrieved from the embedding matrix as dense d_model vectors.', audienceQuestion: 'Why multiply by sqrt(d_model)?', expectedAnswer: 'To ensure embedding values match the variance of positional encodings.', misconception: 'Confirm vector dimensions.', transition: 'Now let\'s inspect the Residual Stream highway.' }],
    speakerNotes: { teachingTips: ['Explain how token IDs index rows of W_embed.'], misconceptions: ['Token IDs are arbitrary integers; embedding vectors carry semantic meaning.'], suggestedQuestions: ['How does subword tokenization handle rare words?'] }
  },
  'residual-stream': {
    eyebrow: 'TRANSFORMER SETUP: ARCHITECTURAL HIGHWAY',
    title: 'Residual Stream & Normalization Flow',
    fourQuestions: {
      q1: 'What is the residual stream in a Transformer block?',
      a1: 'The continuous vector highway passed through skip connections across sublayers.',
      q2: 'Which sublayers add their outputs back into the residual stream?',
      a2: 'Multi-Head Attention and Position-wise Feed Forward Networks.',
      q3: 'Why are residual connections critical for deep Transformers?',
      a3: 'They enable unimpeded gradient flow during backpropagation and prevent vanishing gradients.',
      q4: 'What is the function of Layer Normalization in the residual highway?',
      a4: 'It stabilizes vector scales and activations across sublayers.'
    },
    body: { beginner: 'The residual stream acts as an information highway where attention and FFN sublayers add updates without destroying original features.', mtech: 'X_{l} = \\text{LayerNorm}(X_{l-1} + \\text{SubLayer}(X_{l-1})).', research: 'Pre-LN vs Post-LN architectural variants offer different training stability trade-offs.' },
    deepDive: { title: 'Residual Stream Representation', content: 'Sublayers write additive updates delta_X onto the persistent residual stream.' },
    whyPanel: { summary: 'Residual connections preserve representation stability across multi-layer deep networks.' },
    beforeAfter: { before: 'Isolated sublayers', after: 'Unified residual stream highway' },
    quickCheck: {
      question: 'Why are skip connections used in every Transformer sublayer?',
      options: ['To allow gradients to flow cleanly and preserve information', 'To reduce vocabulary size', 'To disable attention', 'To speed up tokenization'],
      correctIndex: 0,
      explanation: 'Residual skip connections prevent vanishing gradients and preserve information flow.',
      distractorNotes: { 1: 'Incorrect. Vocabulary size is independent.', 2: 'Incorrect. Attention is active.', 3: 'Incorrect. Tokenization occurs prior.' }
    },
    pytorch: [{ id: 'code-residual-stream', code: 'x = x + self.attn(x)\nx = self.norm1(x)' }],
    equationTerms: [{ id: 'eq-residual-stream', tex: 'X_{\\text{out}} = \\text{LayerNorm}(X_{\\text{in}} + \\text{SubLayer}(X_{\\text{in}}))' }],
    syncMap: [{ vizElementId: 'viz-residual-stream', equationTermId: 'eq-residual-stream', codeLineId: 'code-residual-stream' }],
    narration: [{ duration: '~30s', objective: 'Explain the residual stream highway.', script: 'The residual stream passes information cleanly through addition and LayerNorm operations.', audienceQuestion: 'What happens if we remove skip connections?', expectedAnswer: 'Gradients vanish and deep networks fail to train.', misconception: 'Highlight the additive nature of transformer sublayers.', transition: 'Now let\'s begin Linear Projections for Multi-Head Attention!' }],
    speakerNotes: { teachingTips: ['Visualize the residual stream as a main highway with sublayer offramps.'], misconceptions: ['Sublayers do not replace vectors; they add updates to the stream.'], suggestedQuestions: ['What is the difference between Pre-LN and Post-LN Transformer architectures?'] }
  },
  'decoder-stack': {
    eyebrow: 'TRANSFORMER DECODER: LAYER STACKING',
    title: 'Decoder Stack (Depth N)',
    fourQuestions: {
      q1: 'How are decoder blocks stacked in deep Transformer architectures?',
      a1: 'The output of LayerNorm 3 of block N becomes the input representation to block N+1.',
      q2: 'Does each decoder block re-attend to the same encoder output?',
      a2: 'Yes, every decoder block performs Cross-Attention using the same final encoder output (encoder.ln2_outputs).',
      q3: 'What is the default layer depth N in Vaswani et al. (2017)?',
      a3: 'N = 6 identical decoder blocks.',
      q4: 'Why stack multiple decoder blocks?',
      a4: 'To build increasingly abstract and complex target language representations.'
    },
    body: { beginner: 'Multiple identical decoder blocks are stacked on top of each other, each refining target representations and re-attending to source context.', mtech: 'X_{l} = \\text{DecoderBlock}_l(X_{l-1}, H_{\\text{encoder}}) for l = 1 \\dots N.', research: 'Deep decoder stacks enable complex syntactic restructuring and target language modeling.' },
    deepDive: { title: 'Cross-Attention Persistence Across Stack', content: 'Encoder outputs are computed once and fed into all N decoder blocks in parallel.' },
    whyPanel: { summary: 'Stacking decoder blocks deepens contextual reasoning and auto-regressive generation capability.' },
    beforeAfter: { before: 'Single Decoder Block', after: 'Stacked N-layer Decoder Architecture' },
    quickCheck: {
      question: 'Which encoder tensor is fed to cross-attention in every stacked decoder layer?',
      options: ['encoder.ln2_outputs', 'encoder.xPe', 'encoder.embeddings', 'decoder.ln1_outputs'],
      correctIndex: 0,
      explanation: 'Every decoder layer uses the final encoder output encoder.ln2_outputs for its Keys and Values.',
      distractorNotes: { 1: 'Incorrect. encoder.xPe is un-contextualized.', 2: 'Incorrect. Embeddings lack transformer layer context.', 3: 'Incorrect. decoder.ln1_outputs is internal decoder state.' }
    },
    pytorch: [{ id: 'code-decoder-stack', code: 'for layer in self.decoder_layers:\n    x = layer(x, memory)' }],
    equationTerms: [{ id: 'eq-decoder-stack', tex: 'X^{(N)} = \\text{DecoderStack}(X^{(0)}, H_{\\text{enc}})' }],
    syncMap: [{ vizElementId: 'viz-decoder-stack', equationTermId: 'eq-decoder-stack', codeLineId: 'code-decoder-stack' }],
    narration: [{ duration: '~30s', objective: 'Explain Decoder Stack depth.', script: 'We stack N decoder layers where each layer refines target features while querying the same encoder output.', audienceQuestion: 'Does the encoder re-run for each decoder layer?', expectedAnswer: 'No, encoder outputs are cached and reused across all N layers.', misconception: 'Confirm encoder caching across all decoder layers.', transition: 'Now let\'s project final decoder outputs to vocabulary logits.' }],
    speakerNotes: { teachingTips: ['Demonstrate how layer depth N enhances feature abstraction.'], misconceptions: ['The encoder is NOT re-run for each decoder layer.'], suggestedQuestions: ['Why does cross-attention use the same encoder output across all layers?'] }
  },
  'vocab-projection': {
    eyebrow: 'TRANSFORMER HEAD: LINEAR PROJECTION',
    title: 'Vocabulary Linear Projection',
    fourQuestions: {
      q1: 'Which matrix projects decoder outputs to vocabulary logits?',
      a1: 'W_vocab of shape [d_model, vocab_size].',
      q2: 'What is the output tensor shape of the vocabulary projection?',
      a2: '[target_seq_len, vocab_size].',
      q3: 'What do raw logit values represent?',
      a3: 'Unnormalized scores for every word in the vocabulary at each sequence position.',
      q4: 'Can weight tying be used here?',
      a4: 'Yes, Vaswani et al. share weights between target embedding matrix and W_vocab.'
    },
    body: { beginner: 'The output linear layer maps target d_model vectors into raw score logits for every word in the dictionary.', mtech: 'Z_vocab = X_dec_ln3 * W_vocab + b_vocab of shape [L_dec, |V|].', research: 'Weight tying (sharing W_vocab with target embedding matrix) significantly reduces parameter count and improves generalization.' },
    deepDive: { title: 'Weight Tying Strategy', content: 'Sharing weights between embedding and classification head enforces semantic alignment between input and output tokens.' },
    whyPanel: { summary: 'Vocabulary projection translates d_model vector representations into word score vectors.' },
    beforeAfter: { before: 'Decoder final output [L_dec, d_model]', after: 'Unnormalized logits [L_dec, |V|]' },
    quickCheck: {
      question: 'What is the shape of the vocabulary logit matrix Z_vocab?',
      options: ['[target_seq_len, vocab_size]', '[target_seq_len, d_model]', '[vocab_size, d_model]', '[d_model, d_model]'],
      correctIndex: 0,
      explanation: 'Vocabulary projection maps d_model to vocab_size for each target position.',
      distractorNotes: { 1: 'Incorrect. d_model is feature dimension.', 2: 'Incorrect. Weight matrix shape.', 3: 'Incorrect. Square model matrix.' }
    },
    pytorch: [{ id: 'code-vocab-projection', code: 'logits = self.linear_head(dec_output)' }],
    equationTerms: [{ id: 'eq-vocab-projection', tex: 'Z_{\\text{vocab}} = X_{\\text{dec\\_final}} W_{\\text{vocab}} + b_{\\text{vocab}}' }],
    syncMap: [{ vizElementId: 'viz-vocab-projection', equationTermId: 'eq-vocab-projection', codeLineId: 'code-vocab-projection' }],
    narration: [{ duration: '~30s', objective: 'Explain vocabulary linear projection.', script: 'We project the d_model vector at each target position into raw logits for every word in our vocabulary.', audienceQuestion: 'What is the dimension of W_vocab?', expectedAnswer: 'd_model by vocabulary size.', misconception: 'Clarify that logits are raw, unnormalized scores.', transition: 'Next, we apply Softmax to get probability distributions.' }],
    speakerNotes: { teachingTips: ['Explain weight tying between target embeddings and W_vocab.'], misconceptions: ['Logits are not probabilities; they must be normalized by Softmax.'], suggestedQuestions: ['Why is weight tying beneficial in translation models?'] }
  },
  'softmax-output': {
    eyebrow: 'TRANSFORMER HEAD: PROBABILITY DISTRIBUTION',
    title: 'Vocabulary Softmax Probabilities',
    fourQuestions: {
      q1: 'How are vocabulary probabilities computed?',
      a1: 'Softmax is applied across vocabulary columns for each target token row: P[i] = softmax(Z[i]).',
      q2: 'What is the sum of probabilities across the vocabulary for position i?',
      a2: 'Exactly 1.0 (100%).',
      q3: 'What does P[i][v] represent?',
      a3: 'The probability that the token at position i is vocabulary word v.',
      q4: 'What loss function is paired with these probabilities during training?',
      a4: 'Categorical Cross-Entropy Loss with label smoothing.'
    },
    body: { beginner: 'Softmax converts raw word logits into a probability distribution where all word probabilities for a position sum to 100%.', mtech: 'P(x_t = v | x_<t) = softmax(Z_vocab[t, v]) over vocabulary V.', research: 'Softmax temperature scaling can control generation diversity during sampling.' },
    deepDive: { title: 'Probability Normalization across Vocabulary', content: 'Softmax ensures a valid categorical probability distribution P over thousands of vocabulary tokens.' },
    whyPanel: { summary: 'Softmax produces normalized probabilities for word prediction and cross-entropy loss.' },
    beforeAfter: { before: 'Unnormalized logits Z_vocab', after: 'Normalized probability matrix P' },
    quickCheck: {
      question: 'What is the sum of probabilities across all vocabulary words for a single position?',
      options: ['1.0', '0.0', 'vocab_size', 'd_model'],
      correctIndex: 0,
      explanation: 'Softmax normalizes each token position row so probabilities sum to 1.0.',
      distractorNotes: { 1: 'Incorrect. Sum is 1.0.', 2: 'Incorrect. vocab_size is number of columns.', 3: 'Incorrect. d_model is feature dimension.' }
    },
    pytorch: [{ id: 'code-softmax-output', code: 'probs = F.softmax(logits, dim=-1)' }],
    equationTerms: [{ id: 'eq-softmax-output', tex: 'P(x_t = v \\mid x_{<t}) = \\frac{\\exp(Z_t(v))}{\\sum_{w \\in V} \\exp(Z_t(w))}' }],
    syncMap: [{ vizElementId: 'viz-softmax-output', equationTermId: 'eq-softmax-output', codeLineId: 'code-softmax-output' }],
    narration: [{ duration: '~30s', objective: 'Explain Softmax probability distribution.', script: 'Softmax normalizes raw word scores into probabilities that sum to 1.0 for each position.', audienceQuestion: 'What does each row represent?', expectedAnswer: 'A probability distribution over all vocabulary words.', misconception: 'Confirm row normalization.', transition: 'Now we select the predicted next token using Argmax.' }],
    speakerNotes: { teachingTips: ['Show how high probabilities correspond to top candidate words.'], misconceptions: ['Probabilities are per position row, not per vocabulary column.'], suggestedQuestions: ['How does temperature scaling affect Softmax probabilities?'] }
  },
  'next-token': {
    eyebrow: 'TRANSFORMER HEAD: TOKEN SELECTION',
    title: 'Next-Token Prediction',
    fourQuestions: {
      q1: 'How is the next token selected during greedy decoding?',
      a1: 'predicted_token = argmax_v P(last_token_pos, v).',
      q2: 'What position in the decoder sequence determines the next generated word?',
      a2: 'The last token position (t = L_dec).',
      q3: 'What is Top-K sampling?',
      a3: 'Restricting candidate selection to the K highest probability vocabulary words.',
      q4: 'What is the predicted token in our interactive pipeline?',
      a4: 'The token with the highest Softmax probability at the last sequence position.'
    },
    body: { beginner: 'Greedy decoding selects the word with the highest probability at the final sequence position as the next generated token.', mtech: 'y_t = argmax_{v in V} P(x_t = v | x_<t).', research: 'Beam search and Top-p (nucleus) sampling provide alternative generation strategies to greedy argmax.' },
    deepDive: { title: 'Greedy vs Sampling Decoding Strategies', content: 'Greedy argmax is deterministic, while Top-k and nucleus sampling add creative diversity.' },
    whyPanel: { summary: 'Argmax picks the single most likely token to extend the sequence.' },
    beforeAfter: { before: 'Full vocabulary probability vector', after: 'Selected next token y_t' },
    quickCheck: {
      question: 'Which position in the decoder sequence determines the next token prediction?',
      options: ['The last position (L_dec)', 'The first position (position 0)', 'The middle position', 'All positions averaged'],
      correctIndex: 0,
      explanation: 'In auto-regressive generation, prediction is driven by the representation at the last position L_dec.',
      distractorNotes: { 1: 'Incorrect. Position 0 is the start token.', 2: 'Incorrect. Middle tokens predict historical tokens.', 3: 'Incorrect. Average is not used.' }
    },
    pytorch: [{ id: 'code-next-token', code: 'next_token_id = torch.argmax(probs[:, -1, :], dim=-1)' }],
    equationTerms: [{ id: 'eq-next-token', tex: 'y_t = \\text{argmax}_{v \\in V} P(x_t = v \\mid x_{<t}, H_{\\text{enc}})' }],
    syncMap: [{ vizElementId: 'viz-next-token', equationTermId: 'eq-next-token', codeLineId: 'code-next-token' }],
    narration: [{ duration: '~30s', objective: 'Explain next-token prediction via Argmax.', script: 'We inspect the last sequence position and pick the token with highest probability as our prediction.', audienceQuestion: 'Why do we look at the last position?', expectedAnswer: 'Because the last position aggregates all prior target context.', misconception: 'Clarify that only the last row predicts the next un-generated token.', transition: 'Now let\'s simulate the complete auto-regressive loop.' }],
    speakerNotes: { teachingTips: ['Demonstrate how Argmax selects the top token from the Top-5 distribution.'], misconceptions: ['Prediction looks at the last token row, not all rows.'], suggestedQuestions: ['What is the difference between greedy decoding and beam search?'] }
  },
  'autoregressive-generation': {
    eyebrow: 'TRANSFORMER DECODER: GENERATION LOOP',
    title: 'Autoregressive Generation Loop',
    fourQuestions: {
      q1: 'What defines auto-regressive generation in Transformers?',
      a1: 'Previously predicted tokens are appended to the target sequence and fed back as input for the next step.',
      q2: 'When does auto-regressive generation stop?',
      a2: 'When the model predicts the special End-of-Sequence token (<eos>) or reaches max length.',
      q3: 'Why is auto-regressive generation sequential during inference?',
      a3: 'Because token t cannot be predicted until token t-1 is known.',
      q4: 'Is auto-regressive generation sequential during training?',
      a4: 'No! Training uses teacher forcing with causal masking to compute all positions in parallel.'
    },
    body: { beginner: 'Autoregressive generation predicts tokens one by one, appending each new word to the target sentence until <eos> is emitted.', mtech: 'x_{t+1} = \\text{Decoder}(x_1 \\dots x_t, H_{\\text{enc}}) repeatedly until x_{t+1} = \\text{<eos>}.', research: 'KV-caching avoids re-computing past key-value states, turning O(N^2) decoding step complexity into O(N).' },
    deepDive: { title: 'Inference Efficiency and KV Caching', content: 'Caching Keys and Values from previous decoding steps allows single-token forward passes during inference.' },
    whyPanel: { summary: 'The auto-regressive loop enables full sentence translation and text generation.' },
    beforeAfter: { before: 'Initial target sequence', after: 'Completed generated sentence ending with <eos>' },
    quickCheck: {
      question: 'Why is training parallel while inference is sequential in Transformers?',
      options: ['Training uses teacher forcing with causal masking; inference depends on generated tokens.', 'Inference has no GPUs.', 'Training has no attention layers.', 'Inference ignores encoder outputs.'],
      correctIndex: 0,
      explanation: 'Teacher forcing during training allows parallel forward passes, whereas inference must generate sequentially.',
      distractorNotes: { 1: 'Incorrect. Both use GPUs.', 2: 'Incorrect. Both use attention.', 3: 'Incorrect. Inference relies heavily on encoder outputs.' }
    },
    pytorch: [{ id: 'code-autoregressive-generation', code: 'while next_token != eos_id:\n    out = model(target, memory)\n    next_token = out.argmax(-1)\n    target.append(next_token)' }],
    equationTerms: [{ id: 'eq-autoregressive-generation', tex: 'P(y_1, \\dots, y_M \\mid X) = \\prod_{t=1}^{M} P(y_t \\mid y_{<t}, X)' }],
    syncMap: [{ vizElementId: 'viz-autoregressive-generation', equationTermId: 'eq-autoregressive-generation', codeLineId: 'code-autoregressive-generation' }],
    narration: [{ duration: '~35s', objective: 'Explain auto-regressive generation loop.', script: 'In the auto-regressive loop, each predicted token is appended to the target sequence and fed back into the decoder.', audienceQuestion: 'When does generation stop?', expectedAnswer: 'When <eos> is predicted or max length is reached.', misconception: 'Highlight the contrast between parallel training and sequential inference.', transition: 'The entire Transformer Architecture is now fully implemented and verified!' }],
    speakerNotes: { teachingTips: ['Demonstrate clicking "Step Next Token" to show sequential sequence growth.'], misconceptions: ['Inference cannot be parallelized across sequence steps.'], suggestedQuestions: ['How does KV-caching accelerate auto-regressive inference?'] }
  }
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
