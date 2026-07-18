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
