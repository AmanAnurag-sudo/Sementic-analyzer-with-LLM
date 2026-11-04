# Presentation Outline

## Slide 1 — Problem and objective

- Semantic relations help explain how lexical items are connected.
- Objective: compare a classical lexical resource (WordNet) with an LLM on the
  same word pairs.

## Slide 2 — Preprocessing and parsing

- Tokenize sentence text with NLTK `word_tokenize`.
- Assign Penn Treebank tags with NLTK `pos_tag`.
- Use the tag to select a WordNet POS category before lemmatization.
- Apply an NLTK `RegexpParser` grammar to identify NP, PP, and VP chunks.
- Show the token/tag/lemma table and bracketed parse tree in the app.
- Note that the app uses an explicit light theme with dark text for readability.
- Report input-level lexical diversity, lemmatization rate, parse coverage,
  POS distribution, and WordNet term coverage.
- Extract content-word pairs from the input, compare WordNet relations with an
  LLM's labels for the same pairs, and inspect pairwise agreement.
- Clarify that rule-based chunking is shallow parsing, not a full syntactic
  parser.
- Explain that these intrinsic metrics describe processing; labeled examples
  are required to calculate classifier accuracy, precision, recall, and F1.
- Explain that WordNet is a lexical database, not a trained model, and that
  agreement with it is not ground-truth accuracy.

## Slide 3 — System workflow

1. Paste a sentence or paragraph into the text analysis input.
2. Click **Compare this text: WordNet + LLM** to preprocess it, extract
   content-word pairs, analyze them with WordNet, and send the same pairs and
   full text to the selected LLM in one request.
3. Inspect processing metrics, tokenization, POS tags, lemmas, and shallow
   parsing.
4. Run WordNet and LLM evaluations against the same labeled benchmark.
5. Review the comprehensive comparison, error analysis, confusion matrices,
   and conclusions.

## Slide 4 — Classical NLP and WordNet

- NLTK handles preprocessing and WordNet access.
- The analyzer checks synonym, hypernym, hyponym, antonym, meronym, and holonym
  links.
- Relations are directional; for example, `dog` → `animal` is labeled a
  hypernym relation because the right word is a broader category.

## Slide 5 — LLM approach

- A constrained prompt asks for a supported relation label, explanation, and
  confidence.
- The response is parsed and validated before it is shown or scored.
- The user opts in to each LLM action; the configured endpoint receives the
  submitted word pair or input text and its extracted pair list.
- Gemini is the default provider. Without a key, the app displays a short
  warning; after a key is entered, it confirms entry and enables LLM actions.
  The key is actually checked by the provider when a request is made.

## Slide 6 — Evaluation and comparison

- Use the labeled pairs in `data/evaluation_pairs.csv`.
- Report accuracy, per-class precision/recall/F1, and macro-F1.
- Compare both classifiers on the exact same rows; report agreement and
  model-specific correct examples.
- Inspect per-class scores, per-model and shared error counts, row-level error
  patterns, and confusion matrices.
- Present the benchmark conclusions alongside their supporting metrics.
- Draw conclusions from observed results and explain their limitations; the
  benchmark is small and hand-labeled, so do not claim statistical significance.

## Slide 7 — Demonstration and limitations

- Demonstrate the preprocessing output and explain the NP/VP/PP chunks.
- Demonstrate a known benchmark relation such as `dog` → `animal`, then review
  a disagreement or shared error from comparative analysis.
- Discuss WordNet sense coverage and ambiguity.
- Discuss LLM variation, endpoint/API dependence, and benchmark limitations.
- Possible next step: expand and independently validate the evaluation set.
