# Lexical Relationship Lab

A small NLP project that preprocesses input text, reports processing and
WordNet-coverage metrics, and evaluates WordNet and an optional LLM against a
labeled teaching benchmark.

## What the project demonstrates

- **Data preprocessing and classical NLP:** sentence tokenization, Penn
  Treebank POS tagging, POS-aware lemmatization, rule-based shallow parsing
  (NP/VP/PP chunks), WordNet sense lookup, and lexical relation checks across
  words extracted from the user's text.
- **Model development and evaluation:** a rule-based WordNet relation classifier
  and accuracy, per-class precision/recall/F1, and macro-F1 on
  `data/evaluation_pairs.csv`.
- **LLM implementation and comparative analysis:** opt-in OpenAI-compatible
  and Gemini API requests, relation-label validation, pairwise agreement, and
  optional benchmark evaluation.
- **Documentation and presentation:** setup/run instructions below and a
  presentation outline in `docs/presentation_outline.md`, plus a standalone
  project workflow diagram in `docs/workflow_diagram.svg`.

## Supported relations and direction

The relation is interpreted from **left to right**:

| Label | Meaning |
|---|---|
| `synonym` | The words share a WordNet synset or are treated as synonyms by the LLM. |
| `hypernym` | Right is a broader category of left (for example, `dog` → `animal`). |
| `hyponym` | Right is a narrower category of left (for example, `animal` → `dog`). |
| `antonym` | Left and right have an antonym link. |
| `meronym` | Right is a part, member, or substance of left (for example, `bird` → `wing`). |
| `holonym` | Right contains left (for example, `wing` → `bird`). |
| `unrelated` | No direct relation from the supported labels was identified. |

WordNet can report multiple relations when different senses support different
links. Hypernym and hyponym search follows transitive WordNet paths; antonym and
part-whole links use direct WordNet edges. Its primary label follows the order
in the relation table above. An LLM prediction is counted as agreeing when its
label is among WordNet's reported relations.

## Text preprocessing and parsing

Paste a sentence or paragraph into **Text to analyze** at the top of the app.
The pipeline:

1. Tokenizes the input with NLTK `word_tokenize`.
2. Assigns Penn Treebank POS tags with NLTK `pos_tag`.
3. Maps each POS tag to a WordNet part of speech and lemmatizes the token.
4. Applies an NLTK `RegexpParser` grammar to produce noun phrase (NP),
   prepositional phrase (PP), and verb phrase (VP) chunks.

The app shows a token/tag/lemma table and the resulting bracketed parse tree.
This grammar-based output is **shallow chunk parsing**, not a full dependency
or constituency parse. It also reports token/word counts, lexical diversity,
lemmatization rate, shallow-parse coverage, POS distribution, and WordNet
coverage for the entered text. WordNet coverage is the fraction of unique
content-word lemma/POS pairs with at least one WordNet sense.

These are intrinsic processing/coverage metrics, not classification accuracy.
A single unlabeled input has no known correct tags or relation labels, so
accuracy, precision, recall, and F1 cannot be computed for that text alone.
Those supervised classification metrics are calculated on the labeled
benchmark below.

## Compare semantic relations in your input

Click **Compare this text: WordNet + LLM** to preprocess the input, extract up
to 12 distinct noun, verb, and adjective lemmas, check every unique pair
against WordNet using each term's POS tag from the input, and send those same
pairs plus the full text context to the selected LLM in one API request. The
POS restriction helps avoid false relations caused by other parts of speech
for an ambiguous word. The app displays both predictions,
explanations, and agreement for each pair. WordNet is a lexical database with
a rule-based relation analyzer; it is not a model trained by this project.

The app compares the LLM's interpretation with WordNet's lexical relations;
agreement is not accuracy against human-verified truth. If no LLM API key is
configured, the same action runs the local WordNet analysis only. This
input-driven comparison does not need labeled data; for accuracy, precision,
recall, and F1 use the separate labeled benchmark. Pair-by-pair results show
the WordNet and LLM labels, plus the LLM explanation and any WordNet evidence.

## Setup

Use Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_nltk_data.py
```

The NLTK download step fetches tokenizer, POS-tagger, and WordNet data used by
the preprocessing and WordNet analysis. The rule-based chunk parser does not
require an additional model download. Run the setup step once before launching
the app.

## Run

```powershell
python -m streamlit run app.py
```

The WordNet analysis works without an LLM key. **Gemini is selected by
default** in the sidebar. Create a key in
[Google AI Studio](https://aistudio.google.com/apikey), then paste it into the
sidebar or set `GEMINI_API_KEY`. With no key, the app shows a short warning;
after you enter a key, it confirms that the key was entered and enables LLM
actions. The app checks whether the key works only when you make a request.
The default model is `gemini-3.5-flash-lite`;
change it in the sidebar or set `GEMINI_MODEL` if another model is available to
your account. Google offers a limited free tier for eligible models; model
availability, quotas, and terms can change. Check the current
[Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) before use.
Google states that free-tier content may be used to improve its products, so do
not send sensitive or confidential data.

The existing OpenAI-compatible option remains available. Enter its API key in
the app sidebar or set `OPENAI_API_KEY`. The default endpoint is
`https://api.openai.com/v1` and the default model is `gpt-4o-mini`. For another
OpenAI-compatible service, set `OPENAI_BASE_URL` and `OPENAI_MODEL`, or enter
these values in the sidebar. API keys are not written to disk by the app.

LLM requests are sent only after choosing an LLM action. Input-driven semantic
comparison uses one request containing the text and extracted pairs. Benchmark
evaluation sends one request per labeled row. Free-tier quotas or provider
charges may apply.

## Appearance

The app uses the light Streamlit theme configured in
`.streamlit/config.toml`, with dark text and a light background for readable
contrast. If you change the theme configuration while the app is running,
restart Streamlit for the new settings to take effect. If text still appears
faded after restarting, refresh the browser with `Ctrl+F5`.

## Evaluation

Select **Evaluate WordNet baseline** to run the bundled labeled pairs locally.
Configure an API key to enable **Evaluate LLM**. The app displays accuracy,
macro-F1, per-class precision/recall/F1, and row-level predictions. After both
runs complete, **Comprehensive comparison, error analysis, and conclusions**
reports side-by-side scores, per-class metrics, prediction agreement, each
model's unique correct examples, per-model and shared error counts, row-level
error patterns, confusion matrices, and data-derived conclusions. Results are
aligned and compared only when both classifiers have predictions for the same
ordered benchmark rows.

The benchmark is intentionally small and hand-labeled for demonstration. It is
not a statistically representative evaluation; results can vary with WordNet
coverage, sense ambiguity, benchmark composition, and LLM/provider behavior.
The error analysis and conclusions describe this benchmark only; they do not
establish statistical significance or production performance.

Run the unit tests with:

```powershell
python -m unittest discover -s tests -v
```

## Project layout

```text
app.py                       Streamlit UI and comparative workflow
lexical_analyzer.py          Preprocessing, WordNet relations, LLM client
evaluation.py                Benchmark loading and classification metrics
data/evaluation_pairs.csv    Small labeled evaluation set
download_nltk_data.py        One-time NLTK resource setup
tests/                       Focused unit tests
docs/presentation_outline.md Suggested project presentation slides
.streamlit/config.toml         Streamlit light theme and color settings
docs/workflow_diagram.svg      Visual workflow diagram for both app workflows
```
