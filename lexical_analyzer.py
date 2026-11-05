"""WordNet-based lexical relation analysis and LLM comparison helpers."""

from __future__ import annotations

from collections import Counter
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import nltk
import requests
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

RELATIONS = (
    "synonym",
    "hypernym",
    "hyponym",
    "antonym",
    "meronym",
    "holonym",
    "unrelated",
)
_WORDNET_POS = {
    "J": wn.ADJ,
    "N": wn.NOUN,
    "R": wn.ADV,
    "V": wn.VERB,
}
MAX_TEXT_RELATION_TERMS = 12
_CHUNK_PARSER = nltk.RegexpParser(
    r"""
    NP: {<DT>?<JJ.*>*<NN.*>+}
    PP: {<IN><NP>}
    VP: {<VB.*><RB.*>*<NP|PP>*}
    """
)
_RELATION_METHODS = {
    "meronym": ("part_meronyms", "member_meronyms", "substance_meronyms"),
    "holonym": ("part_holonyms", "member_holonyms", "substance_holonyms"),
}
_LEMMATIZER = WordNetLemmatizer()


def _wordnet_pos_for_tag(tag: str) -> str | None:
    """Convert a Penn Treebank tag prefix to WordNet POS, if supported."""
    return _WORDNET_POS.get(tag[:1])


def _correct_contextual_pos_tags(
    tagged_tokens: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Correct a few common NLTK tagger ambiguities using local grammar context."""
    corrected = list(tagged_tokens)
    third_person_pronouns = {"he", "she", "it"}
    forms_of_be = {"am", "is", "are", "was", "were", "be", "been", "being"}

    for index, (token, tag) in enumerate(tagged_tokens):
        word = token.lower()
        previous_word = (
            tagged_tokens[index - 1][0].lower() if index > 0 else ""
        )
        previous_tag = tagged_tokens[index - 1][1] if index > 0 else ""
        next_tag = (
            tagged_tokens[index + 1][1]
            if index + 1 < len(tagged_tokens)
            else ""
        )

        if (
            previous_word in forms_of_be
            and tag.startswith("R")
            and wn.synsets(word, pos=wn.ADJ)
        ):
            corrected[index] = (token, "JJ")
        elif (
            tag.startswith("J")
            and previous_tag.startswith("N")
            and next_tag.startswith("N")
            and wn.synsets(word, pos=wn.VERB)
        ):
            corrected[index] = (token, "VBP")
        elif (
            tag.startswith("N")
            and previous_word in third_person_pronouns
            and previous_tag == "PRP"
            and word.endswith("s")
            and wn.synsets(word, pos=wn.VERB)
        ):
            corrected[index] = (token, "VBZ")

    return corrected


@dataclass(frozen=True)
class PreprocessedTerm:
    original: str
    lemma: str
    pos: str


@dataclass(frozen=True)
class PreprocessedToken:
    token: str
    pos: str
    lemma: str


@dataclass(frozen=True)
class TextAnalysis:
    tokens: tuple[PreprocessedToken, ...]
    parse_tree: str
    chunk_count: int
    chunked_word_count: int


@dataclass(frozen=True)
class TextMetrics:
    token_count: int
    word_count: int
    unique_lemmas: int
    lexical_diversity: float
    lemmatized_word_count: int
    lemmatization_rate: float
    content_word_count: int
    wordnet_content_terms: int
    wordnet_covered_terms: int
    wordnet_coverage: float
    chunk_count: int
    chunked_word_count: int
    chunk_coverage: float
    pos_distribution: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class TextWordPair:
    left: str
    right: str
    wordnet_relations: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class TextRelationAnalysis:
    terms: tuple[str, ...]
    pairs: tuple[TextWordPair, ...]


@dataclass(frozen=True)
class WordNetAnalysis:
    left: PreprocessedTerm
    right: PreprocessedTerm
    relations: tuple[str, ...]
    evidence: tuple[str, ...]
    left_senses: tuple[str, ...]
    right_senses: tuple[str, ...]

    @property
    def primary_relation(self) -> str:
        return next((relation for relation in RELATIONS if relation in self.relations), "unrelated")


@dataclass(frozen=True)
class LLMAnalysis:
    relation: str
    explanation: str
    confidence: float | None = None


@dataclass(frozen=True)
class LLMTextRelation:
    left: str
    right: str
    relation: str
    explanation: str


def preprocess_text(text: str) -> TextAnalysis:
    """Tokenize, POS-tag, lemmatize, and shallow-parse free text."""
    if not text.strip():
        raise ValueError("Enter text to preprocess.")

    try:
        tokens = nltk.word_tokenize(text.strip())
    except LookupError as error:
        raise RuntimeError(
            "The NLTK tokenizer data is missing. Run `python download_nltk_data.py`."
        ) from error
    if not tokens:
        raise ValueError("Enter text containing at least one token.")

    try:
        tagged_tokens = nltk.pos_tag(tokens)
    except LookupError as error:
        raise RuntimeError(
            "The NLTK POS-tagger data is missing. Run `python download_nltk_data.py`."
        ) from error

    try:
        tagged_tokens = _correct_contextual_pos_tags(tagged_tokens)
    except LookupError as error:
        raise RuntimeError(
            "The NLTK WordNet data is missing. Run `python download_nltk_data.py`."
        ) from error

    processed_tokens: list[PreprocessedToken] = []
    for token, tag in tagged_tokens:
        wordnet_pos = _wordnet_pos_for_tag(tag)
        try:
            lemma = (
                _LEMMATIZER.lemmatize(token.lower(), wordnet_pos)
                if wordnet_pos is not None
                else token.lower()
            )
        except LookupError as error:
            raise RuntimeError(
                "The NLTK WordNet data is missing. Run `python download_nltk_data.py`."
            ) from error
        processed_tokens.append(
            PreprocessedToken(token=token, pos=tag, lemma=lemma)
        )

    parsed_tree = _CHUNK_PARSER.parse(tagged_tokens)
    chunks = [
        child
        for child in parsed_tree
        if isinstance(child, nltk.Tree) and child.label() in {"NP", "VP", "PP"}
    ]
    chunked_word_count = sum(
        token.isalpha()
        for chunk in chunks
        for token, _tag in chunk.leaves()
    )
    return TextAnalysis(
        tokens=tuple(processed_tokens),
        parse_tree=parsed_tree.pformat(margin=100),
        chunk_count=len(chunks),
        chunked_word_count=chunked_word_count,
    )


def evaluate_text_analysis(analysis: TextAnalysis) -> TextMetrics:
    """Calculate intrinsic processing metrics; these are not classifier scores."""
    words = [token for token in analysis.tokens if token.token.isalpha()]
    unique_lemmas = {token.lemma for token in words}
    lemmatized_word_count = sum(
        token.token.lower() != token.lemma for token in words
    )
    content_tokens = [
        token for token in words if token.pos[:1] in {"N", "V", "J"}
    ]
    content_terms = {
        (token.lemma, wordnet_pos)
        for token in content_tokens
        if (wordnet_pos := _wordnet_pos_for_tag(token.pos)) is not None
    }

    covered_terms = 0
    try:
        for lemma, pos in content_terms:
            if wn.synsets(lemma, pos=pos):
                covered_terms += 1
    except LookupError as error:
        raise RuntimeError(
            "The NLTK WordNet corpus is missing. Run `python download_nltk_data.py`."
        ) from error

    pos_categories = Counter(
        {
            "Noun": 0,
            "Verb": 0,
            "Adjective": 0,
            "Adverb": 0,
            "Determiner": 0,
            "Punctuation": 0,
            "Other": 0,
        }
    )
    for token in analysis.tokens:
        if not any(character.isalnum() for character in token.token):
            category = "Punctuation"
        elif token.pos.startswith("D"):
            category = "Determiner"
        else:
            category = {
                wn.NOUN: "Noun",
                wn.VERB: "Verb",
                wn.ADJ: "Adjective",
                wn.ADV: "Adverb",
            }.get(_wordnet_pos_for_tag(token.pos), "Other")
        pos_categories[category] += 1

    word_count = len(words)
    content_term_count = len(content_terms)
    return TextMetrics(
        token_count=len(analysis.tokens),
        word_count=word_count,
        unique_lemmas=len(unique_lemmas),
        lexical_diversity=len(unique_lemmas) / word_count if word_count else 0.0,
        lemmatized_word_count=lemmatized_word_count,
        lemmatization_rate=(
            lemmatized_word_count / word_count if word_count else 0.0
        ),
        content_word_count=len(content_tokens),
        wordnet_content_terms=content_term_count,
        wordnet_covered_terms=covered_terms,
        wordnet_coverage=(
            covered_terms / content_term_count if content_term_count else 0.0
        ),
        chunk_count=analysis.chunk_count,
        chunked_word_count=analysis.chunked_word_count,
        chunk_coverage=(
            analysis.chunked_word_count / word_count if word_count else 0.0
        ),
        pos_distribution=tuple(pos_categories.items()),
    )


def analyze_text_with_wordnet(analysis: TextAnalysis) -> TextRelationAnalysis:
    """Find WordNet relations among content-word lemmas in one input text."""
    preprocessed_terms: dict[str, PreprocessedTerm] = {}
    for token in analysis.tokens:
        if (
            token.token.isalpha()
            and token.pos[:1] in {"N", "V", "J"}
            and token.lemma not in {"be", "do"}
        ):
            preprocessed_terms.setdefault(
                token.lemma,
                PreprocessedTerm(
                    original=token.token,
                    lemma=token.lemma,
                    pos=token.pos,
                ),
            )
    terms = tuple(preprocessed_terms)
    if len(terms) < 2:
        raise ValueError(
            "Enter text containing at least two distinct nouns, verbs, or adjectives "
            "to compare semantic relations."
        )
    if len(terms) > MAX_TEXT_RELATION_TERMS:
        raise ValueError(
            f"This text contains {len(terms)} distinct content words. "
            f"Shorten it to at most {MAX_TEXT_RELATION_TERMS} so all word pairs "
            "can be compared consistently."
        )

    pairs = tuple(
        TextWordPair(
            left=left,
            right=right,
            wordnet_relations=(
                result := _analyze_preprocessed_terms(
                    preprocessed_terms[left],
                    preprocessed_terms[right],
                )
            ).relations,
            evidence=result.evidence,
        )
        for index, left in enumerate(terms)
        for right in terms[index + 1 :]
    )
    return TextRelationAnalysis(terms=terms, pairs=pairs)


def preprocess_term(text: str) -> PreprocessedTerm:
    """Tokenize, POS-tag, and lemmatize one input term."""
    try:
        tokens = nltk.word_tokenize(text.strip())
    except LookupError as error:
        raise RuntimeError(
            "The NLTK tokenizer data is missing. Run `python download_nltk_data.py`."
        ) from error
    if len(tokens) != 1 or not tokens[0].isalpha():
        raise ValueError("Enter exactly one alphabetic word for each term.")

    token = tokens[0].lower()
    try:
        tagged_tokens = nltk.pos_tag([token])
    except LookupError as error:
        raise RuntimeError(
            "The NLTK POS-tagger data is missing. Run `python download_nltk_data.py`."
        ) from error

    tag = tagged_tokens[0][1]
    wordnet_pos = _wordnet_pos_for_tag(tag)
    try:
        lemma = (
            _LEMMATIZER.lemmatize(token, wordnet_pos)
            if wordnet_pos is not None
            else token
        )
        surface_synsets = wn.synsets(token)
    except LookupError as error:
        raise RuntimeError(
            "The NLTK WordNet data is missing. Run `python download_nltk_data.py`."
        ) from error
    if any(
        token == name.lower().replace("_", " ")
        for synset in surface_synsets
        for name in synset.lemma_names()
    ):
        lemma = token
    return PreprocessedTerm(original=text.strip(), lemma=lemma, pos=tag)


def _lemma_names(synsets: list[Any]) -> set[str]:
    return {
        lemma.name().lower().replace("_", " ")
        for synset in synsets
        for lemma in synset.lemmas()
    }


def analyze_with_wordnet(
    left_text: str,
    right_text: str,
    *,
    left_pos: str | None = None,
    right_pos: str | None = None,
) -> WordNetAnalysis:
    """Find direct WordNet relations, oriented from the left term to the right."""
    left = preprocess_term(left_text)
    right = preprocess_term(right_text)
    return _analyze_preprocessed_terms(
        left,
        right,
        left_pos=left_pos,
        right_pos=right_pos,
    )


def _analyze_preprocessed_terms(
    left: PreprocessedTerm,
    right: PreprocessedTerm,
    *,
    left_pos: str | None = None,
    right_pos: str | None = None,
) -> WordNetAnalysis:
    """Analyze terms that already have sentence-level POS and lemmas."""
    left_wordnet_pos = _wordnet_pos_for_tag(
        left.pos if left_pos is None else left_pos
    )
    right_wordnet_pos = _wordnet_pos_for_tag(
        right.pos if right_pos is None else right_pos
    )
    try:
        left_synsets = wn.synsets(left.lemma, pos=left_wordnet_pos)
        right_synsets = wn.synsets(right.lemma, pos=right_wordnet_pos)
    except LookupError as error:
        raise RuntimeError(
            "The NLTK WordNet corpus is missing. Run `python download_nltk_data.py`."
        ) from error

    found: dict[str, set[str]] = {relation: set() for relation in RELATIONS if relation != "unrelated"}
    right_names = _lemma_names(right_synsets)
    for left_synset in left_synsets:
        if left_synset in right_synsets:
            found["synonym"].add(left_synset.name())

        for left_lemma in left_synset.lemmas():
            for antonym in left_lemma.antonyms():
                if antonym.name().lower().replace("_", " ") in right_names:
                    found["antonym"].add(f"{left_lemma.name()} ↔ {antonym.name()}")

        for relation, methods in _RELATION_METHODS.items():
            for method_name in methods:
                related_synsets = getattr(left_synset, method_name)()
                if any(synset in right_synsets for synset in related_synsets):
                    found[relation].add(f"{left_synset.name()} → {method_name}")

        if any(
            synset in right_synsets
            for synset in left_synset.closure(lambda current: current.hypernyms())
        ):
            found["hypernym"].add(f"{left_synset.name()} → {right.lemma}")
        if any(
            synset in right_synsets
            for synset in left_synset.closure(lambda current: current.hyponyms())
        ):
            found["hyponym"].add(f"{left_synset.name()} → {right.lemma}")

    relations = tuple(relation for relation in RELATIONS if found.get(relation))
    evidence = tuple(
        f"{relation}: {item}"
        for relation in RELATIONS
        if relation in found
        for item in sorted(found[relation])
    )
    return WordNetAnalysis(
        left=left,
        right=right,
        relations=relations or ("unrelated",),
        evidence=evidence,
        left_senses=tuple(synset.definition() for synset in left_synsets[:5]),
        right_senses=tuple(synset.definition() for synset in right_synsets[:5]),
    )


def _request_llm_json(
    prompt: str,
    user_payload: dict[str, Any],
    *,
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    """Send one JSON request to a configured provider and parse its response."""
    provider_name = provider.strip().lower()
    if provider_name not in {"openai", "gemini"}:
        raise ValueError("Provider must be 'openai' or 'gemini'.")

    env_key = "GEMINI_API_KEY" if provider_name == "gemini" else "OPENAI_API_KEY"
    key = api_key or os.getenv(env_key)
    if not key:
        raise ValueError(f"Set an API key in the app or the {env_key} environment variable.")

    if provider_name == "gemini":
        model_name = model or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"
        if not re.fullmatch(r"[A-Za-z0-9._-]+", model_name):
            raise ValueError("Gemini model names may contain only letters, numbers, '.', '_' and '-'.")
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(model_name, safe='')}:generateContent"
        )
    else:
        endpoint = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        endpoint = endpoint if endpoint.endswith("/chat/completions") else f"{endpoint}/chat/completions"
        model_name = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    if provider_name == "gemini":
        response = requests.post(
            endpoint,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": prompt}]},
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": json.dumps(user_payload)}],
                    }
                ],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
            },
            timeout=timeout,
        )
    else:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model_name,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
            },
            timeout=timeout,
        )
    response.raise_for_status()
    payload = response.json()
    try:
        if provider_name == "gemini":
            content = "".join(
                part["text"]
                for part in payload["candidates"][0]["content"]["parts"]
                if isinstance(part.get("text"), str)
            )
        else:
            content = payload["choices"][0]["message"]["content"]
        result = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("The LLM response did not contain valid relation JSON.") from error
    if not isinstance(result, dict):
        raise ValueError("The LLM response must be a JSON object.")
    return result


def analyze_with_llm(
    left: str,
    right: str,
    *,
    provider: str = "openai",
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 30,
) -> LLMAnalysis:
    """Ask OpenAI-compatible or Gemini API for one lexical relation."""
    provider_name = provider.strip().lower()
    result = _request_llm_json(
        (
            "Classify the lexical semantic relation from LEFT to RIGHT. "
            "Use exactly one label: synonym, hypernym, hyponym, antonym, meronym, holonym, unrelated. "
            "Direction: hypernym means RIGHT is a broader category of LEFT; "
            "hyponym means RIGHT is a narrower category of LEFT; "
            "meronym means RIGHT is a part/member/substance of LEFT; holonym means RIGHT contains LEFT. "
            "Choose unrelated when no clear direct lexical relation is established. "
            "Return a JSON object with relation, explanation, and confidence (0 to 1)."
        ),
        {"left": left, "right": right},
        provider=provider_name,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
    )
    relation = result.get("relation")
    explanation = result.get("explanation")
    confidence = result.get("confidence")
    if relation not in RELATIONS or not isinstance(explanation, str):
        raise ValueError("The LLM response must include a supported relation and explanation.")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError("The LLM confidence must be a number from 0 to 1.")
    return LLMAnalysis(
        relation=relation,
        explanation=explanation.strip(),
        confidence=float(confidence) if confidence is not None else None,
    )


def analyze_text_with_llm(
    text: str,
    word_pairs: tuple[tuple[str, str], ...],
    *,
    provider: str = "openai",
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 60,
) -> tuple[LLMTextRelation, ...]:
    """Classify all supplied input-text word pairs in a single LLM request."""
    if not text.strip():
        raise ValueError("Enter text to analyze.")
    if not word_pairs:
        raise ValueError("At least one word pair is required for LLM comparison.")
    if len(set(word_pairs)) != len(word_pairs):
        raise ValueError("Word pairs must be unique.")

    prompt = (
        "Analyze the supplied text and classify the lexical semantic relation "
        "for every supplied word pair. Use the text as context, but do not "
        "invent, omit, reorder, or alter any supplied pair. Return exactly one "
        "result per pair. Supported labels: synonym, hypernym, hyponym, antonym, "
        "meronym, holonym, unrelated. Direction: hypernym means RIGHT is broader "
        "than LEFT; hyponym means RIGHT is narrower than LEFT; meronym means "
        "RIGHT is part/member/substance of LEFT; holonym means RIGHT contains "
        "LEFT. Use unrelated when there is no clear direct relation. Return a "
        'JSON object: {"relations":[{"left":"...","right":"...","relation":"...",'
        '"explanation":"..."}]}.'
    )
    result = _request_llm_json(
        prompt,
        {"text": text, "word_pairs": [{"left": left, "right": right} for left, right in word_pairs]},
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
    )
    raw_relations = result.get("relations")
    if not isinstance(raw_relations, list):
        raise ValueError("The LLM response must include a relations list.")

    expected_pairs = set(word_pairs)
    parsed: dict[tuple[str, str], LLMTextRelation] = {}
    for item in raw_relations:
        if not isinstance(item, dict):
            raise ValueError("Each LLM relation must be a JSON object.")
        left = item.get("left")
        right = item.get("right")
        relation = item.get("relation")
        explanation = item.get("explanation")
        pair = (left, right)
        if pair not in expected_pairs:
            raise ValueError("The LLM response contains a pair that was not supplied.")
        if pair in parsed:
            raise ValueError("The LLM response contains a duplicate word pair.")
        if relation not in RELATIONS or not isinstance(explanation, str):
            raise ValueError(
                "Each LLM relation must include a supported label and explanation."
            )
        parsed[pair] = LLMTextRelation(
            left=left,
            right=right,
            relation=relation,
            explanation=explanation.strip(),
        )
    if set(parsed) != expected_pairs:
        raise ValueError("The LLM response must classify every supplied word pair.")
    return tuple(parsed[pair] for pair in word_pairs)
