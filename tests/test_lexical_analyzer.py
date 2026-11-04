import unittest
from unittest.mock import patch

from evaluation import calculate_metrics, compare_classifiers
from lexical_analyzer import (
    LLMAnalysis,
    PreprocessedToken,
    PreprocessedTerm,
    RELATIONS,
    TextAnalysis,
    TextWordPair,
    WordNetAnalysis,
    analyze_text_with_llm,
    analyze_text_with_wordnet,
    analyze_with_llm,
    analyze_with_wordnet,
    evaluate_text_analysis,
    preprocess_text,
    preprocess_term,
)


class FakeLemma:
    def __init__(self, name, antonyms=()):
        self._name = name
        self._antonyms = [FakeLemma(antonym) for antonym in antonyms]

    def name(self):
        return self._name

    def antonyms(self):
        return self._antonyms


class FakeSynset:
    def __init__(self, name, lemmas=(), definition=""):
        self._name = name
        self._lemmas = [FakeLemma(lemma) for lemma in lemmas]
        self._definition = definition
        self._hypernyms = []
        self._hyponyms = []
        self._part_meronyms = []
        self._part_holonyms = []

    def name(self):
        return self._name

    def lemmas(self):
        return self._lemmas

    def definition(self):
        return self._definition

    def hypernyms(self):
        return self._hypernyms

    def hyponyms(self):
        return self._hyponyms

    def part_meronyms(self):
        return self._part_meronyms

    def member_meronyms(self):
        return []

    def substance_meronyms(self):
        return []

    def part_holonyms(self):
        return self._part_holonyms

    def closure(self, relation):
        return iter(relation(self))

    def member_holonyms(self):
        return []

    def substance_holonyms(self):
        return []


class LexicalAnalyzerTests(unittest.TestCase):
    @patch(
        "lexical_analyzer.nltk.pos_tag",
        return_value=[
            ("The", "DT"),
            ("quick", "JJ"),
            ("foxes", "NNS"),
            ("jump", "VBP"),
            ("quickly", "RB"),
            (".", "."),
        ],
    )
    @patch(
        "lexical_analyzer._LEMMATIZER.lemmatize",
        side_effect=lambda token, _pos: {"foxes": "fox"}.get(token, token),
    )
    @patch(
        "lexical_analyzer.nltk.word_tokenize",
        return_value=["The", "quick", "foxes", "jump", "quickly", "."],
    )
    def test_preprocess_text_tokenizes_tags_lemmatizes_and_parses(
        self, tokenize, _lemmatize, tag
    ):
        result = preprocess_text("The quick foxes jump quickly.")

        self.assertEqual(
            [(token.token, token.pos, token.lemma) for token in result.tokens],
            [
                ("The", "DT", "the"),
                ("quick", "JJ", "quick"),
                ("foxes", "NNS", "fox"),
                ("jump", "VBP", "jump"),
                ("quickly", "RB", "quickly"),
                (".", ".", "."),
            ],
        )
        self.assertIn("NP", result.parse_tree)
        self.assertIn("VP", result.parse_tree)
        tokenize.assert_called_once_with("The quick foxes jump quickly.")
        tag.assert_called_once_with(["The", "quick", "foxes", "jump", "quickly", "."])

    def test_preprocess_text_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "Enter text"):
            preprocess_text("  ")

    @patch("lexical_analyzer.wn.synsets")
    def test_text_metrics_measure_processing_and_wordnet_coverage(self, synsets):
        synsets.side_effect = lambda lemma, pos: [object()] if lemma == "dog" else []
        analysis = TextAnalysis(
            tokens=(
                PreprocessedToken("Dogs", "NNS", "dog"),
                PreprocessedToken("run", "VBP", "run"),
                PreprocessedToken(".", ".", "."),
            ),
            parse_tree="(S (NP (NNS Dogs)) (VP (VBP run)) (. .))",
            chunk_count=2,
            chunked_word_count=2,
        )

        metrics = evaluate_text_analysis(analysis)

        self.assertEqual(metrics.token_count, 3)
        self.assertEqual(metrics.word_count, 2)
        self.assertEqual(metrics.unique_lemmas, 2)
        self.assertEqual(metrics.lemmatized_word_count, 1)
        self.assertAlmostEqual(metrics.lemmatization_rate, 0.5)
        self.assertEqual(metrics.wordnet_content_terms, 2)
        self.assertEqual(metrics.wordnet_covered_terms, 1)
        self.assertAlmostEqual(metrics.wordnet_coverage, 0.5)
        self.assertEqual(metrics.chunk_count, 2)
        self.assertAlmostEqual(metrics.chunk_coverage, 1.0)

    @patch("lexical_analyzer.analyze_with_wordnet")
    def test_text_wordnet_analysis_checks_all_unique_content_word_pairs(self, analyze):
        analyze.return_value = WordNetAnalysis(
            left=PreprocessedTerm("dog", "dog", "NN"),
            right=PreprocessedTerm("animal", "animal", "NN"),
            relations=("hypernym",),
            evidence=("hypernym: dog.n.01 → animal",),
            left_senses=(),
            right_senses=(),
        )
        analysis = TextAnalysis(
            tokens=(
                PreprocessedToken("A", "DT", "a"),
                PreprocessedToken("dogs", "NNS", "dog"),
                PreprocessedToken("are", "VBP", "be"),
                PreprocessedToken("loyal", "JJ", "loyal"),
                PreprocessedToken("animal", "NN", "animal"),
                PreprocessedToken("dog", "NN", "dog"),
            ),
            parse_tree="(S)",
            chunk_count=0,
            chunked_word_count=0,
        )

        result = analyze_text_with_wordnet(analysis)

        self.assertEqual(result.terms, ("dog", "loyal", "animal"))
        self.assertEqual(len(result.pairs), 3)
        self.assertEqual(
            [(pair.left, pair.right) for pair in result.pairs],
            [("dog", "loyal"), ("dog", "animal"), ("loyal", "animal")],
        )
        self.assertTrue(all(pair.wordnet_relations == ("hypernym",) for pair in result.pairs))
    @patch("lexical_analyzer.analyze_with_wordnet")
    def test_text_wordnet_analysis_uses_input_pos_for_each_pair(self, analyze):
        analyze.return_value = WordNetAnalysis(
            left=PreprocessedTerm("complex", "complex", "JJ"),
            right=PreprocessedTerm("concept", "concept", "NN"),
            relations=("unrelated",),
            evidence=(),
            left_senses=(),
            right_senses=(),
        )
        analysis = TextAnalysis(
            tokens=(
                PreprocessedToken("complex", "JJ", "complex"),
                PreprocessedToken("concept", "NN", "concept"),
            ),
            parse_tree="(S (NP (JJ complex) (NN concept)))",
            chunk_count=1,
            chunked_word_count=2,
        )

        analyze_text_with_wordnet(analysis)

        analyze.assert_called_once_with(
            "complex", "concept", left_pos="JJ", right_pos="NN"
        )

    def test_text_wordnet_analysis_rejects_too_many_terms(self):
        analysis = TextAnalysis(
            tokens=tuple(
                PreprocessedToken(chr(97 + index) * 2, "NN", chr(97 + index) * 2)
                for index in range(13)
            ),
            parse_tree="(S)",
            chunk_count=0,
            chunked_word_count=0,
        )
        with self.assertRaisesRegex(ValueError, "at most 12"):
            analyze_text_with_wordnet(analysis)

    @patch("lexical_analyzer.nltk.pos_tag", return_value=[("cars", "NNS")])
    @patch("lexical_analyzer._LEMMATIZER.lemmatize", return_value="car")
    @patch("lexical_analyzer.nltk.word_tokenize", return_value=["cars"])
    def test_preprocess_term_tokenizes_tags_and_lemmatizes(
        self, _tokenize, _lemmatize, _tag
    ):
        result = preprocess_term("cars")
        self.assertEqual((result.lemma, result.pos), ("car", "NNS"))
        _tokenize.assert_called_once_with("cars")
        _lemmatize.assert_called_once_with("cars", "n")
        _tag.assert_called_once_with(["cars"])

    @patch("lexical_analyzer.nltk.word_tokenize", return_value=["ice", "cream"])
    def test_preprocess_rejects_multiple_tokens(self, _tokenize):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            preprocess_term("ice cream")
        _tokenize.assert_called_once_with("ice cream")

    @patch("lexical_analyzer.preprocess_term")
    @patch("lexical_analyzer.wn.synsets")
    def test_wordnet_detects_directional_hypernym(self, synsets, preprocess):
        from lexical_analyzer import PreprocessedTerm

        dog = FakeSynset("dog.n.01", ["dog"], "a domesticated canine")
        animal = FakeSynset("animal.n.01", ["animal"], "a living organism")
        dog._hypernyms = [animal]
        preprocess.side_effect = [
            PreprocessedTerm("dog", "dog", "NN"),
            PreprocessedTerm("animal", "animal", "NN"),
        ]
        synsets.side_effect = [[dog], [animal]]

        result = analyze_with_wordnet("dog", "animal")

        self.assertIn("hypernym", result.relations)
        self.assertEqual(result.primary_relation, "hypernym")

    def test_metrics_compute_accuracy_and_macro_f1(self):
        metrics = calculate_metrics(
            ["synonym", "antonym", "unrelated"],
            ["synonym", "unrelated", "unrelated"],
        )
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertIn("macro_f1", metrics)
        self.assertEqual(metrics["count"], 3)

    def test_classifier_comparison_reports_paired_errors_and_conclusions(self):
        wordnet_rows = [
            {
                "left": "car",
                "right": "automobile",
                "expected_relation": "synonym",
                "predicted_relation": "synonym",
            },
            {
                "left": "dog",
                "right": "animal",
                "expected_relation": "hypernym",
                "predicted_relation": "unrelated",
            },
            {
                "left": "hot",
                "right": "cold",
                "expected_relation": "antonym",
                "predicted_relation": "antonym",
            },
            {
                "left": "table",
                "right": "banana",
                "expected_relation": "unrelated",
                "predicted_relation": "synonym",
            },
        ]
        llm_rows = [
            {**wordnet_rows[0]},
            {**wordnet_rows[1], "predicted_relation": "hypernym"},
            {**wordnet_rows[2]},
            {**wordnet_rows[3], "predicted_relation": "unrelated"},
        ]

        result = compare_classifiers(wordnet_rows, llm_rows)

        self.assertEqual(result["count"], 4)
        self.assertEqual(result["agreement_count"], 2)
        self.assertEqual(result["wordnet_only_correct"], 0)
        self.assertEqual(result["llm_only_correct"], 2)
        self.assertEqual(result["wordnet_error_count"], 2)
        self.assertEqual(result["llm_error_count"], 0)
        self.assertEqual(result["shared_error_count"], 0)
        self.assertEqual(len(result["error_rows"]), 2)
        self.assertEqual(result["error_rows"][0]["error_pattern"], "WordNet only wrong")
        self.assertEqual(result["error_rows"][1]["error_pattern"], "WordNet only wrong")
        self.assertEqual(len(result["per_class_comparison"]), 4)
        self.assertEqual(len(result["wordnet_confusion"]), len(RELATIONS))
        self.assertTrue(any("small hand-labeled benchmark" in item for item in result["conclusions"]))

    def test_classifier_comparison_rejects_misaligned_rows(self):
        row = {
            "left": "car",
            "right": "automobile",
            "expected_relation": "synonym",
            "predicted_relation": "synonym",
        }
        other_row = {**row, "right": "vehicle"}
        with self.assertRaisesRegex(ValueError, "do not match"):
            compare_classifiers([row], [other_row])

    @patch("lexical_analyzer.requests.post")
    def test_llm_response_is_validated(self, post):
        response = post.return_value
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"relation":"synonym","explanation":"Both name a vehicle.",'
                            '"confidence":0.95}'
                        )
                    }
                }
            ]
        }

        result = analyze_with_llm("car", "automobile", api_key="test-key")

        self.assertEqual(result, LLMAnalysis("synonym", "Both name a vehicle.", 0.95))
        self.assertIn(result.relation, RELATIONS)
        self.assertIn("LEFT to RIGHT", post.call_args.kwargs["json"]["messages"][0]["content"])
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-key",
        )
        response.raise_for_status.assert_called_once()

    @patch("lexical_analyzer.requests.post")
    def test_text_llm_classifies_all_supplied_pairs_in_one_request(self, post):
        post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"relations":['
                            '{"left":"dog","right":"animal","relation":"hypernym",'
                            '"explanation":"An animal is a broader category."},'
                            '{"left":"dog","right":"loyal","relation":"unrelated",'
                            '"explanation":"No direct lexical relation."}'
                            ']}'
                        )
                    }
                }
            ]
        }

        result = analyze_text_with_llm(
            "A dog is loyal, and every dog is an animal.",
            (("dog", "animal"), ("dog", "loyal")),
            api_key="test-key",
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].relation, "hypernym")
        self.assertEqual(result[1].relation, "unrelated")
        self.assertEqual(post.call_count, 1)
        user_payload = post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertIn("A dog is loyal", user_payload)
        self.assertIn('"right": "animal"', user_payload)

    @patch("lexical_analyzer.requests.post")
    def test_text_llm_rejects_missing_pair_results(self, post):
        post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"relations":[{"left":"dog","right":"animal",'
                            '"relation":"hypernym","explanation":"A category relation."}]}'
                        )
                    }
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "every supplied"):
            analyze_text_with_llm(
                "A dog is an animal.",
                (("dog", "animal"), ("dog", "loyal")),
                api_key="test-key",
            )

    @patch("lexical_analyzer.requests.post")
    def test_gemini_response_uses_generate_content_api(self, post):
        response = post.return_value
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    '{"relation":"hypernym","explanation":"A car is a vehicle.",'
                                    '"confidence":0.9}'
                                )
                            }
                        ]
                    }
                }
            ]
        }

        result = analyze_with_llm(
            "car",
            "vehicle",
            provider="gemini",
            api_key="test-key",
        )

        self.assertEqual(result.relation, "hypernym")
        self.assertEqual(result.confidence, 0.9)
        args, kwargs = post.call_args
        self.assertEqual(
            args[0],
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-3.5-flash-lite:generateContent",
        )
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "test-key")
        self.assertEqual(
            kwargs["json"]["generationConfig"]["responseMimeType"],
            "application/json",
        )
        response.raise_for_status.assert_called_once()

    def test_gemini_requires_its_own_api_key(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                analyze_with_llm("car", "vehicle", provider="gemini", api_key="")

    @patch("lexical_analyzer.requests.post")
    def test_llm_rejects_unknown_relation(self, post):
        post.return_value.json.return_value = {
            "choices": [{"message": {"content": '{"relation":"similar","explanation":"close"}'}}]
        }
        with self.assertRaisesRegex(ValueError, "supported relation"):
            analyze_with_llm("car", "vehicle", api_key="test-key")


if __name__ == "__main__":
    unittest.main()
