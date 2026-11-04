"""Interactive WordNet and LLM lexical-relation comparison app."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

from evaluation import (
    calculate_metrics,
    compare_classifiers,
    evaluate_classifier,
    load_benchmark,
)
from lexical_analyzer import (
    RELATIONS,
    analyze_text_with_llm,
    analyze_text_with_wordnet,
    analyze_with_llm,
    evaluate_text_analysis,
    preprocess_text,
)

st.set_page_config(page_title="Lexical Relationship Lab", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --ink: #17243b;
        --muted: #64748b;
        --line: #e2e8f0;
        --accent: #4968d8;
        --surface: #ffffff;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(86, 115, 224, 0.09), transparent 34rem),
            #f5f7fb;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }
    .hero {
        padding: 2rem 2.2rem;
        margin: 0 0 1.8rem;
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 22px;
        color: #fff;
        background: linear-gradient(118deg, #17243b 0%, #263c67 58%, #4968d8 100%);
        box-shadow: 0 18px 45px rgba(32, 54, 100, .15);
    }
    .hero-kicker {
        color: #c8d4ff;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: .35rem 0 .45rem;
        color: #fff;
        font-size: clamp(2rem, 4vw, 2.8rem);
        letter-spacing: -.04em;
    }
    .hero p {
        max-width: 760px;
        margin: 0;
        color: #e0e7ff;
        font-size: 1.04rem;
    }
    h2, h3 { color: var(--ink); letter-spacing: -.02em; }
    [data-testid="stCaptionContainer"] p { color: var(--muted); }
    [data-testid="stTextArea"] textarea {
        border: 1px solid #d5deed;
        border-radius: 13px;
        background: #fff;
        line-height: 1.65;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(73, 104, 216, .12);
    }
    [data-testid="stMetric"] {
        min-height: 104px;
        padding: 1rem 1.05rem;
        border: 1px solid var(--line);
        border-radius: 15px;
        background: var(--surface);
        box-shadow: 0 5px 16px rgba(30, 48, 82, .045);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stBaseButton-primary"] {
        border: 0;
        border-radius: 11px;
        background: linear-gradient(105deg, #425fd0, #5b77e2);
        box-shadow: 0 7px 16px rgba(73, 104, 216, .18);
    }
    [data-testid="stBaseButton-secondary"] { border-radius: 11px; }
    [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: #fff;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
        background: #fbfcff;
    }
    @media (max-width: 700px) {
        .block-container { padding: 1rem .85rem 2rem; }
        .hero { padding: 1.4rem; border-radius: 17px; }
    }
    </style>
    <section class="hero">
      <div class="hero-kicker">Natural Language Processing | WordNet + LLM</div>
      <h1>Lexical Relationship Lab</h1>
      <p>Explore your text, inspect its linguistic structure, and compare
      WordNet’s lexical evidence with an LLM’s interpretation.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("LLM settings")
    provider = st.selectbox("Provider", ("Gemini", "OpenAI-compatible"))
    is_gemini = provider == "Gemini"
    env_api_key = os.getenv("GEMINI_API_KEY" if is_gemini else "OPENAI_API_KEY", "")
    api_key = st.text_input(
        "Gemini API key" if is_gemini else "OpenAI-compatible API key",
        value=env_api_key,
        type="password",
        key="gemini_api_key" if is_gemini else "openai_api_key",
        help="Used only when you click an LLM action. It is not saved by this app.",
    ).strip()
    if api_key:
        st.success(
            f"{provider} API key entered. LLM actions are enabled; "
            "the key will be checked when a request is made."
        )
    else:
        st.warning("API key not entered. LLM comparison is unavailable.")
    if is_gemini:
        base_url = None
        model = st.text_input(
            "Gemini model",
            value=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            key="gemini_model",
        )
        st.markdown("[Get a Gemini API key](https://aistudio.google.com/apikey)")
        st.caption(
            "Google AI Studio offers a limited free tier for eligible models; "
            "quotas, model availability, and terms may change."
        )
        st.warning(
            "Google may use free-tier prompts to improve its products. "
            "Do not send sensitive or confidential data."
        )
    else:
        base_url = st.text_input(
            "OpenAI-compatible API base URL",
            value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    st.caption(
        "Your input text or benchmark word pairs are sent to the selected "
        "provider only when you request an LLM action."
    )

st.subheader("01 - Process your text")
st.caption(
    "Paste a sentence or paragraph to inspect tokenization, POS tags, "
    "POS-aware lemmas, and a rule-based shallow parse."
)
preprocessing_text = st.text_area(
    "Text to analyze",
    value="A dog is an animal. A car is a vehicle.",
    key="preprocessing_text",
)
st.caption(
    "The comparison uses content-word pairs extracted from this text. "
    "When an API key is configured, the same pairs and full text context are "
    "sent to the selected LLM in one request."
)
run_text_analysis = st.button(
    "Compare this text: WordNet + LLM" if api_key else "Analyze this text with WordNet",
    type="primary",
)
if run_text_analysis:
    for key in (
        "text_analysis",
        "text_analysis_input",
        "text_metrics",
        "text_relation_analysis",
        "text_relation_input",
        "text_llm_relations",
        "text_llm_input",
    ):
        st.session_state.pop(key, None)
    try:
        text_analysis = preprocess_text(preprocessing_text)
        text_metrics = evaluate_text_analysis(text_analysis)
        relation_analysis = analyze_text_with_wordnet(text_analysis)
        st.session_state["text_analysis"] = text_analysis
        st.session_state["text_analysis_input"] = preprocessing_text
        st.session_state["text_metrics"] = text_metrics
        st.session_state["text_relation_analysis"] = relation_analysis
        st.session_state["text_relation_input"] = preprocessing_text
    except (ValueError, RuntimeError) as error:
        st.error(str(error))
    else:
        if api_key and relation_analysis.pairs:
            try:
                llm_relations = analyze_text_with_llm(
                    preprocessing_text,
                    tuple((pair.left, pair.right) for pair in relation_analysis.pairs),
                    provider="gemini" if is_gemini else "openai",
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                )
            except (ValueError, RuntimeError) as error:
                st.error(f"LLM comparison failed: {error}")
            except requests.HTTPError as error:
                if error.response is not None and error.response.status_code == 429:
                    st.error(
                        "LLM quota or rate limit reached (HTTP 429). Check the "
                        "provider's quota/billing or retry after its limit resets."
                    )
                else:
                    st.error(f"LLM request failed: {error}")
            except requests.RequestException as error:
                st.error(f"LLM request failed: {error}")
            else:
                st.session_state["text_llm_relations"] = llm_relations
                st.session_state["text_llm_input"] = preprocessing_text
        elif api_key:
            st.info(
                "This text needs at least two distinct content-word lemmas "
                "to compare semantic relationships."
            )

text_analysis = st.session_state.get("text_analysis")
if (
    text_analysis is not None
    and st.session_state.get("text_analysis_input") == preprocessing_text
):
    text_metrics = st.session_state.get("text_metrics")
    if text_metrics is not None:
        st.markdown("#### Processing metrics")
        st.caption(
            "These describe this text's processing and WordNet coverage. "
            "Accuracy, precision, recall, and F1 require labeled expected outputs "
            "and are reported separately for the benchmark."
        )
        metric_columns = st.columns(4)
        metric_columns[0].metric("Tokens", text_metrics.token_count)
        metric_columns[1].metric("Words", text_metrics.word_count)
        metric_columns[2].metric("Unique lemmas", text_metrics.unique_lemmas)
        metric_columns[3].metric(
            "Lexical diversity", f"{text_metrics.lexical_diversity:.1%}"
        )
        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Lemmatized words", text_metrics.lemmatized_word_count
        )
        metric_columns[1].metric(
            "Lemmatization rate", f"{text_metrics.lemmatization_rate:.1%}"
        )
        metric_columns[2].metric(
            "WordNet term coverage",
            f"{text_metrics.wordnet_covered_terms}/{text_metrics.wordnet_content_terms}",
            f"{text_metrics.wordnet_coverage:.1%}",
        )
        metric_columns[3].metric(
            "Shallow parse coverage", f"{text_metrics.chunk_coverage:.1%}"
        )
        with st.expander("Part-of-speech distribution"):
            pos_counts = pd.DataFrame(
                [
                    {"POS category": category, "Token count": count}
                    for category, count in text_metrics.pos_distribution
                ]
            ).set_index("POS category")
            st.bar_chart(pos_counts, use_container_width=True)
    with st.expander("Inspect tokens and lemmas"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Token": token.token, "POS tag": token.pos, "Lemma": token.lemma}
                    for token in text_analysis.tokens
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with st.expander("View shallow parse tree"):
        st.code(text_analysis.parse_tree, language="text")
        st.caption(
            "The NP/VP/PP chunks are produced by an NLTK rule-based chunk parser; "
            "this is shallow parsing, not a full syntactic dependency parser."
        )

    st.divider()
    st.subheader("02 - Compare semantic relationships")
    st.caption(
        "WordNet is a lexical database, not a trained model. It checks every "
        "pair of extracted content words against WordNet; the selected LLM "
        "checks the same pairs using your full text as context."
    )
    st.caption(
        f"To keep one LLM request manageable and compare the same pairs, "
        f"the text may contain at most 12 distinct content-word lemmas "
        f"({12 * 11 // 2} pairs)."
    )
    relation_analysis = st.session_state.get("text_relation_analysis")
    relation_analysis_is_current = (
        relation_analysis is not None
        and st.session_state.get("text_relation_input") == preprocessing_text
    )

    if relation_analysis_is_current:
        st.write("**WordNet content terms:** " + ", ".join(relation_analysis.terms))
        st.caption(
            f"WordNet checked {len(relation_analysis.pairs)} distinct term pairs."
        )
        if not api_key:
            st.caption(
                "WordNet results are shown below. Configure an API key and run "
                "the analysis again to compare with the LLM."
            )

        llm_relations = st.session_state.get("text_llm_relations")
        llm_results_are_current = (
            llm_relations is not None
            and st.session_state.get("text_llm_input") == preprocessing_text
        )
        llm_by_pair = (
            {(item.left, item.right): item for item in llm_relations}
            if llm_results_are_current
            else {}
        )
        if llm_results_are_current:
            agreement_count = sum(
                llm_by_pair[(pair.left, pair.right)].relation
                in pair.wordnet_relations
                for pair in relation_analysis.pairs
            )
            wordnet_related_count = sum(
                pair.wordnet_relations != ("unrelated",)
                for pair in relation_analysis.pairs
            )
            llm_related_count = sum(
                llm_item.relation != "unrelated"
                for llm_item in llm_relations
            )
            result_metrics = st.columns(3)
            result_metrics[0].metric(
                "Pairs where WordNet found a relation", wordnet_related_count
            )
            result_metrics[1].metric(
                "Pairs where LLM found a relation", llm_related_count
            )
            result_metrics[2].metric(
                "Relation agreement",
                f"{agreement_count}/{len(relation_analysis.pairs)}",
                (
                    f"{agreement_count / len(relation_analysis.pairs):.1%}"
                    if relation_analysis.pairs
                    else None
                ),
            )
        if relation_analysis.pairs:
            st.markdown("#### Pair-by-pair results")
            for pair in relation_analysis.pairs:
                llm_item = llm_by_pair.get((pair.left, pair.right))
                with st.container(border=True):
                    pair_heading, agreement_column = st.columns([4, 1])
                    pair_heading.markdown(
                        f"##### {pair.left}  →  {pair.right}"
                    )
                    if llm_item is not None:
                        if llm_item.relation in pair.wordnet_relations:
                            agreement_column.success("Agree")
                        else:
                            agreement_column.warning("Differ")
                    wordnet_column, llm_column = st.columns(2)
                    wordnet_column.markdown("**WordNet**")
                    wordnet_column.write(", ".join(pair.wordnet_relations))
                    llm_column.markdown("**LLM**")
                    llm_column.write(
                        llm_item.relation if llm_item is not None else "Not compared"
                    )
                    if llm_item is not None and llm_item.explanation:
                        st.caption(llm_item.explanation)
                    if pair.evidence:
                        with st.expander("WordNet evidence"):
                            for evidence in pair.evidence:
                                st.write(f"- {evidence}")
        else:
            st.info("No content-word pairs were found in this text.")
        if llm_results_are_current:
            st.caption(
                "Agreement means the LLM's single label is among WordNet's "
                "relations for that pair. This is a comparison, not accuracy "
                "against human-verified labels."
            )

st.divider()
st.subheader("03 - Evaluate on a labeled benchmark")
st.caption(
    "The bundled labeled pairs are a small teaching benchmark. WordNet metrics run locally; "
    "LLM evaluation sends each pair to the selected provider. Gemini free-tier quotas may apply; "
    "paid API usage may incur provider costs."
)
try:
    benchmark = load_benchmark()
except (OSError, ValueError) as error:
    st.error(f"Could not load benchmark: {error}")
else:
    with st.expander(f"View benchmark examples ({len(benchmark)} pairs)"):
        st.dataframe(pd.DataFrame(benchmark), use_container_width=True, hide_index=True)
    eval_wordnet_col, eval_llm_col = st.columns(2)
    with eval_wordnet_col:
        evaluate_wordnet = st.button("Evaluate WordNet baseline")
    with eval_llm_col:
        evaluate_llm = st.button(
            f"Evaluate LLM ({len(benchmark)} requests)",
            disabled=not bool(api_key),
            help="Configure the selected provider's API key to enable LLM benchmark evaluation.",
        )

    if evaluate_wordnet:
        st.session_state.pop("wordnet_metrics", None)
        st.session_state.pop("wordnet_rows", None)
        try:
            wordnet_metrics, wordnet_rows = evaluate_classifier(benchmark)
        except (ValueError, RuntimeError) as error:
            st.error(f"WordNet evaluation failed: {error}")
        else:
            st.session_state["wordnet_metrics"] = wordnet_metrics
            st.session_state["wordnet_rows"] = wordnet_rows

    if evaluate_llm:
        st.session_state.pop("llm_metrics", None)
        st.session_state.pop("llm_rows", None)
        st.session_state.pop("llm_run_config", None)
        llm_progress = st.progress(0, text="Evaluating LLM benchmark...")
        llm_rows: list[dict[str, str]] = []
        try:
            for index, row in enumerate(benchmark):
                prediction = analyze_with_llm(
                    row["left"],
                    row["right"],
                    provider="gemini" if is_gemini else "openai",
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                )
                llm_rows.append({**row, "predicted_relation": prediction.relation})
                llm_progress.progress((index + 1) / len(benchmark), text=f"Evaluated {index + 1}/{len(benchmark)}")
        except (ValueError, RuntimeError) as error:
            llm_progress.empty()
            st.error(f"LLM evaluation stopped: {error}")
        except requests.RequestException as error:
            llm_progress.empty()
            st.error(f"LLM request failed; evaluation stopped: {error}")
        else:
            st.session_state["llm_metrics"] = calculate_metrics(
                (row["expected_relation"] for row in llm_rows),
                (row["predicted_relation"] for row in llm_rows),
            )
            st.session_state["llm_rows"] = llm_rows
            st.session_state["llm_run_config"] = {
                "provider": "Gemini" if is_gemini else "OpenAI-compatible",
                "model": model,
                "base_url": base_url,
            }
            llm_progress.empty()

    metric_columns = st.columns(2)
    for column, key, label in (
        (metric_columns[0], "wordnet", "WordNet"),
        (metric_columns[1], "llm", "LLM"),
    ):
        metrics = st.session_state.get(f"{key}_metrics")
        rows = st.session_state.get(f"{key}_rows")
        if metrics is not None and rows is not None:
            with column:
                st.markdown(f"**{label}**")
                st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                st.metric("Macro F1", f"{metrics['macro_f1']:.1%}")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.dataframe(
                    pd.DataFrame.from_dict(metrics["per_class"], orient="index"),
                    use_container_width=True,
                )

    wordnet_rows = st.session_state.get("wordnet_rows")
    llm_rows = st.session_state.get("llm_rows")
    if wordnet_rows is not None and llm_rows is not None:
        try:
            comparison = compare_classifiers(wordnet_rows, llm_rows)
        except ValueError as error:
            st.error(f"Could not compare classifier results: {error}")
        else:
            st.divider()
            st.subheader("Comprehensive comparison, error analysis, and conclusions")
            llm_run_config = st.session_state.get("llm_run_config", {})
            if llm_run_config:
                st.caption(
                    f"LLM run: {llm_run_config['provider']} / "
                    f"{llm_run_config['model']}. Both classifiers were scored "
                    f"on the same {comparison['count']} benchmark rows."
                )
            st.write(
                "The following comparison includes overall scores, per-class "
                "performance, prediction agreement, error cases, and confusion "
                "matrices."
            )

            wordnet_metrics = comparison["wordnet_metrics"]
            llm_metrics = comparison["llm_metrics"]
            score_columns = st.columns(4)
            score_columns[0].metric(
                "WordNet accuracy",
                f"{wordnet_metrics['accuracy']:.1%}",
            )
            score_columns[1].metric(
                "LLM accuracy",
                f"{llm_metrics['accuracy']:.1%}",
                f"{comparison['accuracy_delta']:+.1%} vs WordNet",
            )
            score_columns[2].metric(
                "WordNet macro-F1",
                f"{wordnet_metrics['macro_f1']:.1%}",
            )
            score_columns[3].metric(
                "LLM macro-F1",
                f"{llm_metrics['macro_f1']:.1%}",
                f"{comparison['macro_f1_delta']:+.1%} vs WordNet",
            )
            agreement_col, advantage_col = st.columns(2)
            agreement_col.metric(
                "Prediction agreement",
                f"{comparison['agreement_rate']:.1%}",
                f"{comparison['agreement_count']}/{comparison['count']} rows",
            )
            advantage_col.write(
                "**Correct on different examples**  \n"
                f"WordNet only: {comparison['wordnet_only_correct']}  \n"
                f"LLM only: {comparison['llm_only_correct']}"
            )

            st.markdown("**Per-class precision, recall, and F1**")
            per_class_frame = pd.DataFrame(comparison["per_class_comparison"])
            st.dataframe(
                per_class_frame.style.format(
                    {
                        "WordNet precision": "{:.1%}",
                        "LLM precision": "{:.1%}",
                        "WordNet recall": "{:.1%}",
                        "LLM recall": "{:.1%}",
                        "WordNet F1": "{:.1%}",
                        "LLM F1": "{:.1%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**Error analysis**")
            errors = comparison["error_rows"]
            st.write(
                f"{len(errors)} of {comparison['count']} examples contain at "
                "least one wrong prediction. Review each model's mistakes below."
            )
            error_columns = st.columns(3)
            error_columns[0].metric(
                "WordNet errors", comparison["wordnet_error_count"]
            )
            error_columns[1].metric(
                "LLM errors", comparison["llm_error_count"]
            )
            error_columns[2].metric(
                "Shared errors", comparison["shared_error_count"]
            )
            if errors:
                st.dataframe(
                    pd.DataFrame(errors),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success("Both classifiers predicted every benchmark example correctly.")

            confusion_wordnet, confusion_llm = st.columns(2)
            with confusion_wordnet:
                st.markdown("**WordNet confusion matrix**")
                st.dataframe(
                    pd.DataFrame(comparison["wordnet_confusion"]).set_index("actual"),
                    use_container_width=True,
                )
            with confusion_llm:
                st.markdown("**LLM confusion matrix**")
                st.dataframe(
                    pd.DataFrame(comparison["llm_confusion"]).set_index("actual"),
                    use_container_width=True,
                )

            st.markdown("**Conclusions**")
            for conclusion in comparison["conclusions"]:
                st.markdown(f"- {conclusion}")
    elif wordnet_rows is not None or llm_rows is not None:
        st.info(
            "Run both WordNet and LLM evaluation on this benchmark to unlock "
            "the side-by-side comparison, error analysis, confusion matrices, "
            "and conclusions."
        )

st.caption(f"Supported relation labels: {', '.join(RELATIONS)}.")
