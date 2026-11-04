"""Download the NLTK corpora and tagger needed by the application."""

import nltk


def main() -> None:
    packages = ("punkt_tab", "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4")
    for package in packages:
        if not nltk.download(package, raise_on_error=True):
            raise RuntimeError(f"NLTK could not download required package: {package}")
    print("NLTK tokenizer, tagger, and WordNet data are ready.")


if __name__ == "__main__":
    main()
