import random
import nltk
import os
from nltk.corpus import wordnet

# Fix for Streamlit cloud nltk path
NLTK_DATA_DIR = "/home/appuser/nltk_data"

if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR)

nltk.data.path.append(NLTK_DATA_DIR)

# Ensure datasets exist
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", download_dir=NLTK_DATA_DIR)
    nltk.download("omw-1.4", download_dir=NLTK_DATA_DIR)


def synonym_replacement(text):

    words = text.split()

    if not words:
        return text

    word = random.choice(words)

    synonyms = wordnet.synsets(word)

    if synonyms:
        lemma = synonyms[0].lemmas()[0].name()

        if lemma.lower() != word.lower():
            return text.replace(word, lemma)

    return text


def adaptive_augment(texts, labels, report):

    augmented_texts = list(texts)
    augmented_labels = list(labels)

    if report["resource_level"] in ["extremely_low", "low"]:

        for text, label in zip(texts, labels):

            new_text = synonym_replacement(text)

            augmented_texts.append(new_text)
            augmented_labels.append(label)

    return augmented_texts, augmented_labels