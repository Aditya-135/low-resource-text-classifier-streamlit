import random
import nltk
from nltk.corpus import wordnet

# Ensure WordNet corpus is available in the environment
def ensure_wordnet():
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

# Call once when the module loads
ensure_wordnet()


def synonym_replacement(text):
    words = text.split()

    if not words:
        return text

    word = random.choice(words)

    try:
        synonyms = wordnet.synsets(word)
    except LookupError:
        ensure_wordnet()
        synonyms = wordnet.synsets(word)

    if synonyms:
        lemma = synonyms[0].lemmas()[0].name()

        # avoid replacing with same word
        if lemma.lower() != word.lower():
            return text.replace(word, lemma)

    return text


def adaptive_augment(texts, labels, report):

    augmented_texts = list(texts)
    augmented_labels = list(labels)

    # Only augment if dataset is low resource
    if report["resource_level"] in ["extremely_low", "low"]:

        for text, label in zip(texts, labels):

            new_text = synonym_replacement(text)

            augmented_texts.append(new_text)
            augmented_labels.append(label)

    return augmented_texts, augmented_labels