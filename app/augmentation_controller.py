import random
import nltk
from nltk.corpus import wordnet

# Ensure required NLTK resources exist
def ensure_nltk():
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")


# Run check when module loads
ensure_nltk()


def synonym_replacement(text):
    words = text.split()

    if not words:
        return text

    word = random.choice(words)

    try:
        synonyms = wordnet.synsets(word)
    except LookupError:
        return text

    if synonyms:
        lemma = synonyms[0].lemmas()[0].name()
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