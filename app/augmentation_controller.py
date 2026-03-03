import random
from nltk.corpus import wordnet


def synonym_replacement(text):
    words = text.split()
    if not words:
        return text

    word = random.choice(words)
    synonyms = wordnet.synsets(word)

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