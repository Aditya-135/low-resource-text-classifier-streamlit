import numpy as np
from collections import Counter


def shannon_entropy(labels):
    counts = np.array(list(Counter(labels).values()))
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


def analyze_dataset(texts, labels):
    total_samples = len(texts)
    class_counts = Counter(labels)
    imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
    entropy = shannon_entropy(labels)

    if total_samples < 100:
        resource_level = "extremely_low"
    elif total_samples < 1000:
        resource_level = "low"
    elif total_samples < 10000:
        resource_level = "moderate"
    else:
        resource_level = "high"

    imbalance_flag = imbalance_ratio > 3

    return {
        "total_samples": total_samples,
        "resource_level": resource_level,
        "imbalance_ratio": imbalance_ratio,
        "imbalance_flag": imbalance_flag,
        "entropy": entropy
    }