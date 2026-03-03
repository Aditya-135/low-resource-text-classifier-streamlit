import numpy as np

def predict(text, model, vectorizer):
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)
    pred = np.argmax(probs)
    confidence = np.max(probs)
    return pred, confidence