import os
import torch
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
)

from datasets import Dataset

# 🔥 CUDA SAFE DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Classical ML
# =========================
def train_classical(X_train, y_train, X_test, y_test):

    vectorizer = TfidfVectorizer(max_features=5000)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds)

    return acc, f1, cm


# =========================
# Streamlit Progress
# =========================
class StreamlitProgressCallback(TrainerCallback):
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar

    def on_step_end(self, args, state, control, **kwargs):
        if state.max_steps > 0:
            percent = int((state.global_step / state.max_steps) * 100)
            self.progress_bar.progress(min(percent, 100))


# =========================
# Transformer Model
# =========================
def train_xlmr(X_train, y_train, X_test, y_test,
               num_labels, dataset_name, progress_bar):

    # 🔥 Dataset-specific model saving
    safe_name = dataset_name.replace(" ", "_")
    model_dir = f"./saved_models/{safe_name}_{num_labels}_labels"

    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model only if label size matches
    if os.path.exists(model_dir):
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        progress_bar.progress(100)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(device)

    train_dataset = Dataset.from_dict({
        "text": list(X_train),
        "label": list(y_train)
    })

    test_dataset = Dataset.from_dict({
        "text": list(X_test),
        "label": list(y_test)
    })

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_steps=10,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[StreamlitProgressCallback(progress_bar)]
    )

    if not os.path.exists(model_dir):
        trainer.train()
        trainer.save_model(model_dir)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds)

    return acc, f1, cm