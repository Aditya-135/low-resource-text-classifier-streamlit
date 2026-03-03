import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from data_ingestion import encode_labels, split_data
from data_analyzer import analyze_dataset
from augmentation_controller import adaptive_augment
from trainer import train_classical, train_xlmr


def run_dashboard():
    st.title("Low-Resource Text Classification System")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "datasets")

    dataset_option = st.radio(
        "Select Dataset Source:",
        ("Upload CSV", "Use Built-in AG News", "Use Built-in Kaggle News")
    )

    df = None

    if dataset_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)

    elif dataset_option == "Use Built-in AG News":
        df = pd.read_csv(os.path.join(DATASET_DIR, "ag_news.csv"))

    elif dataset_option == "Use Built-in Kaggle News":
        df = pd.read_csv(os.path.join(DATASET_DIR, "kaggle_news.csv"))

    if df is not None:

        # Clean dataset (fix Kaggle NaN issue)
        df = df.dropna(subset=["text", "label"])
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(str)

        if "text" not in df.columns or "label" not in df.columns:
            st.error("Dataset must contain 'text' and 'label' columns.")
            return

        # Encode labels BEFORE split
        df, le = encode_labels(df)

        num_labels = len(le.classes_)  # 🔥 Correct fix

        X_train, X_test, y_train, y_test = split_data(df)

        report = analyze_dataset(X_train, y_train)

        st.subheader("Dataset Analysis")
        st.write(report)

        X_train_aug, y_train_aug = adaptive_augment(X_train, y_train, report)

        st.subheader("Model Training & Evaluation")

        # Classical Model
        with st.spinner("Training Classical ML Model..."):
            acc_classical, f1_classical, cm_classical = train_classical(
                X_train_aug, y_train_aug, X_test, y_test
            )

        st.success("Classical Model Complete")

        # Transformer Model
        st.write("Transformer Training Progress:")
        progress_bar = st.progress(0)

        acc_xlmr, f1_xlmr, cm_xlmr = train_xlmr(
            X_train_aug,
            y_train_aug,
            X_test,
            y_test,
            num_labels,
            dataset_option,  # pass dataset name for model isolation
            progress_bar
        )

        progress_bar.progress(100)
        st.success("Transformer Ready")

        # Results Table
        results_df = pd.DataFrame({
            "Model": ["Classical ML", "XLM-R"],
            "Accuracy": [acc_classical, acc_xlmr],
            "F1 Score": [f1_classical, f1_xlmr]
        })

        st.subheader("Performance Metrics")
        st.dataframe(results_df)

        # Download CSV
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results CSV",
            csv,
            "model_results.csv",
            "text/csv"
        )

        # Graph
        st.subheader("Performance Comparison")

        fig, ax = plt.subplots()
        width = 0.35
        x = np.arange(len(results_df["Model"]))

        ax.bar(x - width/2, results_df["Accuracy"], width, label="Accuracy")
        ax.bar(x + width/2, results_df["F1 Score"], width, label="F1 Score")

        ax.set_xticks(x)
        ax.set_xticklabels(results_df["Model"])
        ax.legend()

        st.pyplot(fig)

        # Confusion Matrix
        st.subheader("Confusion Matrix - Transformer")

        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm_xlmr, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")

        st.pyplot(fig_cm)