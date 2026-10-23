
import time
from pathlib import Path

import ollama
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# 1. Page configuration
# -----------------------------
st.set_page_config(
    page_title="Prompt Optimization Lab",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Complaint Classification")
st.write("Prompt Engineering and Optimization using Llama 3.2")

MODEL = "llama3.2:3b"

CATEGORIES = [
    "Billing",
    "Technical Issue",
    "Delivery",
    "Upgrade",
    "Cancellation",
    "Complaint",
    "Other"
]

# -----------------------------
# 2. Define the common task
# -----------------------------
RULES = """
Classify the customer complaint into exactly one category.

Billing: payment, invoice, or unexpected charge problems.
Technical Issue: software bugs, errors, or service failures.
Delivery: shipping, tracking, or late package problems.
Upgrade: requests to upgrade a plan or service.
Cancellation: requests to cancel a service or subscription.
Complaint: dissatisfaction not covered by another category.
Other: questions or requests outside these categories.

Choose the category that best matches the main issue.
Return only one exact category name from the allowed list.
"""

EXAMPLES = """
Complaint: I was charged twice for my subscription.
Category: Billing

Complaint: My application crashes when I open it.
Category: Technical Issue

Complaint: My package has not arrived.
Category: Delivery

Complaint: I want to move to a premium plan.
Category: Upgrade

Complaint: Please cancel my subscription.
Category: Cancellation
"""

# -----------------------------
# 3. Define prompt techniques
# -----------------------------
PROMPTS = {
    "Zero-shot": RULES,

    "Few-shot": RULES + "\nExamples:\n" + EXAMPLES,

    "Role-based": (
        "You are an experienced customer support classification "
        "specialist.\n" + RULES
    ),

    "Structured": (
        "Follow these steps:\n"
        "1. Identify the customer's main issue.\n"
        "2. Match it to the category definitions.\n"
        "3. Select exactly one allowed category.\n"
        "4. Return only the category name.\n\n"
        + RULES
    ),

    "Optimized candidate": (
        "Carefully identify the primary reason for the customer's "
        "message. Apply the category definitions consistently. "
        "Choose the most specific matching category, avoid adding "
        "explanations, and return only its exact name.\n\n"
        + RULES
    )
}


# -----------------------------
# 4. Call Ollama
# -----------------------------
def classify(complaint, prompt):
    start_time = time.perf_counter()

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": "Complaint:\n" + str(complaint)
            }
        ],
        options={"temperature": 0}
    )

    elapsed = time.perf_counter() - start_time

    raw_answer = response["message"]["content"].strip()
    answer = raw_answer.strip("\"'").strip()

    # Accept only an exact category name.
    for category in CATEGORIES:
        if answer.casefold() == category.casefold():
            return category, elapsed, raw_answer, ""

    return "INVALID", elapsed, raw_answer, ""


# -----------------------------
# 5. Create dashboard tabs
# -----------------------------
live_tab, experiment_tab = st.tabs([
    "Live Classification",
    "Compare Prompt Techniques"
])


# -----------------------------
# 6. Live classification
# -----------------------------
with live_tab:
    st.subheader("Classify one complaint")

    selected_prompt = st.selectbox(
        "Select a prompting technique",
        list(PROMPTS.keys())
    )

    complaint = st.text_area(
        "Enter a customer complaint:",
        placeholder="Example: I was charged twice this month."
    )

    if st.button("Classify Complaint"):
        if not complaint.strip():
            st.warning("Please enter a complaint.")
        else:
            try:
                with st.spinner("Classifying..."):
                    prediction, seconds, raw, error = classify(
                        complaint,
                        PROMPTS[selected_prompt]
                    )

                st.write("Prompt technique:", selected_prompt)
                st.write(f"Response time: {seconds:.2f} seconds")

                if prediction == "INVALID":
                    st.warning("The model returned an invalid category.")
                    st.write("Raw response:", raw)
                else:
                    st.success(
                        f"Predicted category: {prediction}"
                    )

            except Exception as e:
                st.error(
                    "Ollama request failed. Check that Ollama is "
                    "running and the model is installed."
                )
                st.exception(e)


# -----------------------------
# 7. Prompt comparison experiment
# -----------------------------
with experiment_tab:
    st.subheader("Experimental Performance Comparison")

    st.write(
        "Each prompt is evaluated on the same labeled complaints "
        "using the same local model."
    )

    csv_path = Path(__file__).parent / "complaints_test.csv"

    if not csv_path.exists():
        st.error(
            f"Dataset not found: {csv_path}. "
            "Put complaints_test.csv beside experiment.py."
        )

    elif csv_path.stat().st_size == 0:
        st.error("The CSV file is empty.")

    else:
        try:
            data = pd.read_csv(csv_path)
        except (pd.errors.EmptyDataError, UnicodeDecodeError):
            data = None
            st.error("Could not read the CSV file.")

        if data is not None:
            required_columns = {"complaint", "true_category"}

            if not required_columns.issubset(data.columns):
                st.error(
                    "CSV must contain complaint and true_category columns."
                )

            else:
                data = data.dropna(
                    subset=["complaint", "true_category"]
                ).copy()

                data["complaint"] = (
                    data["complaint"].astype(str).str.strip()
                )
                data["true_category"] = (
                    data["true_category"].astype(str).str.strip()
                )

                data = data[
                    (data["complaint"] != "") &
                    (data["true_category"] != "")
                ].copy()

                invalid_labels = sorted(
                    set(data["true_category"].str.casefold())
                    - {c.casefold() for c in CATEGORIES}
                )

                if data.empty:
                    st.error("No valid complaints were found.")

                elif invalid_labels:
                    st.error(
                        "Fix these category labels in the CSV: "
                        + ", ".join(invalid_labels)
                    )

                else:
                    # Standardize label capitalization.
                    label_lookup = {
                        c.casefold(): c for c in CATEGORIES
                    }
                    data["true_category"] = (
                        data["true_category"]
                        .str.casefold()
                        .map(label_lookup)
                    )

                    st.write(f"Dataset size: {len(data)} complaints")
                    st.dataframe(data.head(10), use_container_width=True)

                    st.warning(
                        "Use a held-out test set for final reporting. "
                        "Do not use its labels to design or tune prompts."
                    )

                    if st.button("Run All Prompt Experiments"):
                        all_results = []
                        progress = st.progress(0)

                        total = len(PROMPTS) * len(data)
                        completed = 0

                        for technique, prompt in PROMPTS.items():
                            for _, row in data.iterrows():
                                try:
                                    prediction, seconds, raw, error = (
                                        classify(
                                            row["complaint"],
                                            prompt
                                        )
                                    )
                                except Exception as e:
                                    prediction = "ERROR"
                                    seconds = 0.0
                                    raw = str(e)
                                    error = str(e)

                                all_results.append({
                                    "technique": technique,
                                    "complaint": row["complaint"],
                                    "true_category": row["true_category"],
                                    "predicted_category": prediction,
                                    "seconds": seconds,
                                    "raw_response": raw,
                                    "error": error
                                })

                                completed += 1
                                progress.progress(completed / total)

                        results_df = pd.DataFrame(all_results)

                        # Only completed requests are used for
                        # the main classification metrics.
                        completed_df = results_df[
                            results_df["error"] == ""
                        ].copy()

                        metric_rows = []

                        for technique in PROMPTS:
                            subset = completed_df[
                                completed_df["technique"] == technique
                            ]

                            if subset.empty:
                                metric_rows.append({
                                    "Technique": technique,
                                    "Accuracy": float("nan"),
                                    "Macro F1": float("nan"),
                                    "Avg seconds": float("nan"),
                                    "Invalid outputs": 0,
                                    "Request errors": int(
                                        (
                                            results_df[
                                                results_df["technique"]
                                                == technique
                                            ]["error"] != ""
                                        ).sum()
                                    )
                                })
                                continue

                            y_true = subset["true_category"]
                            y_pred = subset["predicted_category"]

                            metric_rows.append({
                                "Technique": technique,
                                "Accuracy": accuracy_score(
                                    y_true, y_pred
                                ),
                                "Macro F1": f1_score(
                                    y_true,
                                    y_pred,
                                    labels=CATEGORIES,
                                    average="macro",
                                    zero_division=0
                                ),
                                "Avg seconds": subset["seconds"].mean(),
                                "Invalid outputs": int(
                                    (y_pred == "INVALID").sum()
                                ),
                                "Request errors": int(
                                    (
                                        results_df[
                                            results_df["technique"]
                                            == technique
                                        ]["error"] != ""
                                    ).sum()
                                )
                            })

                        metrics_df = pd.DataFrame(metric_rows)

                        st.subheader("1. Performance summary")

                        st.dataframe(
                            metrics_df.style.format({
                                "Accuracy": "{:.1%}",
                                "Macro F1": "{:.3f}",
                                "Avg seconds": "{:.2f}"
                            }),
                            use_container_width=True
                        )

                        st.subheader("2. Accuracy comparison")

                        chart_data = metrics_df.set_index("Technique")[
                            ["Accuracy"]
                        ]
                        st.bar_chart(chart_data)

                        st.subheader("3. Macro-F1 comparison")

                        f1_chart = metrics_df.set_index("Technique")[
                            ["Macro F1"]
                        ]
                        st.bar_chart(f1_chart)

                        # Choose by Macro-F1, then accuracy.
                        valid_metrics = metrics_df.dropna(
                            subset=["Macro F1", "Accuracy"]
                        )

                        if not valid_metrics.empty:
                            best = valid_metrics.sort_values(
                                ["Macro F1", "Accuracy"],
                                ascending=False
                            ).iloc[0]

                            st.subheader("4. Best measured candidate")
                            st.write(
                                f"**{best['Technique']}** has the "
                                f"highest Macro-F1 among these runs: "
                                f"{best['Macro F1']:.3f}."
                            )

                            st.caption(
                                "This is the best measured prompt on "
                                "this dataset, not proof that it will "
                                "perform best on unseen complaints."
                            )

                        st.subheader("5. Confusion matrix")

                        chosen = st.selectbox(
                            "Choose technique to inspect",
                            list(PROMPTS.keys()),
                            key="matrix_technique"
                        )

                        selected_results = completed_df[
                            completed_df["technique"] == chosen
                        ]

                        if not selected_results.empty:
                            cm = confusion_matrix(
                                selected_results["true_category"],
                                selected_results["predicted_category"],
                                labels=CATEGORIES
                            )

                            cm_df = pd.DataFrame(
                                cm,
                                index=CATEGORIES,
                                columns=CATEGORIES
                            )

                            st.write(
                                "Rows are the correct categories; "
                                "columns are predicted categories."
                            )
                            st.dataframe(cm_df, use_container_width=True)

                            report = classification_report(
                                selected_results["true_category"],
                                selected_results["predicted_category"],
                                labels=CATEGORIES,
                                output_dict=True,
                                zero_division=0
                            )

                            st.subheader("6. Per-category scores")
                            st.dataframe(
                                pd.DataFrame(report).T,
                                use_container_width=True
                            )

                        st.subheader("7. Inspect individual predictions")
                        st.dataframe(results_df, use_container_width=True)

                        st.download_button(
                            "Download all predictions",
                            results_df.to_csv(index=False),
                            file_name="prompt_predictions.csv",
                            mime="text/csv"
                        )

                        st.download_button(
                            "Download performance metrics",
                            metrics_df.to_csv(index=False),
                            file_name="prompt_metrics.csv",
                            mime="text/csv"
                        )
