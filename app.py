import time
from pathlib import Path

import ollama
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ─────────────────────────────────────────
# 1. Page configuration
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Optimization Lab",
    page_icon="WOW!!!!",
    layout="wide",
)

# ── Custom CSS ────────────────────────────
st.markdown(
    """
    <style>
    /* metric cards */
    [data-testid="metric-container"] {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 12px 18px;
    }
    [data-testid="metric-container"] label {
        color: #cdd6f4 !important;
        font-size: 0.78rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
    }
    /* tab strip */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
    }
    /* category badge */
    .cat-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 1rem;
        font-weight: 700;
        color: #fff;
        background: linear-gradient(135deg,#7c3aed,#2563eb);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
# 2. Constants
# ─────────────────────────────────────────
MODEL = "llama3.2:3b"

CATEGORIES = [
    "Billing",
    "Technical Issue",
    "Delivery",
    "Upgrade",
    "Cancellation",
    "Complaint",
    "Other",
]

CATEGORY_COLORS = {
    "Billing":        "#f59e0b",
    "Technical Issue":"#3b82f6",
    "Delivery":       "#10b981",
    "Upgrade":        "#8b5cf6",
    "Cancellation":   "#ef4444",
    "Complaint":      "#f97316",
    "Other":          "#6b7280",
    "INVALID":        "#dc2626",
    "ERROR":          "#dc2626",
}

# ─────────────────────────────────────────
# 3. Prompt definitions
# ─────────────────────────────────────────
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
    "Optimized": (
        "Carefully identify the primary reason for the customer's "
        "message. Apply the category definitions consistently. "
        "Choose the most specific matching category, avoid adding "
        "explanations, and return only its exact name.\n\n"
        + RULES
    ),
}

PROMPT_DESCRIPTIONS = {
    "Zero-shot":  "Rules only — no examples, no persona.",
    "Few-shot":   "Rules + five labelled examples.",
    "Role-based": "Gives the model an expert persona before the rules.",
    "Structured": "Explicit step-by-step reasoning chain.",
    "Optimized":  "Concise combined instruction emphasising specificity.",
}

# ─────────────────────────────────────────
# 4. Classification helper
# ─────────────────────────────────────────
def classify(complaint: str, prompt: str):
    """Return (category, elapsed_seconds, raw_response, error_message)."""
    start = time.perf_counter()
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": "Complaint:\n" + str(complaint)},
            ],
            options={"temperature": 0},
        )
    except Exception as exc:
        return "ERROR", time.perf_counter() - start, "", str(exc)

    elapsed = time.perf_counter() - start
    raw = response["message"]["content"].strip()
    answer = raw.strip("\"'").strip()

    for cat in CATEGORIES:
        if answer.casefold() == cat.casefold():
            return cat, elapsed, raw, ""

    return "INVALID", elapsed, raw, ""


# ─────────────────────────────────────────
# 5. Page header
# ─────────────────────────────────────────
col_title, col_model = st.columns([3, 1])
with col_title:
    st.title("🧪 Prompt Optimization Lab")
    st.caption("Customer Complaint Classification · Prompt Engineering with Llama 3.2")
with col_model:
    st.info(f"**Model:** `{MODEL}`", icon="🤖")

st.divider()

# ─────────────────────────────────────────
# 6. Tabs
# ─────────────────────────────────────────
live_tab, experiment_tab = st.tabs([
    "💬  Live Classification",
    "📊  Compare Prompt Techniques",
])

# ═════════════════════════════════════════
# TAB 1 — Live Classification
# ═════════════════════════════════════════
with live_tab:

    st.subheader("Classify a single complaint in real time")
    st.write("Pick a prompting technique, type a complaint, and see the model's decision instantly.")

    left, right = st.columns([1, 1], gap="large")

    with left:
        selected_technique = st.selectbox(
            "Prompting technique",
            list(PROMPTS.keys()),
            help="Each technique structures the system prompt differently.",
        )
        st.caption(f"_{PROMPT_DESCRIPTIONS[selected_technique]}_")

        complaint_input = st.text_area(
            "Customer complaint",
            placeholder="Example: I was charged twice this month.",
            height=130,
        )

        show_prompt = st.toggle("Show full system prompt", value=False)
        if show_prompt:
            with st.expander("System prompt sent to the model", expanded=True):
                st.code(PROMPTS[selected_technique], language="text")

        classify_btn = st.button("⚡ Classify", type="primary", use_container_width=True)

    with right:
        st.markdown("#### Result")
        result_placeholder = st.empty()
        result_placeholder.info("Enter a complaint on the left and click **Classify**.")

        if classify_btn:
            if not complaint_input.strip():
                result_placeholder.warning("⚠️ Please enter a complaint before classifying.")
            else:
                with st.spinner("Asking the model…"):
                    pred, secs, raw_resp, err = classify(
                        complaint_input, PROMPTS[selected_technique]
                    )

                if err:
                    result_placeholder.error(
                        f"**Ollama error:** {err}\n\n"
                        "Make sure Ollama is running and the model is installed."
                    )
                elif pred == "INVALID":
                    result_placeholder.warning(
                        f"**Model returned an unexpected response.**\n\nRaw: `{raw_resp}`"
                    )
                else:
                    color = CATEGORY_COLORS.get(pred, "#6b7280")
                    result_placeholder.markdown(
                        f"""
                        <div style='padding:24px;border-radius:12px;
                                    background:{color}22;border:2px solid {color};'>
                          <p style='margin:0;font-size:0.85rem;color:#888;'>Predicted category</p>
                          <p style='margin:4px 0 12px;font-size:2rem;font-weight:800;color:{color};'>{pred}</p>
                          <p style='margin:0;font-size:0.8rem;color:#888;'>
                            ⏱ {secs:.2f} s &nbsp;|&nbsp; technique: <strong>{selected_technique}</strong>
                          </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ── Quick multi-category explorer ──────────────────────
    st.divider()
    with st.expander("🔍 Try the same complaint across all techniques at once"):
        batch_complaint = st.text_area(
            "Complaint to compare",
            placeholder="Enter a complaint to run through every technique.",
            height=80,
            key="batch_single",
        )
        if st.button("Run across all techniques", key="run_all_single"):
            if not batch_complaint.strip():
                st.warning("Please enter a complaint.")
            else:
                cols = st.columns(len(PROMPTS))
                for col, (tech, prompt) in zip(cols, PROMPTS.items()):
                    with col:
                        with st.spinner(tech):
                            p, s, r, e = classify(batch_complaint, prompt)
                        color = CATEGORY_COLORS.get(p, "#6b7280")
                        st.markdown(
                            f"""
                            <div style='text-align:center;padding:14px;border-radius:10px;
                                        background:{color}22;border:1.5px solid {color};'>
                              <p style='margin:0;font-size:0.72rem;color:#888;'>{tech}</p>
                              <p style='margin:4px 0 0;font-size:1.1rem;font-weight:700;
                                        color:{color};'>{p}</p>
                              <p style='margin:0;font-size:0.7rem;color:#888;'>{s:.2f}s</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


# ═════════════════════════════════════════
# TAB 2 — Experiment
# ═════════════════════════════════════════
with experiment_tab:

    st.subheader("Benchmark all prompt techniques on the labelled dataset")

    csv_path = Path(__file__).parent / "complaints_test.csv"

    # ── Dataset validation ─────────────────
    if not csv_path.exists():
        st.error(f"Dataset not found: `{csv_path}`")
        st.stop()

    if csv_path.stat().st_size == 0:
        st.error("The CSV file is empty.")
        st.stop()

    try:
        data = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()

    required_columns = {"complaint", "true_category"}
    if not required_columns.issubset(data.columns):
        st.error("CSV must contain `complaint` and `true_category` columns.")
        st.stop()

    data = data.dropna(subset=["complaint", "true_category"]).copy()
    data["complaint"]      = data["complaint"].astype(str).str.strip()
    data["true_category"]  = data["true_category"].astype(str).str.strip()
    data = data[(data["complaint"] != "") & (data["true_category"] != "")].copy()

    if data.empty:
        st.error("No valid rows found in the CSV.")
        st.stop()

    invalid_labels = sorted(
        set(data["true_category"].str.casefold())
        - {c.casefold() for c in CATEGORIES}
    )
    if invalid_labels:
        st.error("Unknown category labels in CSV: " + ", ".join(invalid_labels))
        st.stop()

    label_lookup = {c.casefold(): c for c in CATEGORIES}
    data["true_category"] = data["true_category"].str.casefold().map(label_lookup)

    # ── Dataset preview ────────────────────
    with st.expander(f"📂 Dataset preview — {len(data)} complaints", expanded=False):
        st.dataframe(data, use_container_width=True, height=250)

    st.info(
        "Each of the five prompt techniques is evaluated on every complaint. "
        "Use a held-out set for final reporting — do not tune prompts on these labels.",
        icon="ℹ️",
    )

    # ── Technique selector ─────────────────
    techniques_to_run = st.multiselect(
        "Select techniques to benchmark",
        list(PROMPTS.keys()),
        default=list(PROMPTS.keys()),
    )

    run_btn = st.button(
        "🚀 Run Benchmark",
        type="primary",
        disabled=len(techniques_to_run) == 0,
    )

    if run_btn:

        total_calls = len(techniques_to_run) * len(data)
        all_results = []
        completed = 0

        progress_bar  = st.progress(0, text="Starting…")
        status_text   = st.empty()
        live_table_ph = st.empty()

        for technique in techniques_to_run:
            prompt = PROMPTS[technique]
            for _, row in data.iterrows():
                try:
                    pred, secs, raw, err = classify(row["complaint"], prompt)
                except Exception as exc:
                    pred, secs, raw, err = "ERROR", 0.0, "", str(exc)

                all_results.append({
                    "Technique":        technique,
                    "Complaint":        row["complaint"],
                    "True":             row["true_category"],
                    "Predicted":        pred,
                    "Correct":          "✅" if pred == row["true_category"] else "❌",
                    "Seconds":          round(secs, 2),
                    "Raw response":     raw,
                    "Error":            err,
                })

                completed += 1
                pct = completed / total_calls
                progress_bar.progress(
                    pct,
                    text=f"Running **{technique}** — {completed}/{total_calls}",
                )
                status_text.caption(f"Last: _{row['complaint'][:60]}…_  →  **{pred}**")

                # Live preview of results so far
                live_table_ph.dataframe(
                    pd.DataFrame(all_results)[
                        ["Technique", "Complaint", "True", "Predicted", "Correct", "Seconds"]
                    ],
                    use_container_width=True,
                    height=200,
                )

        progress_bar.empty()
        status_text.empty()
        live_table_ph.empty()

        results_df   = pd.DataFrame(all_results)
        completed_df = results_df[results_df["Error"] == ""].copy()

        # ── Metrics ───────────────────────────
        metric_rows = []
        for technique in techniques_to_run:
            subset = completed_df[completed_df["Technique"] == technique]
            n_err  = int((results_df[results_df["Technique"] == technique]["Error"] != "").sum())

            if subset.empty:
                metric_rows.append({
                    "Technique":      technique,
                    "Accuracy":       float("nan"),
                    "Macro F1":       float("nan"),
                    "Avg time (s)":   float("nan"),
                    "Invalid":        0,
                    "Errors":         n_err,
                })
                continue

            y_true = subset["True"]
            y_pred = subset["Predicted"]

            metric_rows.append({
                "Technique":    technique,
                "Accuracy":     accuracy_score(y_true, y_pred),
                "Macro F1":     f1_score(y_true, y_pred, labels=CATEGORIES,
                                         average="macro", zero_division=0),
                "Avg time (s)": subset["Seconds"].mean(),
                "Invalid":      int((y_pred == "INVALID").sum()),
                "Errors":       n_err,
            })

        metrics_df = pd.DataFrame(metric_rows).sort_values(
            "Macro F1", ascending=False
        )

        # ── Summary metric cards ──────────────
        st.divider()
        st.subheader("1 · Overall performance")

        best_row = metrics_df.dropna(subset=["Macro F1"]).iloc[0] if not metrics_df.dropna(subset=["Macro F1"]).empty else None
        card_cols = st.columns(len(techniques_to_run))

        for col, (_, row) in zip(card_cols, metrics_df.iterrows()):
            is_best = best_row is not None and row["Technique"] == best_row["Technique"]
            with col:
                acc_str = f"{row['Accuracy']:.1%}" if pd.notna(row["Accuracy"]) else "—"
                f1_str  = f"{row['Macro F1']:.3f}"  if pd.notna(row["Macro F1"])  else "—"
                border  = "2px solid #7c3aed" if is_best else "1px solid #313244"
                crown   = " 👑" if is_best else ""
                st.markdown(
                    f"""
                    <div style='padding:14px;border-radius:10px;background:#1e1e2e;
                                border:{border};text-align:center;'>
                      <p style='margin:0;font-size:0.75rem;color:#888;'>{row["Technique"]}{crown}</p>
                      <p style='margin:4px 0 2px;font-size:1.6rem;font-weight:800;color:#cdd6f4;'>{acc_str}</p>
                      <p style='margin:0;font-size:0.75rem;color:#888;'>acc &nbsp;|&nbsp; F1 {f1_str}</p>
                      <p style='margin:4px 0 0;font-size:0.7rem;color:#555;'>⏱ {row["Avg time (s)"]:.2f}s avg</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Metrics table ─────────────────────
        st.dataframe(
            metrics_df.style.format({
                "Accuracy":     "{:.1%}",
                "Macro F1":     "{:.3f}",
                "Avg time (s)": "{:.2f}",
            }).highlight_max(subset=["Accuracy", "Macro F1"], color="#1a3a1a")
              .highlight_min(subset=["Avg time (s)"],           color="#1a2e3a"),
            use_container_width=True,
        )

        # ── Charts ────────────────────────────
        st.subheader("2 · Visual comparison")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**Accuracy**")
            st.bar_chart(
                metrics_df.set_index("Technique")[["Accuracy"]],
                height=260,
                color="#7c3aed",
            )
        with chart_col2:
            st.markdown("**Macro F1**")
            st.bar_chart(
                metrics_df.set_index("Technique")[["Macro F1"]],
                height=260,
                color="#2563eb",
            )

        # ── Best candidate callout ────────────
        if best_row is not None:
            st.success(
                f"🏆 **{best_row['Technique']}** achieved the highest Macro-F1 "
                f"({best_row['Macro F1']:.3f}) in this run.",
                icon="✅",
            )

        # ── Confusion matrix ──────────────────
        st.subheader("3 · Confusion matrix")

        chosen_tech = st.selectbox(
            "Choose technique",
            techniques_to_run,
            key="matrix_technique",
        )

        subset_cm = completed_df[completed_df["Technique"] == chosen_tech]

        if not subset_cm.empty:
            cm = confusion_matrix(
                subset_cm["True"],
                subset_cm["Predicted"],
                labels=CATEGORIES,
            )
            cm_df = pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES)

            st.caption("Rows = true category · Columns = predicted category")
            st.dataframe(
                cm_df.style.background_gradient(cmap="Blues"),
                use_container_width=True,
            )

            # Per-category scores
            st.subheader("4 · Per-category scores")
            report = classification_report(
                subset_cm["True"],
                subset_cm["Predicted"],
                labels=CATEGORIES,
                output_dict=True,
                zero_division=0,
            )
            report_df = pd.DataFrame(report).T.drop(
                columns=["support"], errors="ignore"
            )
            st.dataframe(
                report_df.style.format("{:.3f}").background_gradient(cmap="Greens"),
                use_container_width=True,
            )
        else:
            st.info("No completed results for this technique.")

        # ── Detailed predictions ──────────────
        st.subheader("5 · Inspect individual predictions")

        filter_tech = st.multiselect(
            "Filter by technique",
            techniques_to_run,
            default=techniques_to_run,
            key="filter_detail",
        )
        filter_correct = st.radio(
            "Show",
            ["All", "Correct only ✅", "Incorrect only ❌"],
            horizontal=True,
            key="filter_correct",
        )

        view_df = results_df[results_df["Technique"].isin(filter_tech)].copy()
        if filter_correct == "Correct only ✅":
            view_df = view_df[view_df["Correct"] == "✅"]
        elif filter_correct == "Incorrect only ❌":
            view_df = view_df[view_df["Correct"] == "❌"]

        st.dataframe(
            view_df[["Technique", "Complaint", "True", "Predicted", "Correct", "Seconds"]],
            use_container_width=True,
            height=320,
        )

        # ── Downloads ─────────────────────────
        st.subheader("6 · Download results")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Download all predictions (CSV)",
                results_df.to_csv(index=False),
                file_name="prompt_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇️ Download performance metrics (CSV)",
                metrics_df.to_csv(index=False),
                file_name="prompt_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
