!pip install numpy pandas matplotlib seaborn scikit-learn
!pip install gradio -q

import gradio as gr

def predict_credit(
    income,
    age,
    credit_amount,
    duration,
    dti,
    missed_payments
):

    score = 500

    score += min(income / 200000 * 120, 120)

    if 35 <= age <= 55:
        score += 30
    elif age >= 25:
        score += 15

    score -= dti * 2.5
    score -= missed_payments * 35

    ratio = credit_amount / max(income, 1)

    if ratio < 0.2:
        score += 20
    elif ratio > 0.5:
        score -= 30

    score = max(300, min(850, round(score)))

    if score >= 680:
        verdict = "✅ APPROVED"
    elif score >= 540:
        verdict = "⚠️ MANUAL REVIEW"
    else:
        verdict = "❌ DENIED"

    return f"""
    Credit Score: {score}

    Decision: {verdict}
    """

iface = gr.Interface(
    fn=predict_credit,

    inputs=[
        gr.Slider(10000, 200000, label="Income"),
        gr.Slider(18, 70, label="Age"),
        gr.Slider(1000, 50000, label="Credit Amount"),
        gr.Slider(6, 72, label="Duration (Months)"),
        gr.Slider(0, 100, label="DTI %"),
        gr.Slider(0, 10, step=1, label="Missed Payments")
    ],

    outputs="text",

    title="🚀 CreditIQ Live Credit Scoring Model",

    description="Interactive AI-powered credit approval simulator"
)

iface.launch(share=True)
# ============================================================
#  CreditIQ - Credit Scoring Model (Python)
#  Converted from HTML/JS simulation to real ML implementation
#  Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score
)

# ─────────────────────────────────────────────
#  STYLE CONFIG
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#050a0f",
    "axes.facecolor":    "#0d1620",
    "axes.edgecolor":    "#1e3448",
    "axes.labelcolor":   "#e8f4f8",
    "xtick.color":       "#5a7a8a",
    "ytick.color":       "#5a7a8a",
    "text.color":        "#e8f4f8",
    "grid.color":        "#1e3448",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   "#00d4ff",
})

COLORS = {
    "accent":  "#00d4ff",
    "green":   "#00ff88",
    "orange":  "#ff6b35",
    "warn":    "#ffb830",
    "danger":  "#ff3b5c",
    "purple":  "#c084fc",
    "muted":   "#5a7a8a",
    "text":    "#e8f4f8",
}

MODEL_COLORS = {
    "Logistic Regression":  "#00d4ff",
    "Decision Tree":        "#ff6b35",
    "Random Forest":        "#00ff88",
    "Gradient Boosting":    "#ff3b5c",
}

# ─────────────────────────────────────────────
#  1. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────
def load_and_preprocess():
    print("=" * 60)
    print("  CreditIQ — ML Engine v2.4")
    print("=" * 60)
    print("\n[LOADING]  Fetching German Credit dataset...")

    # Load German Credit dataset from OpenML
    data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()

    print(f"[OK]       Loaded {len(df)} records, {df.shape[1]} raw features")

    # Encode categorical columns
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include="category").columns.tolist()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Target: 'class' → 1 = good credit, 0 = default
    df["target"] = (df["class"] == 1).astype(int)
    df.drop(columns=["class"], inplace=True)

    # ── Feature Engineering ──
    # DTI proxy: credit_amount / (duration * checking_status)
    df["dti_ratio"]       = df["credit_amount"] / (df["duration"] + 1)
    df["credit_per_month"]= df["credit_amount"] / df["duration"]
    df["age_credit_ratio"]= df["age"] / (df["credit_amount"] / 1000 + 1)

    print("[OK]       Feature engineering complete (3 derived features added)")
    print(f"[OK]       Missing values: {df.isnull().sum().sum()}")

    # Select top features (mirroring HTML version)
    feature_cols = [
        "credit_amount", "duration", "dti_ratio", "checking_status",
        "age", "credit_history", "employment", "savings_status",
        "installment_commitment", "existing_credits",
        "credit_per_month", "age_credit_ratio",
        "property_magnitude", "purpose", "housing"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["target"]

    print(f"[OK]       Final feature set: {len(feature_cols)} features")
    return X, y, feature_cols


# ─────────────────────────────────────────────
#  2. TRAIN / TEST SPLIT + SCALING
# ─────────────────────────────────────────────
def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[OK]       Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ─────────────────────────────────────────────
#  3. TRAIN ALL MODELS
# ─────────────────────────────────────────────
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    trained = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[TRAINING] Fitting models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        trained[name] = {"model": model, "cv": cv_scores}
        print(f"           {name:<25} CV Acc: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return trained


# ─────────────────────────────────────────────
#  4. EVALUATE MODELS
# ─────────────────────────────────────────────
def evaluate_models(trained, X_test, y_test):
    print("\n[EVAL]     Evaluating on test set...")
    results = {}
    for name, info in trained.items():
        model = info["model"]
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model":     model,
            "cv":        info["cv"],
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "accuracy":  accuracy_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall":    recall_score(y_test, y_pred, average="weighted"),
            "auc":       roc_auc_score(y_test, y_prob),
        }
        print(f"           {name:<25} AUC: {results[name]['auc']:.3f}  "
              f"Acc: {results[name]['accuracy']:.3f}  "
              f"F1: {results[name]['f1']:.3f}")

    best = max(results, key=lambda k: results[k]["auc"])
    print(f"\n[BEST]     ✓ {best}  (AUC = {results[best]['auc']:.3f})")
    return results, best


# ─────────────────────────────────────────────
#  5. PLOT: OVERVIEW DASHBOARD
# ─────────────────────────────────────────────
def plot_overview(results, best, X, y, feature_cols):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("CreditIQ — Overview Dashboard", fontsize=16,
                 color=COLORS["accent"], fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Metric Summary (top-left) ──
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.set_title("Best Model Metrics", pad=10)
    r = results[best]
    metrics = [
        ("ROC-AUC",   r["auc"],       COLORS["green"]),
        ("F1-Score",  r["f1"],        COLORS["accent"]),
        ("Precision", r["precision"], COLORS["warn"]),
        ("Recall",    r["recall"],    COLORS["orange"]),
        ("Accuracy",  r["accuracy"],  COLORS["text"]),
    ]
    for i, (label, val, col) in enumerate(metrics):
        y_pos = 0.85 - i * 0.18
        ax0.text(0.05, y_pos, label, transform=ax0.transAxes,
                 fontsize=10, color=COLORS["muted"])
        ax0.text(0.65, y_pos, f"{val:.3f}", transform=ax0.transAxes,
                 fontsize=14, color=col, fontweight="bold")
    ax0.text(0.05, 0.02, f"Model: {best}", transform=ax0.transAxes,
             fontsize=8, color=COLORS["muted"])

    # ── Feature Importance (top-mid) ──
    ax1 = fig.add_subplot(gs[0, 1])
    rf = results["Random Forest"]["model"]
    imp = rf.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": imp})
    fi_df = fi_df.sort_values("importance", ascending=True).tail(10)
    bars = ax1.barh(fi_df["feature"], fi_df["importance"],
                    color=[COLORS["accent"], COLORS["green"], COLORS["warn"],
                           COLORS["orange"], COLORS["danger"]] * 3)
    ax1.set_title("Feature Importance (RF)")
    ax1.set_xlabel("Importance", color=COLORS["muted"])
    ax1.tick_params(axis="y", labelsize=7)

    # ── Class Distribution (top-right) ──
    ax2 = fig.add_subplot(gs[0, 2])
    vc = y.value_counts()
    wedges, texts, autotexts = ax2.pie(
        vc.values,
        labels=["Good Credit", "Default"],
        autopct="%1.1f%%",
        colors=[COLORS["green"] + "cc", COLORS["danger"] + "cc"],
        startangle=90,
        wedgeprops={"edgecolor": "#1e3448", "linewidth": 2},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_color("#050a0f"); at.set_fontsize(9); at.set_fontweight("bold")
    centre = plt.Circle((0, 0), 0.5, fc="#0d1620")
    ax2.add_patch(centre)
    ax2.text(0, 0, f"{len(y)}\nRECORDS", ha="center", va="center",
             fontsize=9, color=COLORS["text"], fontweight="bold")
    ax2.set_title("Class Distribution")

    # ── Score Distribution (bottom-left + mid) ──
    ax3 = fig.add_subplot(gs[1, :2])
    y_prob = results[best]["y_prob"]
    y_test_arr = results[best]["y_pred"]   # proxy — use actual test labels if stored

    # Simulate distribution using predicted probabilities (good vs default)
    # We'll use the full dataset predictions for illustration
    rf_model = results["Random Forest"]["model"]
    good_probs  = y_prob[y_test_arr == 1]
    bad_probs   = y_prob[y_test_arr == 0]
    ax3.hist(good_probs, bins=25, alpha=0.6, color=COLORS["green"],
             label="Good Credit", edgecolor="#050a0f")
    ax3.hist(bad_probs,  bins=25, alpha=0.6, color=COLORS["danger"],
             label="Default",     edgecolor="#050a0f")
    ax3.set_title("Score Distribution by Risk Class")
    ax3.set_xlabel("Predicted Probability (Good Credit)")
    ax3.set_ylabel("Count")
    ax3.legend(facecolor="#122030", edgecolor="#1e3448", labelcolor=COLORS["text"])
    ax3.grid(True, alpha=0.3)

    # ── CV Scores (bottom-right) ──
    ax4 = fig.add_subplot(gs[1, 2])
    cv_scores = results[best]["cv"]
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    bars = ax4.bar(folds, cv_scores, color=COLORS["accent"] + "99",
                   edgecolor=COLORS["accent"], linewidth=1.2)
    ax4.axhline(cv_scores.mean(), color=COLORS["green"], linestyle="--",
                linewidth=1.5, label=f"Mean: {cv_scores.mean():.3f}")
    ax4.set_ylim(0.7, 1.0)
    ax4.set_title(f"5-Fold Cross Validation ({best})")
    ax4.set_ylabel("Accuracy")
    ax4.legend(facecolor="#122030", edgecolor="#1e3448", labelcolor=COLORS["text"])
    ax4.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, cv_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=8, color=COLORS["text"])

    plt.savefig("creditiq_overview.png", dpi=150, bbox_inches="tight",
                facecolor="#050a0f")
    plt.show(block=False)
    print("[SAVED]    creditiq_overview.png")


# ─────────────────────────────────────────────
#  6. PLOT: MODEL COMPARISON
# ─────────────────────────────────────────────
def plot_model_comparison(results, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("CreditIQ — Model Comparison", fontsize=15,
                 color=COLORS["accent"], fontweight="bold")
    fig.patch.set_facecolor("#050a0f")

    # ── ROC Curves ──
    ax = axes[0]
    ax.set_facecolor("#0d1620")
    ax.plot([0, 1], [0, 1], "--", color=COLORS["muted"], linewidth=1, label="Random (AUC=0.50)")
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2,
                label=f"{name} (AUC={r['auc']:.3f})")
    ax.set_title("ROC Curves", color=COLORS["accent"])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(facecolor="#122030", edgecolor="#1e3448",
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Metric Bar Chart ──
    ax = axes[1]
    ax.set_facecolor("#0d1620")
    metric_names = ["AUC", "Accuracy", "F1", "Precision", "Recall"]
    metric_keys  = ["auc", "accuracy", "f1", "precision", "recall"]
    n_models  = len(results)
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    bar_w = 0.18

    for i, (name, r) in enumerate(results.items()):
        vals = [r[k] for k in metric_keys]
        offset = (i - n_models / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w,
                      label=name, color=MODEL_COLORS[name] + "cc",
                      edgecolor=MODEL_COLORS[name], linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Metric Comparison", color=COLORS["accent"])
    ax.set_ylabel("Score")
    ax.legend(facecolor="#122030", edgecolor="#1e3448",
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("creditiq_model_comparison.png", dpi=150, bbox_inches="tight",
                facecolor="#050a0f")
    plt.show(block=False)
    print("[SAVED]    creditiq_model_comparison.png")


# ─────────────────────────────────────────────
#  7. PLOT: PERFORMANCE METRICS
# ─────────────────────────────────────────────
def plot_performance_metrics(results, best, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"CreditIQ — Performance Metrics ({best})",
                 fontsize=14, color=COLORS["accent"], fontweight="bold")
    fig.patch.set_facecolor("#050a0f")

    r = results[best]
    y_pred = r["y_pred"]
    y_prob = r["y_prob"]

    # ── Confusion Matrix ──
    ax = axes[0]
    ax.set_facecolor("#0d1620")
    cm = confusion_matrix(y_test, y_pred)
    cm_labels = np.array([["TN", "FP"], ["FN", "TP"]])
    cell_colors = [
        [COLORS["accent"] + "33",  COLORS["warn"] + "33"],
        [COLORS["danger"] + "33",  COLORS["green"] + "33"],
    ]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j-0.5, 1.5-i-0.5), 1, 1,
                                       color=cell_colors[i][j], zorder=0))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                xticklabels=["Pred: Good", "Pred: Default"],
                yticklabels=["Actual: Good", "Actual: Default"],
                annot_kws={"size": 18, "weight": "bold", "color": COLORS["text"]},
                cbar=False, linewidths=2, linecolor="#1e3448",
                alpha=0.7)
    ax.set_title("Confusion Matrix", color=COLORS["accent"])
    ax.tick_params(colors=COLORS["muted"])

    # ── Precision-Recall Curve ──
    ax = axes[1]
    ax.set_facecolor("#0d1620")
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(rec, prec, color=COLORS["green"], linewidth=2,
            label=f"AP = {ap:.3f}")
    ax.fill_between(rec, prec, alpha=0.1, color=COLORS["green"])
    baseline = y_test.mean()
    ax.axhline(baseline, color=COLORS["muted"], linestyle="--",
               linewidth=1, label=f"Baseline ({baseline:.2f})")
    ax.set_title("Precision-Recall Curve", color=COLORS["accent"])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.legend(facecolor="#122030", edgecolor="#1e3448",
              labelcolor=COLORS["text"])
    ax.grid(True, alpha=0.3)

    # ── Classification Report Heatmap ──
    ax = axes[2]
    ax.set_facecolor("#0d1620")
    report = classification_report(y_test, y_pred,
                                   target_names=["Default", "Good Credit"],
                                   output_dict=True)
    report_df = pd.DataFrame(report).T.drop(columns=["support"], errors="ignore")
    report_df = report_df.loc[["Default", "Good Credit", "macro avg", "weighted avg"]]
    sns.heatmap(report_df.astype(float), annot=True, fmt=".3f", ax=ax,
                cmap="YlOrRd", vmin=0.5, vmax=1.0,
                linewidths=1, linecolor="#1e3448",
                annot_kws={"size": 10, "color": "#050a0f"})
    ax.set_title("Classification Report", color=COLORS["accent"])
    ax.tick_params(colors=COLORS["muted"])

    plt.tight_layout()
    plt.savefig("creditiq_performance.png", dpi=150, bbox_inches="tight",
                facecolor="#050a0f")
    plt.show(block=False)
    print("[SAVED]    creditiq_performance.png")


# ─────────────────────────────────────────────
#  8. PLOT: FEATURE CORRELATION HEATMAP
# ─────────────────────────────────────────────
def plot_correlation_heatmap(X, feature_cols):
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#050a0f")
    ax.set_facecolor("#0d1620")
    fig.suptitle("CreditIQ — Feature Correlation Heatmap",
                 fontsize=14, color=COLORS["accent"], fontweight="bold")

    corr = pd.DataFrame(X, columns=feature_cols).corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # show full matrix

    sns.heatmap(corr, ax=ax, cmap="RdYlGn", vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.5, linecolor="#050a0f",
                xticklabels=feature_cols, yticklabels=feature_cols,
                cbar_kws={"shrink": 0.8})
    ax.tick_params(axis="x", rotation=45, labelsize=8, colors=COLORS["muted"])
    ax.tick_params(axis="y", rotation=0,  labelsize=8, colors=COLORS["muted"])

    plt.tight_layout()
    plt.savefig("creditiq_heatmap.png", dpi=150, bbox_inches="tight",
                facecolor="#050a0f")
    plt.show(block=False)
    print("[SAVED]    creditiq_heatmap.png")


# ─────────────────────────────────────────────
#  9. LIVE PREDICTION FUNCTION
# ─────────────────────────────────────────────
def compute_credit_score(income, age, credit_amount, duration,
                          dti, history_months, employment,
                          existing_credits, balance, missed_payments):
    """
    Rule-based credit score calculator (300–850 scale).
    Mirrors the JavaScript scoring logic from the HTML app.
    """
    score = 500.0
    score += min(income / 200_000 * 120, 120)
    if 35 <= age <= 55:
        score += 30
    elif age >= 25:
        score += 15
    score -= dti * 2.5
    score -= (duration - 6) / 66 * 40
    score += min(history_months / 240 * 60, 60)
    score += employment * 15       # 0=unemployed, 1=self, 2=part, 3=full
    score += balance * 20          # 1=negative, 2=low, 3=medium, 4=high
    score -= missed_payments * 35
    score += 10 if existing_credits <= 2 else -10
    ratio = credit_amount / max(income, 1)
    if ratio < 0.2:
        score += 20
    elif ratio > 0.5:
        score -= 30
    return max(300, min(850, round(score)))


def live_prediction(model, scaler, feature_cols, **kwargs):
    """
    Predict creditworthiness for a new applicant.
    kwargs should match feature column names used during training.
    """
    # Build a feature vector of zeros, fill in known values
    row = {col: 0 for col in feature_cols}
    row.update(kwargs)

    # Derived features
    dur = row.get("duration", 1)
    amt = row.get("credit_amount", 0)
    age = row.get("age", 30)
    row["dti_ratio"]        = amt / (dur + 1)
    row["credit_per_month"] = amt / max(dur, 1)
    row["age_credit_ratio"] = age / (amt / 1000 + 1)

    X_new = pd.DataFrame([row])[feature_cols]
    X_sc  = scaler.transform(X_new)
    prob  = model.predict_proba(X_sc)[0][1]   # P(good credit)

    # Compute rule-based score for display
    score = compute_credit_score(
        income=kwargs.get("income_proxy", 50000),
        age=age,
        credit_amount=amt,
        duration=dur,
        dti=kwargs.get("dti_pct", 30),
        history_months=kwargs.get("history_months", 48),
        employment=kwargs.get("employment_level", 2),
        existing_credits=row.get("existing_credits", 1),
        balance=kwargs.get("balance_level", 2),
        missed_payments=kwargs.get("missed_payments", 0),
    )

    if score >= 680:
        verdict = "✅ APPROVED"
    elif score >= 540:
        verdict = "⚠️  MANUAL REVIEW"
    else:
        verdict = "❌ DENIED"

    confidence = abs(prob - 0.5) * 2 * 100

    print("\n" + "─" * 50)
    print("  LIVE PREDICTION RESULT")
    print("─" * 50)
    print(f"  Credit Score   : {score} / 850")
    print(f"  P(Good Credit) : {prob*100:.1f}%")
    print(f"  Confidence     : {confidence:.0f}%")
    print(f"  Decision       : {verdict}")
    print("─" * 50)
    return {"score": score, "probability": prob, "verdict": verdict}


# ─────────────────────────────────────────────
#  10. PRINT SUMMARY TABLE
# ─────────────────────────────────────────────
def print_summary(results):
    print("\n" + "=" * 70)
    print(f"  {'MODEL':<26} {'AUC':>7} {'ACC':>7} {'F1':>7} {'PREC':>7} {'REC':>7}")
    print("=" * 70)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["auc"]):
        marker = " ← BEST" if r["auc"] == max(v["auc"] for v in results.values()) else ""
        print(f"  {name:<26} {r['auc']:>7.3f} {r['accuracy']:>7.3f} "
              f"{r['f1']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f}{marker}")
    print("=" * 70)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load & preprocess
    X, y, feature_cols = load_and_preprocess()

    # 2. Split & scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # 3. Train all models
    trained = train_models(X_train, y_train)

    # 4. Evaluate
    results, best_model = evaluate_models(trained, X_test, y_test)

    # 5. Print summary table
    print_summary(results)

    # 6. Plots
    plot_overview(results, best_model, X, y, feature_cols)
    plot_model_comparison(results, y_test)
    plot_performance_metrics(results, best_model, y_test)
    plot_correlation_heatmap(X, feature_cols)

    # 7. Live Prediction Demo
    best_clf = results[best_model]["model"]
    live_prediction(
        model=best_clf,
        scaler=scaler,
        feature_cols=feature_cols,
        credit_amount=8000,
        duration=24,
        age=35,
        existing_credits=2,
        # Extra kwargs for rule-based score display
        income_proxy=55000,
        dti_pct=28,
        history_months=48,
        employment_level=3,
        balance_level=3,
        missed_payments=0,
    )