import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="notebook")


def plot_feedback_distribution(df_feedback: pd.DataFrame) -> None:
    """Bar plot for feedback label distribution."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df_feedback, x="feedback", y="count", ax=ax, palette="muted")
    ax.set_title("User Feedback Distribution")
    ax.set_xlabel("Feedback Label")
    ax.set_ylabel("Count")
    fig.tight_layout()
    plt.show()


def plot_objective_metrics(df_obj: pd.DataFrame) -> None:
    """Grouped bar chart for objective recommendation metrics."""
    metrics = ["AUC", "RT@1", "RT@10", "RT@50", "R@1", "R@10", "R@50"]
    long_df = df_obj.melt(
        id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long_df, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_title("Objective Metrics by Model")
    ax.set_ylim(0, 1)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    plt.show()


def plot_subjective_metrics(df_subj: pd.DataFrame, title: str) -> None:
    """Grouped bar chart for subjective response quality metrics."""
    metrics = ["Emo Int", "Emo Pers", "Log Pers", "Info", "Life"]
    long_df = df_subj.melt(
        id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long_df, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_title(title)
    ax.set_ylim(0, 9)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    plt.show()


def plot_subjective_radar(df_subj: pd.DataFrame, model_name: str, title: str) -> None:
    """Radar chart for one model on subjective metrics."""
    metrics = ["Emo Int", "Emo Pers", "Log Pers", "Info", "Life"]
    row = df_subj.loc[df_subj["Model"] == model_name]
    if row.empty:
        raise ValueError(f"Model '{model_name}' not found.")
    values = row.iloc[0][metrics].tolist()

    values_closed = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles_closed, values_closed, linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.2)
    ax.set_xticks(angles)
    ax.set_xticklabels(metrics)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_model_rankings(df_obj: pd.DataFrame, metric: str = "AUC") -> None:
    """Simple ranking chart for one objective metric."""
    ranked = df_obj.sort_values(metric, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=ranked, x="Model", y=metric, ax=ax, palette="crest")
    ax.set_title(f"Model Ranking by {metric}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    fig.tight_layout()
    plt.show()
