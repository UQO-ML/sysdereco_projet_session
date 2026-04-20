"""Visualisations centralisees pour l'experience ECR.

Chaque fonction correspond a un aspect de l'article
"Towards Empathetic Conversational Recommender Systems" (RecSys 2024) et peut etre
appelee directement depuis le notebook `ecr_experiment.ipynb`.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# Dataset / EDA plots (ReDial)
# ---------------------------------------------------------------------------


def plot_feedback_distribution(df_feedback: pd.DataFrame) -> None:
    """Bar plot de la distribution des feedbacks `like / dislike / not say`.

    Article Section 5.1 : distribution 81.1 / 4.9 / 14.0 (%).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=df_feedback,
        x="feedback",
        y="count",
        hue="feedback",
        palette="muted",
        legend=False,
        ax=ax,
    )
    total = df_feedback["count"].sum()
    for patch, (_, row) in zip(ax.patches, df_feedback.iterrows()):
        pct = 100 * row["count"] / total
        ax.annotate(
            f"{pct:.1f}%",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )
    ax.set_title("ReDial - User Feedback Distribution")
    ax.set_xlabel("Feedback label")
    ax.set_ylabel("Count")
    fig.tight_layout()
    plt.show()


def plot_feedback_weights(weights: Mapping[str, float]) -> None:
    """Visualise la fonction de mapping `m(f)` du reweighting (Eq. 7).

    Article Section 5.4 : like=2.0, dislike=1.0, not say=0.5.
    """
    df = pd.DataFrame({"feedback": list(weights.keys()), "weight": list(weights.values())})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=df,
        x="feedback",
        y="weight",
        hue="feedback",
        palette="flare",
        legend=False,
        ax=ax,
    )
    for patch, value in zip(ax.patches, df["weight"].tolist()):
        ax.annotate(
            f"{value:.1f}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
        )
    ax.set_title("Feedback-aware item reweighting m(f) - Section 5.4")
    ax.set_xlabel("User feedback")
    ax.set_ylabel("Weight scalar")
    fig.tight_layout()
    plt.show()


def plot_emotion_label_distribution(df_emotions: pd.DataFrame) -> None:
    """Distribution des 9 labels d'emotion annotes par GPT-3.5 (Section 4.1.1).

    Labels : like, curious, happy, grateful, negative, neutral, nostalgia,
    agreement, surprise.
    """
    order = df_emotions.sort_values("share", ascending=False)["emotion"].tolist()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        data=df_emotions,
        x="emotion",
        y="share",
        hue="emotion",
        order=order,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    for patch, value in zip(ax.patches, df_emotions.set_index("emotion").loc[order, "share"].tolist()):
        ax.annotate(
            f"{value:.1f}%",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Utterance-level user emotion labels (GPT-3.5 annotation)")
    ax.set_xlabel("Emotion label")
    ax.set_ylabel("Share (%)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    plt.show()


def plot_review_coverage(df_reviews: pd.DataFrame) -> None:
    """Volume d'avis IMDb collectes par backbone (Section 5.1).

    DialoGPT : 34,953 avis / 4,092 films ; Llama 2-7B-Chat : 2,459 / 1,553.
    """
    long_df = df_reviews.melt(id_vars=["backbone"], var_name="entity", value_name="count")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(
        data=long_df[long_df["entity"] == "reviews"],
        x="backbone",
        y="count",
        hue="backbone",
        palette="crest",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Filtered IMDb reviews per backbone")
    axes[0].set_ylabel("# reviews")
    sns.barplot(
        data=long_df[long_df["entity"] == "movies"],
        x="backbone",
        y="count",
        hue="backbone",
        palette="flare",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Movies covered per backbone")
    axes[1].set_ylabel("# movies")
    for ax in axes:
        ax.set_xlabel("")
        for patch in ax.patches:
            ax.annotate(
                f"{int(patch.get_height()):,}",
                (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Objective metrics (Table 1) + ablation (Table 3)
# ---------------------------------------------------------------------------


def plot_objective_metrics(df_obj: pd.DataFrame) -> None:
    """Bar chart groupe des metriques objectives Table 1."""
    metrics = ["AUC", "RT@1", "RT@10", "RT@50", "R@1", "R@10", "R@50"]
    long_df = df_obj.melt(
        id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long_df, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_title("Objective metrics per model (Table 1 of the paper)")
    ax.set_ylim(0, 1)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    plt.show()


def plot_model_rankings(df_obj: pd.DataFrame, metric: str = "AUC") -> None:
    """Classement simple des modeles sur une metrique objective."""
    ranked = df_obj.sort_values(metric, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=ranked,
        x="Model",
        y=metric,
        hue="Model",
        palette="crest",
        legend=False,
        ax=ax,
    )
    for patch, value in zip(ax.patches, ranked[metric].tolist()):
        ax.annotate(
            f"{value:.3f}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(f"Model ranking by {metric}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    fig.tight_layout()
    plt.show()


def plot_ablation_study(df_ablation: pd.DataFrame) -> None:
    """Heatmap de l'etude d'ablation (Table 3).

    Variants : UniCRS, ECR[L], ECR[LS], ECR[LG], ECR.
    """
    metrics = [c for c in df_ablation.columns if c != "Model"]
    mat = df_ablation.set_index("Model")[metrics]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Score"},
        ax=ax,
    )
    ax.set_title("Ablation study on recommendation (Table 3)")
    ax.set_ylabel("")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Subjective metrics (Table 2)
# ---------------------------------------------------------------------------


def plot_subjective_metrics(df_subj: pd.DataFrame, title: str) -> None:
    """Bar chart groupe des 5 metriques subjectives (Emo Int / Emo Pers / ...)."""
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
    """Radar chart pour un modele sur les 5 metriques subjectives."""
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
    ax.set_ylim(0, 9)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_llm_vs_human_correlation(
    df_subj_llm: pd.DataFrame, df_subj_human: pd.DataFrame
) -> None:
    """Scatter `LLM-based scorer` vs `Human annotators` pour chaque metrique.

    Article Section 5.6 : accord Cohen's kappa = 0.62 (substantial agreement).
    """
    metrics = ["Emo Int", "Emo Pers", "Log Pers", "Info", "Life"]
    merged = df_subj_llm.merge(df_subj_human, on="Model", suffixes=("_llm", "_human"))

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), sharey=True)
    for ax, metric in zip(axes, metrics):
        x = merged[f"{metric}_llm"]
        y = merged[f"{metric}_human"]
        sns.scatterplot(x=x, y=y, hue=merged["Model"], s=90, ax=ax, legend=(ax is axes[-1]))
        lo = float(np.nanmin([x.min(), y.min()])) - 0.5
        hi = float(np.nanmax([x.max(), y.max()])) + 0.5
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(metric)
        ax.set_xlabel("LLM scorer")
    axes[0].set_ylabel("Human annotators")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle("LLM-based scorer vs human annotators (Table 2)")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------


def plot_training_loss(
    history: pd.DataFrame,
    step_col: str = "step",
    loss_col: str = "loss",
    hue_col: Optional[str] = "stage",
) -> None:
    """Courbe(s) de loss par stage d'entrainement (pre / rec / emp).

    `history` doit contenir au moins les colonnes `step_col` et `loss_col`.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    if hue_col and hue_col in history.columns:
        sns.lineplot(data=history, x=step_col, y=loss_col, hue=hue_col, ax=ax)
    else:
        sns.lineplot(data=history, x=step_col, y=loss_col, ax=ax)
    ax.set_title("Training loss across ECR stages")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    plt.show()


def plot_hyperparam_sweep(
    df_sweep: pd.DataFrame, param: str, metric: str = "AUC"
) -> None:
    """Sensibilite du modele a un hyperparametre (ex. weight scalars, beta, n_f).

    Article Appendix B : analyse du reweighting `like/dislike/not say` et de la
    quantite de connaissance injectee dans le prompt genere.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=df_sweep, x=param, y=metric, marker="o", ax=ax)
    ax.set_title(f"Sensitivity of {metric} to hyperparameter '{param}'")
    ax.set_xlabel(param)
    ax.set_ylabel(metric)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Helper exposed to the notebook
# ---------------------------------------------------------------------------


ALL_PLOTS: Iterable[str] = (
    "plot_feedback_distribution",
    "plot_feedback_weights",
    "plot_emotion_label_distribution",
    "plot_review_coverage",
    "plot_objective_metrics",
    "plot_model_rankings",
    "plot_ablation_study",
    "plot_subjective_metrics",
    "plot_subjective_radar",
    "plot_llm_vs_human_correlation",
    "plot_training_loss",
    "plot_hyperparam_sweep",
)
