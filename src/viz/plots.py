"""Visualisations centralisees pour l'experience ECR.

Chaque fonction correspond a un aspect de l'article
"Towards Empathetic Conversational Recommender Systems" (RecSys 2024) et peut etre
appelee directement depuis le notebook `ecr_experiment.ipynb`.

Persistance :
    Par defaut, chaque fonction appelle `plt.show()` pour affichage inline.
    Pour persister les figures sur disque (rapport, CI, etc.) :

        from src.viz import plots
        plots.set_save_dir("results/figures")
        plots.plot_objective_metrics(df_obj)   # -> results/figures/objective_metrics.png
        plots.set_save_dir(None)               # restaure le mode affichage seul

    Le format est PNG @ 150 dpi, bbox tight. Les noms de fichier sont derives
    automatiquement du slug passe en interne (stable, versionnable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# Save-or-show dispatch
# ---------------------------------------------------------------------------

_SAVE_DIR: Optional[Path] = None


def set_save_dir(path: Union[str, Path, None]) -> Optional[Path]:
    """Fixe le dossier de persistance PNG (None = mode affichage seul).

    Cree le dossier s'il n'existe pas. Retourne le Path resolu (ou None).
    """
    global _SAVE_DIR
    if path is None:
        _SAVE_DIR = None
        return None
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    _SAVE_DIR = p
    return p


def get_save_dir() -> Optional[Path]:
    return _SAVE_DIR


def _save_or_show(fig: "plt.Figure", slug: str) -> Optional[Path]:
    """Sauve la figure dans `_SAVE_DIR/<slug>.png` (si defini) + affiche inline.

    Retourne le chemin ecrit ou None.
    """
    out: Optional[Path] = None
    if _SAVE_DIR is not None:
        out = _SAVE_DIR / f"{slug}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    return out


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
    _save_or_show(fig, "eda_feedback_distribution")


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
    _save_or_show(fig, "eda_feedback_weights")


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
    _save_or_show(fig, "eda_emotion_distribution")


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
    _save_or_show(fig, "eda_review_coverage")


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
    _save_or_show(fig, "objective_metrics")


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
    _save_or_show(fig, f"model_ranking_{metric.replace('@','at').lower()}")


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
    _save_or_show(fig, "ablation_study")


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
    slug = "subjective_" + (
        "llm" if "LLM" in title or "llm" in title.lower() else
        "human" if "Human" in title or "human" in title.lower() else
        "other"
    )
    _save_or_show(fig, slug)


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
    safe_model = model_name.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_").lower()
    scorer = "llm" if "LLM" in title or "llm" in title.lower() else (
        "human" if "Human" in title or "human" in title.lower() else "other"
    )
    _save_or_show(fig, f"subjective_radar_{safe_model}_{scorer}")


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
    _save_or_show(fig, "llm_vs_human_correlation")


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
    _save_or_show(fig, "training_loss")


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
    _save_or_show(fig, f"hyperparam_sweep_{param}_{metric.replace('@','at').lower()}")


# ---------------------------------------------------------------------------
# Persisted-runs analysis (uses results/run_*.csv accumulated across runs)
# ---------------------------------------------------------------------------


def _short_run_id(rid: str, maxlen: int = 16) -> str:
    """Raccourci lisible pour un run_id timestamp (YYYY-MM-DD_HH-MM-SS)."""
    rid = str(rid)
    return rid if len(rid) <= maxlen else rid[:maxlen]


def plot_runs_vs_article(
    df_comparison: pd.DataFrame,
    metrics: Iterable[str] = ("AUC", "R@1", "R@10", "R@50"),
) -> None:
    """Compare nos runs (lignes `ECR (our run)`) a la reference article `ECR (article)`.

    Attend le schema de `run_comparison_objective.csv` : colonnes `run_id`,
    `Model`, `AUC`, `RT@k`, `R@k`, plus les colonnes de timing/config.

    La barre "reference" utilise une hachure (//) pour marquer qu'elle provient
    de la Table 1 de l'article (Zhang et al. 2024) et non d'une execution locale.
    """
    df = df_comparison.copy()
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        print("[plot_runs_vs_article] aucune metrique reconnue dans df_comparison")
        return
    # Fabrique un label court par ligne : reference = 'paper', sinon short run_id.
    def _label(row):
        rid = str(row.get("run_id", ""))
        if rid == "reference" or "article" in str(row.get("Model", "")).lower():
            return "paper"
        return _short_run_id(rid)

    df["label"] = df.apply(_label, axis=1)
    long_df = df.melt(
        id_vars=["label"], value_vars=list(metrics), var_name="Metric", value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(df) * len(metrics) / 4), 5))
    # Ordre stable du hue (paper en premier -> toujours le meme cote visuel)
    hue_order = ["paper"] + [l for l in df["label"].unique() if l != "paper"]
    sns.barplot(
        data=long_df, x="Metric", y="Score", hue="label",
        hue_order=hue_order, palette="viridis", ax=ax,
    )
    # Hachure sur le container correspondant au hue "paper" (reference article).
    for container, hue_label in zip(ax.containers, hue_order):
        if hue_label == "paper":
            for patch in container:
                patch.set_hatch("//")
                patch.set_edgecolor("black")
    ax.set_title("Our runs vs article reference - objective metrics")
    ax.set_ylim(0, max(0.6, long_df["Score"].max() * 1.15))
    ax.legend(title="Run", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save_or_show(fig, "runs_vs_article")


def plot_phase_timings(df_timing: pd.DataFrame) -> None:
    """Barre empilee du wall-time par phase (pre / rec / rec_export / emp / infer).

    Attend le schema de `run_timing.csv`. Ignore les phases absentes (NaN).
    """
    phases = [
        ("elapsed_pre_sec",        "pre"),
        ("elapsed_rec_sec",        "rec"),
        ("elapsed_rec_export_sec", "rec_export"),
        ("elapsed_emp_sec",        "emp"),
        ("elapsed_infer_sec",      "infer"),
    ]
    avail = [(c, n) for c, n in phases if c in df_timing.columns]
    if not avail:
        print("[plot_phase_timings] aucune colonne elapsed_*_sec trouvee")
        return

    df = df_timing.copy()
    df["label"] = df["run_id"].astype(str).map(_short_run_id)
    stacked = df.set_index("label")[[c for c, _ in avail]].fillna(0) / 60.0  # minutes
    stacked.columns = [n for _, n in avail]

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(stacked) + 6), 5))
    stacked.plot(kind="bar", stacked=True, colormap="Set2", ax=ax, edgecolor="white")
    for i, total in enumerate(stacked.sum(axis=1).tolist()):
        ax.text(i, total + 0.5, f"{total:.0f} min", ha="center", va="bottom", fontsize=9)
    ax.set_title("Wall-time par phase et par run (minutes)")
    ax.set_xlabel("Run")
    ax.set_ylabel("Duree (min)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Phase", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save_or_show(fig, "phase_timings")


def plot_pre_convergence(df_pre: pd.DataFrame) -> None:
    """Trajectoire des metriques PRE (Emotional Semantic Fusion) a travers les runs.

    Attend le schema de `run_pre_metrics.csv` : `recall@k`, `ndcg@k`, `mrr@k`, `loss`.
    """
    df = df_pre.copy()
    df["label"] = df["run_id"].astype(str).map(_short_run_id)
    metrics = ["recall@1", "recall@10", "recall@50", "ndcg@10", "mrr@10"]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        print("[plot_pre_convergence] aucune colonne recall/ndcg/mrr trouvee")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [3, 1]})
    long_df = df.melt(id_vars=["label"], value_vars=metrics, var_name="Metric", value_name="Score")
    sns.lineplot(data=long_df, x="label", y="Score", hue="Metric", marker="o", ax=ax1)
    ax1.set_title("PRE metrics per run")
    ax1.set_xlabel("Run")
    ax1.set_ylabel("Score")
    ax1.tick_params(axis="x", rotation=20)
    ax1.legend(title="Metric", fontsize=8)

    if "loss" in df.columns:
        sns.lineplot(data=df.sort_values("run_id"), x="label", y="loss", marker="o",
                     ax=ax2, color="crimson")
        ax2.set_title("PRE final loss")
        ax2.set_xlabel("Run")
        ax2.set_ylabel("Loss")
        ax2.tick_params(axis="x", rotation=20)
    else:
        ax2.set_visible(False)
    fig.tight_layout()
    _save_or_show(fig, "pre_convergence")


def plot_generation_quality(df_gen: pd.DataFrame) -> None:
    """BLEU@k + Dist@k de la phase generation (infer_emp) par run.

    Attend le schema de `run_generation_metrics.csv`.
    """
    df = df_gen.copy()
    df["label"] = df["run_id"].astype(str).map(_short_run_id)
    bleu_cols = [c for c in ("bleu@1", "bleu@2", "bleu@3", "bleu@4") if c in df.columns]
    dist_cols = [c for c in ("dist@1", "dist@2", "dist@3", "dist@4") if c in df.columns]
    if not bleu_cols and not dist_cols:
        print("[plot_generation_quality] aucune colonne bleu@k / dist@k")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    if bleu_cols:
        long_b = df.melt(id_vars=["label"], value_vars=bleu_cols,
                         var_name="Metric", value_name="Score")
        # BLEU@3 / BLEU@4 tombent regulierement en underflow numerique
        # (~1e-308) avec le lissage cumulatif de l'article -> on clippe a 1e-8
        # pour que l'axe log reste lisible. Les valeurs sous le seuil sont
        # marquees visuellement par une hachure "x".
        FLOOR = 1e-8
        long_b["underflow"] = long_b["Score"] < FLOOR
        long_b["ScoreDisp"] = long_b["Score"].clip(lower=FLOOR)
        sns.barplot(data=long_b, x="Metric", y="ScoreDisp", hue="label",
                    palette="crest", ax=ax1)
        # Hachure les barres en underflow (groupees par hue container order).
        hue_order = long_b["label"].drop_duplicates().tolist()
        for container, hue_label in zip(ax1.containers, hue_order):
            sub = long_b[long_b["label"] == hue_label].reset_index(drop=True)
            for patch, is_uf in zip(container, sub["underflow"].tolist()):
                if is_uf:
                    patch.set_hatch("xx")
                    patch.set_edgecolor("black")
                    patch.set_alpha(0.45)
        ax1.set_yscale("log")
        ax1.set_ylim(FLOOR, 1.0)
        ax1.set_title("BLEU@k par run (log-y, hachure=underflow < 1e-8)")
        ax1.set_ylabel("BLEU score")
        ax1.legend(title="Run", fontsize=7)
    if dist_cols:
        long_d = df.melt(id_vars=["label"], value_vars=dist_cols, var_name="Metric", value_name="Score")
        sns.barplot(data=long_d, x="Metric", y="Score", hue="label", palette="flare", ax=ax2)
        ax2.set_title("Dist@k par run (diversite lexicale)")
        ax2.set_ylabel("Dist score")
        ax2.legend(title="Run", fontsize=7)
    fig.tight_layout()
    _save_or_show(fig, "generation_quality")


def plot_dataset_diagnostic(df_diag: pd.DataFrame) -> None:
    """Taille des splits ReDial generes (train/valid/test) par run.

    Attend le schema de `run_dataset_diagnostic.csv`. Si plusieurs runs, prend
    le plus recent (les splits ne devraient pas changer entre runs sauf patch).
    """
    if df_diag.empty:
        print("[plot_dataset_diagnostic] dataframe vide")
        return
    row = df_diag.sort_values("run_id").iloc[-1]
    splits = ["train", "valid", "test"]
    sources = [
        ("redial_gen_{s}_dbpedia_emo", "redial_gen (emo)"),
        ("redial_gen_{s}_processed",   "redial_gen (processed)"),
        ("redial_{s}_dbpedia_emo",     "redial (emo)"),
        ("redial_{s}_processed",       "redial (processed)"),
    ]
    rows = []
    for s in splits:
        for tpl, label in sources:
            col = tpl.format(s=s)
            if col in row.index and pd.notna(row[col]):
                rows.append({"split": s, "source": label, "count": int(row[col])})
    if not rows:
        print("[plot_dataset_diagnostic] aucune colonne split*_* reconnue")
        return
    df_plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=df_plot, x="split", y="count", hue="source", palette="pastel", ax=ax)
    for patch in ax.patches:
        h = patch.get_height()
        if not np.isfinite(h) or h <= 0:
            continue
        ax.annotate(f"{int(h):,}",
                    (patch.get_x() + patch.get_width() / 2.0, h),
                    ha="center", va="bottom", fontsize=8)
    movie_ids = int(row.get("movie_ids_count", 0) or 0)
    entity = int(row.get("entity_count", 0) or 0)
    rec_json = int(row.get("rec_json_lines", 0) or 0)
    subtitle = (
        f"movie_ids={movie_ids:,} | entities={entity:,} | rec.json lines={rec_json:,}"
    )
    ax.set_title(f"Diagnostic dataset ReDial  -  {subtitle}")
    ax.set_xlabel("Split")
    ax.set_ylabel("# examples")
    ax.legend(title="Source file", fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save_or_show(fig, "dataset_diagnostic")


def plot_config_hash_trajectory(df_comparison: pd.DataFrame,
                                 metric: str = "R@10") -> None:
    """Scatter `metric` vs run_id, colore par `config_hash`.

    Permet de voir d'un coup d'oeil l'impact d'un changement de config (bf16 vs
    fp32, compile vs no-compile, etc.) sur une metrique recall-type.
    """
    if "config_hash" not in df_comparison.columns or metric not in df_comparison.columns:
        print(f"[plot_config_hash_trajectory] colonnes manquantes (config_hash / {metric})")
        return
    df = df_comparison.copy()
    # Ignore la ligne reference article (pas de config_hash valide).
    df = df[~df["Model"].astype(str).str.contains("article", case=False, na=False)]
    df = df[df[metric].notna()]
    if df.empty:
        print("[plot_config_hash_trajectory] aucune ligne utile")
        return
    df["label"] = df["run_id"].astype(str).map(_short_run_id)
    df = df.sort_values("run_id")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.scatterplot(
        data=df, x="label", y=metric, hue="config_hash", style="config_hash",
        s=140, ax=ax,
    )
    # Reference article en ligne horizontale si disponible.
    ref = df_comparison[df_comparison["Model"].astype(str).str.contains("article", case=False, na=False)]
    if not ref.empty and metric in ref.columns and pd.notna(ref[metric].iloc[0]):
        ax.axhline(ref[metric].iloc[0], ls="--", color="crimson", alpha=0.6,
                   label=f"paper {metric}={ref[metric].iloc[0]:.3f}")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="config_hash", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title(f"Trajectoire {metric} par run (couleur = config_hash)")
    ax.set_xlabel("Run")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    _save_or_show(fig, f"config_hash_trajectory_{metric.replace('@','at').lower()}")


# ---------------------------------------------------------------------------
# Helper exposed to the notebook
# ---------------------------------------------------------------------------


ALL_PLOTS: Iterable[str] = (
    # EDA / dataset
    "plot_feedback_distribution",
    "plot_feedback_weights",
    "plot_emotion_label_distribution",
    "plot_review_coverage",
    # Table 1 / 3
    "plot_objective_metrics",
    "plot_model_rankings",
    "plot_ablation_study",
    # Table 2
    "plot_subjective_metrics",
    "plot_subjective_radar",
    "plot_llm_vs_human_correlation",
    # Training diagnostics + sweeps
    "plot_training_loss",
    "plot_hyperparam_sweep",
    # New: persisted runs analysis (exploite results/run_*.csv)
    "plot_runs_vs_article",
    "plot_phase_timings",
    "plot_pre_convergence",
    "plot_generation_quality",
    "plot_dataset_diagnostic",
    "plot_config_hash_trajectory",
)
