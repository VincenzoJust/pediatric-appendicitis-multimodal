"""Metrics, patient-level bootstrap CIs, figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

COLORS = {"tab_lr": "#0072B2", "tab_gb": "#56B4E9", "img_lr": "#E69F00", "fused_lr": "#009E73", "cbm": "#CC79A7"}
LABELS = {"tab_lr": "Tabular LR", "tab_gb": "Tabular GB", "img_lr": "Image LR", "fused_lr": "Fused LR", "cbm": "Concept bottleneck"}


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(bins) if (b == k).any())


def sens_at_spec(y, p, spec=0.9):
    fpr, tpr, _ = roc_curve(y, p)
    return tpr[fpr <= 1 - spec].max()


METRICS = {"AUROC": roc_auc_score, "AUPRC": average_precision_score, "Sens@90Spec": sens_at_spec,
           "ECE": ece, "Brier": brier_score_loss}


def bootstrap(y, probs, n=1000, seed=0):
    """Patient-level bootstrap (one row = one patient). Same resamples for every model -> paired.
    Returns {(model, metric): (estimate, array of n bootstrap values)}."""
    rng = np.random.default_rng(seed)
    draws = [i for i in rng.integers(0, len(y), (n, len(y))) if 0 < y[i].sum() < len(i)]
    return {(m, k): (f(y, p), np.array([f(y[i], p[i]) for i in draws]))
            for m, p in probs.items() for k, f in METRICS.items()}


def ci(a):
    return np.percentile(a, [2.5, 97.5])


def plot_roc(y, probs, path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for m, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=COLORS[m], label=f"{LABELS[m]} ({roc_auc_score(y, p):.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.axvline(0.1, color="grey", lw=0.8, ls="--")
    ax.set(xlabel="1 - specificity", ylabel="Sensitivity", title="Test ROC (AUROC)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_reliability(y, raw, temp, path, bins=10):
    fig, axs = plt.subplots(1, len(raw), figsize=(3.2 * len(raw), 3.4), sharey=True)
    for ax, m in zip(axs, raw):
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        for p, ls, lab in ((raw[m], "--", "raw"), (temp[m], "-", "temp-scaled")):
            b = np.minimum((p * bins).astype(int), bins - 1)
            ks = [k for k in range(bins) if (b == k).any()]
            ax.plot([p[b == k].mean() for k in ks], [y[b == k].mean() for k in ks], ls, marker="o",
                    ms=3, color=COLORS[m], alpha=0.5 if lab == "raw" else 1, label=lab)
        ax.set(title=LABELS[m], xlabel="Predicted probability")
        ax.legend(fontsize=7, loc="upper left")
    axs[0].set_ylabel("Observed fraction positive")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_shap(gb, pre, T_test, path, top=15):
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    sv = shap.TreeExplainer(gb).shap_values(T_test)
    shap.summary_plot(sv, T_test, feature_names=names, max_display=top, show=False)
    plt.title("SHAP, tabular gradient boosting (test set)")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_concepts(cbm, p_test, path):
    """Left: label-head coefficient per 1 SD of predicted concept prob (log-odds). Right: contribution
    vs the average training patient, coef x (p - mean p), for the lowest / median / highest-risk test patients."""
    names, w = cbm["names"], cbm["label_head"].coef_[0]
    Ptr, Pt = cbm["concept_probs"]["train"], cbm["concept_probs"]["test"]
    mu, coef = Ptr.mean(0), w * Ptr.std(0)
    order = np.argsort(coef)
    pick = np.argsort(p_test)[[0, len(p_test) // 2, -1]]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    axs[0].barh(range(len(names)), coef[order], color="#009E73")
    axs[0].set_yticks(range(len(names)), [names[i] for i in order], fontsize=8)
    axs[0].set_title("Label-head coef per SD of concept prob")
    for ax, i in zip(axs[1:], pick):
        ax.barh(range(len(names)), (w * (Pt[i] - mu))[order], color="#CC79A7")
        ax.set_title(f"Test patient, P(appendicitis)={p_test[i]:.2f}", fontsize=9)
        ax.set_xlabel("coef x (concept prob - train mean)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
