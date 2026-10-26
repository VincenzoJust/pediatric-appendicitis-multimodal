"""Tabular / image / fused models and the sequential concept bottleneck. All fit on train, tuned on val."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler

from .data import CBM_TAB, SEED, concepts, preproc, tab_features

N_PCA = 32  # image PCs fed to fused model / concept heads (fixed, not tuned)
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
LR = lambda C: LogisticRegression(C=C, max_iter=5000)
GB_GRID = {"n_estimators": [50, 100, 200], "max_depth": [1, 2, 3]}


def tune(make, grid, F, y, tr, va):
    """Fit on train for every grid point, keep the best by validation AUROC."""
    best = (-1, None)
    for par in ParameterGrid(grid):
        m = make(**par).fit(F["train"], y[tr])
        a = roc_auc_score(y[va], m.predict_proba(F["val"])[:, 1])
        if a > best[0]:
            best = (a, m)
    return best[1]


def fit_cbm(d, H, y, s):
    """Concepts predicted from H (image PCs + demographics/labs); diagnosis from predicted concepts only."""
    tr, va = s["train"], s["val"]
    Cm = concepts(d).to_numpy()
    fit_heads = lambda C, rows, X: [LR(C).fit(X[ok], Cm[rows, j][ok])
                                    for j in range(Cm.shape[1])
                                    for ok in [~np.isnan(Cm[rows, j])]]
    probs = lambda hs, X: np.column_stack([h.predict_proba(X)[:, 1] for h in hs])
    cauc = lambda hs, k: np.nanmean([roc_auc_score(Cm[s[k], j][ok], hs[j].predict_proba(H[k][ok])[:, 1])
                                     for j in range(Cm.shape[1]) for ok in [~np.isnan(Cm[s[k], j])]])
    # concept-head C: best mean concept AUROC on val (concept labels only, no diagnosis)
    heads = {C: fit_heads(C, tr, H["train"]) for C in C_GRID}
    Cc = max(C_GRID, key=lambda C: cauc(heads[C], "val"))
    # label head is trained on out-of-fold concept probabilities, so it sees realistic concept noise
    oof = np.zeros((len(tr), Cm.shape[1]))
    for a, b in KFold(5, shuffle=True, random_state=SEED).split(tr):
        oof[b] = probs(fit_heads(Cc, tr[a], H["train"][a]), H["train"][b])
    P = {"train": oof, **{k: probs(heads[Cc], H[k]) for k in ("val", "test")}}
    lab = tune(lambda C: LR(C), {"C": C_GRID}, P, y, tr, va)
    return lab, heads[Cc], P, list(concepts(d).columns)


def fit_all(d, E, s, y):
    """Fit every model with labels y (real or shuffled). Returns {name: (p_val, p_test)} and extras."""
    tr, va = s["train"], s["val"]
    X = tab_features(d)
    pre = preproc(X.iloc[tr]).fit(X.iloc[tr])
    T = {k: pre.transform(X.iloc[i]) for k, i in s.items()}
    Xh = d[CBM_TAB]
    preh = preproc(Xh.iloc[tr]).fit(Xh.iloc[tr])
    sc = StandardScaler().fit(E[tr])
    Z = {k: sc.transform(E[i]) for k, i in s.items()}
    pca = PCA(N_PCA, whiten=True, random_state=SEED).fit(Z["train"])
    P = {k: pca.transform(Z[k]) for k in s}
    out, ex = {}, {"pre": pre, "T": T}

    def run(name, make, grid, F):
        m = tune(make, grid, F, y, tr, va)
        out[name] = tuple(m.predict_proba(F[k])[:, 1] for k in ("val", "test"))
        ex[name] = m

    run("tab_lr", lambda C: LR(C), {"C": C_GRID}, T)
    run("tab_gb", lambda **p: GradientBoostingClassifier(random_state=SEED, **p), GB_GRID, T)
    run("img_lr", lambda C: LR(C), {"C": C_GRID}, Z)
    run("fused_lr", lambda C: LR(C), {"C": C_GRID},
        {k: np.hstack([T[k], P[k]]) for k in s})
    H = {k: np.hstack([preh.transform(Xh.iloc[i]), P[k]]) for k, i in s.items()}
    lab, heads, Pc, names = fit_cbm(d, H, y, s)
    out["cbm"] = tuple(lab.predict_proba(Pc[k])[:, 1] for k in ("val", "test"))
    ex["cbm"] = dict(label_head=lab, heads=heads, concept_probs=Pc, names=names, H=H)
    return out, ex


def fit_temperature(y, p):
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    return minimize_scalar(lambda t: log_loss(y, expit(z / t), labels=[0, 1]),
                           bounds=(0.05, 20), method="bounded").x


def apply_temperature(p, t):
    return expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) / t)
