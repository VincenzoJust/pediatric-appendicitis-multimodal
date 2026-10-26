"""Single entry point: python scripts/run.py  (download -> inspect -> embed -> models -> eval -> figures)."""
import random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

from src import data, evaluate as ev, models

N_PERM = 20
RES = ROOT / "results"
(RES / "figures").mkdir(parents=True, exist_ok=True)
random.seed(data.SEED); np.random.seed(data.SEED)


def inspect(d, imgs):
    full = data.load_all()
    n_img = {n: len(imgs[n]) for n in d.US_Number}
    xl = set(full.US_Number.dropna().astype(int))
    rows = [("funnel", "patients in app_data.xlsx", len(full)),
            ("funnel", "with Diagnosis", full.Diagnosis.notna().sum()),
            ("funnel", "with Diagnosis and US_Number", full.dropna(subset=["Diagnosis", "US_Number"]).shape[0]),
            ("funnel", "cohort: + >=1 image", len(d)),
            ("images", "image files in archive", sum(map(len, imgs.values()))),
            ("images", "distinct subject numbers in archive", len(imgs)),
            ("images", "archive subjects NOT in xlsx US_Number (unlabelled, unused)", len(set(imgs) - xl)),
            ("images", "images in cohort", sum(n_img.values())),
            ("images", "images/patient min", min(n_img.values())),
            ("images", "images/patient median", np.median(list(n_img.values()))),
            ("images", "images/patient mean", round(np.mean(list(n_img.values())), 2)),
            ("images", "images/patient max", max(n_img.values())),
            ("balance", "full dataset: appendicitis", (full.Diagnosis == "appendicitis").sum()),
            ("balance", "full dataset: no appendicitis", (full.Diagnosis == "no appendicitis").sum()),
            ("balance", "cohort: appendicitis", d.y.sum()),
            ("balance", "cohort: no appendicitis", (1 - d.y).sum())]
    pd.DataFrame(rows, columns=["section", "item", "value"]).to_csv(RES / "data_summary.csv", index=False)
    feat = [c for c in full.columns if c not in ("Diagnosis",)]
    pd.DataFrame({"feature": feat, "pct_missing_all": full[feat].isna().mean().mul(100).round(1).values,
                  "pct_missing_cohort": d[feat].isna().mean().mul(100).round(1).values}
                 ).to_csv(RES / "data_missingness.csv", index=False)


def main():
    data.download()
    d, imgs = data.load_cohort()
    inspect(d, imgs)
    E = data.embed(d, imgs)
    s, y = data.split(d), d.y.values
    pd.DataFrame([(k, len(v), int(y[v].sum())) for k, v in s.items()],
                 columns=["split", "n_patients", "n_appendicitis"]).to_csv(RES / "splits.csv", index=False)
    yv, yt = y[s["val"]], y[s["test"]]

    out, ex = models.fit_all(d, E, s, y)
    raw = {m: p[1] for m, p in out.items()}
    Tm = {m: models.fit_temperature(yv, p[0]) for m, p in out.items()}
    temp = {m: models.apply_temperature(raw[m], Tm[m]) for m in out}
    allp = {**raw, **{m + "+temp": temp[m] for m in out}}
    boot = ev.bootstrap(yt, allp)

    rows = []
    for (m, k), (est, b) in boot.items():
        name, var = (m.split("+")[0], "temp") if m.endswith("+temp") else (m, "raw")
        rows.append((name, var, k, est, *ev.ci(b)))
    for a, b_ in (("fused_lr", "cbm"), ("fused_lr", "tab_lr")):  # paired differences, same resamples
        for k in ("AUROC", "AUPRC", "Sens@90Spec"):
            diff = boot[(a, k)][1] - boot[(b_, k)][1]
            rows.append((f"{a} - {b_}", "raw", k, boot[(a, k)][0] - boot[(b_, k)][0], *ev.ci(diff)))
    M = pd.DataFrame(rows, columns=["model", "variant", "metric", "estimate", "ci_lo", "ci_hi"])
    M.to_csv(RES / "metrics.csv", index=False)
    C = M[M.metric.isin(["ECE", "Brier"]) & M.model.isin(out)].pivot(
        index=["model", "metric"], columns="variant", values=["estimate", "ci_lo", "ci_hi"])
    C.columns = [f"{a}_{b}" for a, b in C.columns]
    C = C.reset_index(); C.insert(1, "T_val", C.model.map(Tm)); C.to_csv(RES / "calibration.csv", index=False)

    pd.DataFrame({"US_Number": d.US_Number.iloc[s["test"]].values, "y": yt,
                  **{f"p_{m}": raw[m] for m in out}, **{f"p_{m}_temp": temp[m] for m in out}}
                 ).to_csv(RES / "test_predictions.csv", index=False)
    cbm = ex["cbm"]
    Cm = data.concepts(d).to_numpy()[s["test"]]
    pd.DataFrame({"concept": cbm["names"], "coef": cbm["label_head"].coef_[0],
                  "n_test_observed": (~np.isnan(Cm)).sum(0),
                  "test_auroc": [roc_auc_score(Cm[~np.isnan(Cm[:, j]), j], cbm["heads"][j].predict_proba(
                      cbm["H"]["test"][~np.isnan(Cm[:, j])])[:, 1]) for j in range(Cm.shape[1])]}
                 ).to_csv(RES / "cbm_concepts.csv", index=False)
    pd.DataFrame([("tab_lr", "C", ex["tab_lr"].C), ("img_lr", "C", ex["img_lr"].C),
                  ("fused_lr", "C", ex["fused_lr"].C),
                  ("tab_gb", "n_estimators", ex["tab_gb"].n_estimators),
                  ("tab_gb", "max_depth", ex["tab_gb"].max_depth),
                  ("cbm", "label_head_C", cbm["label_head"].C), ("cbm", "concept_head_C", cbm["heads"][0].C)],
                 columns=["model", "param", "value"]).to_csv(RES / "hparams.csv", index=False)

    F = RES / "figures"
    ev.plot_roc(yt, raw, F / "roc_test.png")
    ev.plot_reliability(yt, raw, temp, F / "reliability.png")
    ev.plot_shap(ex["tab_gb"], ex["pre"], ex["T"]["test"], F / "shap_gb.png")
    ev.plot_concepts(cbm, raw["cbm"], F / "cbm_concepts.png")

    # sanity checks: disjoint splits; label-permutation null (train+val labels permuted, test labels real).
    # A single permuted run is NOT ~0.5: a model fit on noise projects randomly (random sign) onto the real
    # signal direction, so we report the distribution over N_PERM permutations.
    ids = {k: set(d.US_Number.iloc[v]) for k, v in s.items()}
    overlap = sum(len(ids[a] & ids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    tv = np.concatenate([s["train"], s["val"]])
    null = {m: [] for m in out}
    for seed in range(N_PERM):
        ys = y.copy(); ys[tv] = np.random.default_rng(seed).permutation(y[tv])
        for m, p in models.fit_all(d, E, s, ys)[0].items():
            null[m].append(roc_auc_score(yt, p[1]))
    S = [(f"permuted_labels:{m}", "test AUROC", np.mean(v), np.std(v), min(v), max(v)) for m, v in null.items()]
    S += [("patients_in_more_than_one_split", "count", overlap, np.nan, np.nan, np.nan)]
    pd.DataFrame(S, columns=["check", "metric", "mean", "sd", "min", "max"]).to_csv(RES / "sanity.csv", index=False)
    assert overlap == 0

    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(M.round(3).to_string(index=False)); print(pd.read_csv(RES / "sanity.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
