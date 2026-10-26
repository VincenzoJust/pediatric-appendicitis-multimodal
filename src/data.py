"""Data download, cohort, patient-level split, preprocessing, frozen image embeddings."""
import hashlib, re, zipfile
from pathlib import Path

import numpy as np, pandas as pd, requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 0
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://zenodo.org/api/records/7711412/files/{}/content"  # v1.03, CC BY-NC 4.0
MD5 = {"app_data.xlsx": "d17a803f5e27532e518676a38f588b59",
       "US_Pictures.zip": "38b81fd3dddaeed458132be2ec1732ed"}

# Columns never used as features: targets/outcomes, post-hoc fields, identifiers, free text.
LEAK = ["Diagnosis", "Diagnosis_Presumptive", "Management", "Severity", "Length_of_Stay",
        "US_Number", "US_Performed", "y"]
TEXT = ["Abscess_Location", "Lymph_Nodes_Location", "Gynecological_Findings"]
# Inputs of the concept heads: demographics + labs only (no scores, no exam/US findings).
CBM_TAB = ["Age", "Sex", "BMI", "Height", "Weight", "Body_Temperature", "WBC_Count",
           "Neutrophil_Percentage", "RBC_Count", "Hemoglobin", "RDW", "Thrombocyte_Count", "CRP",
           "Ketones_in_Urine", "RBC_in_Urine", "WBC_in_Urine", "Dysuria", "Stool"]


def download():
    DATA.mkdir(exist_ok=True)
    for name, md5 in MD5.items():
        f = DATA / name
        if not f.exists():
            with requests.get(URL.format(name), stream=True, timeout=60) as r, open(f, "wb") as o:
                r.raise_for_status()
                for chunk in r.iter_content(1 << 20):
                    o.write(chunk)
        assert hashlib.md5(f.read_bytes()).hexdigest() == md5, f"md5 mismatch: {name}"
    if not (DATA / "raw").exists():
        zipfile.ZipFile(DATA / "US_Pictures.zip").extractall(DATA / "raw")


def image_index():
    """subject number -> list of image paths (files are named '<subject>.<view> <label>.<ext>')."""
    idx = {}
    for p in sorted((DATA / "raw" / "US_Pictures").iterdir()):
        m = re.match(r"(\d+)\.(\d+)", p.name)
        if m:
            idx.setdefault(int(m.group(1)), []).append(p)
    return idx


def load_all():
    return pd.read_excel(DATA / "app_data.xlsx", sheet_name="All cases")


def load_cohort():
    """Patients with a diagnosis, a US_Number and >=1 image. One row per patient."""
    imgs = image_index()
    d = load_all().dropna(subset=["Diagnosis", "US_Number"])
    d = d.assign(US_Number=d.US_Number.astype(int))
    assert not d.US_Number.duplicated().any()
    d = d[d.US_Number.isin(imgs)].reset_index(drop=True)
    d["y"] = (d.Diagnosis == "appendicitis").astype(int)
    return d, imgs


def split(d):
    """Stratified 60/20/20 patient-level split (rows are patients) + disjointness checks."""
    i = np.arange(len(d))
    tr, te = train_test_split(i, test_size=0.2, stratify=d.y, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.25, stratify=d.y.iloc[tr], random_state=SEED)
    s = {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}
    ids = {k: set(d.US_Number.iloc[v]) for k, v in s.items()}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
    assert sum(map(len, ids.values())) == len(d)
    return s


def tab_features(d):
    return d.drop(columns=LEAK + TEXT)


def preproc(X):
    """Impute + scale/one-hot. Caller must .fit() on the training rows only."""
    num = X.select_dtypes("number").columns
    cat = X.columns.difference(num)
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat)])


def concepts(d):
    """Binary clinical / US-finding concepts (NaN = not recorded). Chosen by coverage >= 60% in the cohort."""
    yn = lambda c: d[c].map({"yes": 1.0, "no": 0.0})
    c = pd.DataFrame({k: yn(k) for k in ["Appendix_on_US", "Free_Fluids", "Lower_Right_Abd_Pain",
                                         "Migratory_Pain", "Contralateral_Rebound_Tenderness",
                                         "Psoas_Sign", "Coughing_Pain", "Nausea", "Loss_of_Appetite"]})
    c["Appendix_Diameter_gt6mm"] = (d.Appendix_Diameter > 6).astype(float).where(d.Appendix_Diameter.notna())
    c["Peritonitis"] = d.Peritonitis.map({"no": 0.0, "local": 1.0, "generalized": 1.0})
    return c


def embed(d, imgs):
    """Frozen ImageNet ResNet-50 (2048-d), mean over each patient's images. Cached per image file."""
    import torch, torchvision as tv
    from PIL import Image
    cache = DATA / "emb_resnet50_imagenet_v2.npz"
    files = [str(p) for n in d.US_Number for p in imgs[n]]
    if cache.exists():
        z = np.load(cache)
        emb = dict(zip(z["files"], z["emb"]))
    else:
        torch.manual_seed(SEED)
        w = tv.models.ResNet50_Weights.IMAGENET1K_V2
        net = tv.models.resnet50(weights=w)
        net.fc = torch.nn.Identity()
        net.eval()
        tf = tv.transforms.Compose([tv.transforms.Resize((224, 224)), tv.transforms.ToTensor(),
                                    tv.transforms.Normalize(w.transforms().mean, w.transforms().std)])
        out = []
        with torch.no_grad():
            for i in range(0, len(files), 32):
                x = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[i:i + 32]])
                out.append(net(x).numpy())
        emb = dict(zip(files, np.concatenate(out)))
        np.savez(cache, files=np.array(files), emb=np.stack([emb[f] for f in files]))
    return np.stack([np.mean([emb[str(p)] for p in imgs[n]], axis=0) for n in d.US_Number])
