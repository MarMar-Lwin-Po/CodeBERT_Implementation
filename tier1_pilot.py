import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
import json
from pathlib import Path

np.random.seed(42)

DATA_PATH = "https://raw.githubusercontent.com/iSMELL2024/iSMELL/main/CodeDetection/updated_dataset.xlsx"
df = pd.read_excel(DATA_PATH)
print("Loaded:", df.shape)

# Parse the precomputed 768-dim CodeBERT embedding column
def parse_vec(s):
    return np.array([float(x) for x in str(s).split(",")], dtype=np.float32)

X = np.stack(df["codebert_vector"].apply(parse_vec).values)
y = df["Label"].values.astype(int)
smell = df["Smell"].to_numpy()
print("X shape:", X.shape, "y balance:", np.bincount(y))

# Stratify by Smell x Label jointly so each smell type is represented proportionally in test
strat_key = df["Smell"].astype(str) + "_" + df["Label"].astype(str)
X_train, X_test, y_train, y_test, smell_train, smell_test = train_test_split(
    X, y, smell, test_size=0.2, random_state=42, stratify=strat_key
)
print("Train:", X_train.shape, "Test:", X_test.shape)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

for name, clf in [
    ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ("MLP (128,64)", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)),
]:
    clf.fit(X_train_s, y_train)
    pred = clf.predict(X_test_s)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    acc = accuracy_score(y_test, pred)
    results[name] = {"precision": p, "recall": r, "f1": f1, "accuracy": acc}
    print(f"\n=== {name} (overall, binary smell/no-smell) ===")
    print(f"Accuracy={acc:.3f} Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

    # Per-smell breakdown
    print("Per-smell-type breakdown:")
    for s in np.unique(smell_test):
        mask = smell_test == s
        if mask.sum() == 0:
            continue
        p_s, r_s, f1_s, _ = precision_recall_fscore_support(
            y_test[mask], pred[mask], average="binary", zero_division=0
        )
        acc_s = accuracy_score(y_test[mask], pred[mask])
        print(f"  {s:16s} n={mask.sum():3d}  Acc={acc_s:.3f} P={p_s:.3f} R={r_s:.3f} F1={f1_s:.3f}")
        results[name][s] = {"n": int(mask.sum()), "accuracy": acc_s, "precision": p_s, "recall": r_s, "f1": f1_s}

with open(Path(__file__).with_name("tier1_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved results to tier1_results.json")
