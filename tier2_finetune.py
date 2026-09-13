import pandas as pd, numpy as np, torch, json, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import os
torch.manual_seed(42)
np.random.seed(42)

df = pd.read_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier2_dataset.jsonl"), lines=True)
print("Loaded tier2 dataset:", df.shape)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
print("Train:", len(train_df), "Test:", len(test_df))

MODEL_NAME = "microsoft/codebert-base"
print("Loading tokenizer/model:", MODEL_NAME)
t0 = time.time()
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
print(f"Loaded in {time.time()-t0:.1f}s")

MAX_LEN = 256

class SmellDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = SmellDataset(train_df["text"], train_df["label"])
test_ds = SmellDataset(test_df["text"], test_df["label"])

BATCH = 8
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

device = torch.device("cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
EPOCHS = 3
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

print(f"Starting fine-tuning: {EPOCHS} epochs, {len(train_loader)} batches/epoch on CPU...")
model.train()
for epoch in range(EPOCHS):
    t_ep = time.time()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(**batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if (step + 1) % 5 == 0:
            print(f"  epoch {epoch+1} step {step+1}/{len(train_loader)} loss={loss.item():.4f} elapsed={time.time()-t_ep:.0f}s")
    print(f"Epoch {epoch+1} done. avg_loss={total_loss/len(train_loader):.4f} time={time.time()-t_ep:.0f}s")

print("Evaluating...")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = torch.argmax(out.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
acc = accuracy_score(all_labels, all_preds)
print(f"\n=== Tier 2 (true end-to-end fine-tuned CodeBERT) test results ===")
print(f"n_test={len(all_labels)} Accuracy={acc:.3f} Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")


results = {"n_train": len(train_df), "n_test": len(test_df), "accuracy": acc, "precision": p, "recall": r, "f1": f1, "epochs": EPOCHS}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier2_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("Saved tier2_results.json")
