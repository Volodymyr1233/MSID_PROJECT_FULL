import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from part_2.implemented_models.LogisticRegrGradient import LogisticRegrGradient
from part_2.pipeline import call_preprocess

df = pd.read_csv("../../depr_dataset.csv")
df['Financial Stress'] = df['Financial Stress'].replace('?', np.nan)



target_column = "Depression"
drop_columns = ["id", "Depression"]

x = df.drop(columns=drop_columns)
y = df[target_column].astype(float).values

preprocess = call_preprocess(["Depression"])

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

fold_results = []

for fold, (train_index, test_index) in enumerate(skf.split(x, y), 1):
    x_train_raw, x_test_raw = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    x_train = preprocess.fit_transform(x_train_raw)
    x_test = preprocess.transform(x_test_raw)

    model = LogisticRegrGradient(lr=0.005, epochs=200, batch_size=64)
    model.fit(x_train.toarray(), y_train)

    y_pred_class, y_pred_prob = model.predict(x_test.toarray())

    acc = accuracy_score(y_test, y_pred_class)
    loss = model.compute_cross_entropy(y_test, y_pred_prob)

    print(f"Krok walidacji: {fold}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cross-Entropy Loss: {loss:.4f}")

    fold_results.append((acc, loss))

accs, losses = zip(*fold_results)
print("\nPodsumowanie wyników cross-walidacji:")
print(f"Średnia Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Średni Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")