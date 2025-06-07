import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from  part_2.implemented_models.LogisticRegrGradient import LogisticRegrGradient
from part_2.pipeline import call_preprocess
from part_3.task2.task2_main import plot_losses
from collections import defaultdict

df = pd.read_csv("../../depr_dataset.csv")
df['Financial Stress'] = df['Financial Stress'].replace('?', np.nan)


target_column = "Depression"
drop_columns = ['id', "CGPA", "Depression"]

x = df.drop(columns=drop_columns)
y = df[target_column].astype(float).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

preprocess = call_preprocess(['CGPA', "Depression"])

x_train_transform = preprocess.fit_transform(x_train)
x_test_transform = preprocess.transform(x_test)

feature_names = preprocess.get_feature_names_out()


model = LogisticRegrGradient(lr=0.05, epochs=60, batch_size=32)

model.fit(x_train_transform.toarray(), y_train, x_test_transform.toarray(), y_test)


model_l1 = LogisticRegrGradient(lr=0.05, epochs=60, batch_size=32, l1_lam=0.01)

model_l1.fit(x_train_transform.toarray(), y_train, x_test_transform.toarray(), y_test)


model_l2 = LogisticRegrGradient(lr=0.05, epochs=60, batch_size=32, l2_lam=0.01)

model_l2.fit(x_train_transform.toarray(), y_train, x_test_transform.toarray(), y_test)


plot_losses(model, "Model bez regularyzacji")
plot_losses(model_l1, "Model z regularyzacją L1")
plot_losses(model_l2, "Model z regularyzacją L2")


def group_weights_by_feature(model, feature_names):
    weights = model.wages[1:]  # pomijamy bias (wages[0])
    grouped = defaultdict(list)

    for name, weight in zip(feature_names, weights):
        if name.startswith("num__"):
            base = name.replace("num__", "")
        elif name.startswith("cat__"):
            base = name.replace("cat__", "").split("_")[0]
        else:
            base = name
        grouped[base].append(weight)

    result = {feature: np.mean(wags) for feature, wags in grouped.items()}
    return pd.DataFrame.from_dict(result, orient="index", columns=["Avg Weight"]).sort_values(by="Avg Weight", ascending=False)


print("\n--- Uśrednione wagi: model bez regularyzacji ---")
print(group_weights_by_feature(model, feature_names))

print("\n--- Uśrednione wagi: model z L1 ---")
print(group_weights_by_feature(model_l1, feature_names))

print("\n--- Uśrednione wagi: model z L2 ---")
print(group_weights_by_feature(model_l2, feature_names))