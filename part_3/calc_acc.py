import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from  part_2.implemented_models.LogisticRegrGradient import LogisticRegrGradient
from part_2.pipeline import call_preprocess
from sklearn.metrics import accuracy_score
from part_2.utils import make_table

df = pd.read_csv("../depr_dataset.csv")
df['Financial Stress'] = df['Financial Stress'].replace('?', np.nan)

target_column = "Depression"
drop_columns = ["id", "CGPA", "Depression"]

x = df.drop(columns=drop_columns)
y = df[target_column].astype(float).values

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)

x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

preprocess = call_preprocess(["CGPA", "Depression"])

x_train_transform = preprocess.fit_transform(x_train)
x_val_transform = preprocess.transform(x_val)
x_test_transform = preprocess.transform(x_test)

# poly = PolynomialFeatures(degree=2, include_bias=False)
# x_train_poly = poly.fit_transform(x_train_transform.toarray())
# x_val_poly = poly.transform(x_val_transform.toarray())
# x_test_poly = poly.transform(x_test_transform.toarray())

model = LogisticRegrGradient(lr=0.005, epochs=60, batch_size=64, l2_lam=0.001)

model.fit(x_train_transform.toarray(), y_train, x_test_transform.toarray(), y_test)

y_pred_train, y_proba_train = model.predict(x_train_transform.toarray())
y_pred_valid, y_proba_valid = model.predict(x_val_transform.toarray())
y_pred_test, y_proba_test = model.predict(x_test_transform.toarray())

logreg_train_acc = round(accuracy_score(y_train, y_pred_train), 4)
logreg_train_loss = round(model.compute_cross_entropy(y_train, y_proba_train), 4)

logreg_valid_acc = round(accuracy_score(y_val, y_pred_valid), 4)
logreg_valid_loss = round(model.compute_cross_entropy(y_val, y_proba_valid), 4)

logreg_test_acc = round(accuracy_score(y_test, y_pred_test), 4)
logreg_test_loss = round(model.compute_cross_entropy(y_test, y_proba_test), 4)


make_table(
    "Regresja logistyczna z regularyzacją L2",
    "Accuracy", "CE",
    logreg_train_acc, logreg_train_loss,
    logreg_valid_acc, logreg_valid_loss,
    logreg_test_acc, logreg_test_loss
)

