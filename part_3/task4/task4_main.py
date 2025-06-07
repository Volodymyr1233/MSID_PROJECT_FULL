import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from part_2.implemented_models.LogisticRegrGradient import LogisticRegrGradient
from part_2.pipeline import call_preprocess
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report

df = pd.read_csv("../../depr_dataset.csv")
df['Financial Stress'] = df['Financial Stress'].replace('?', np.nan)


target_column = "Depression"
drop_columns = ['id', "CGPA", "Depression"]

x = df.drop(columns=drop_columns)
y = df[target_column].astype(float).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.45, random_state=42)

preprocess = call_preprocess(['CGPA', "Depression"])

x_train_transform = preprocess.fit_transform(x_train)
x_test_transform = preprocess.transform(x_test)

# Trening na oryginalnych danych bez jeszcze zbalansowania

model_or = LogisticRegrGradient(lr=0.1, epochs=100, batch_size=64)
model_or.fit(x_train_transform.toarray(), y_train, x_test_transform.toarray(), y_test)

y_pred_or, _ = model_or.predict(x_test_transform.toarray())

print("=== Wyniki oryginalne dane ===")
print(classification_report(y_test, y_pred_or, digits=4))


smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train_transform, y_train)

# Trening na OVERSAMPLING danych
model_smote = LogisticRegrGradient(lr=0.1, epochs=100, batch_size=64)
model_smote.fit(x_train_smote.toarray(), y_train_smote, x_test_transform.toarray(), y_test)

y_pred_smote, _ = model_smote.predict(x_test_transform.toarray())
print("=== Wyniki z oversampling ===")
print(classification_report(y_test, y_pred_smote, digits=4))

under_samp = RandomUnderSampler(random_state=42)
x_train_under_samp, y_train_under_samp = under_samp.fit_resample(x_train_transform, y_train)

# Trening na UNDERSAMPLING danych
model_under_samp = LogisticRegrGradient(lr=0.1, epochs=100, batch_size=64)
model_under_samp.fit(x_train_under_samp.toarray(), y_train_under_samp, x_test_transform.toarray(), y_test)

y_pred_rus, _ = model_under_samp.predict(x_test_transform.toarray())
print("=== Wyniki z undersampling ===")
print(classification_report(y_test, y_pred_rus, digits=4))