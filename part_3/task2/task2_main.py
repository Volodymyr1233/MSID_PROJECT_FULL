import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from  part_2.implemented_models.LogisticRegrGradient import LogisticRegrGradient
from part_2.pipeline import call_preprocess
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("../../depr_dataset.csv")
df['Financial Stress'] = df['Financial Stress'].replace('?', np.nan)

target_column = "Depression"
drop_columns = ['id', 'CGPA', "Depression"]
# drop_columns = ['id', 'Gender', 'Age', 'Profession', 'Work/Study Hours',
#        'CGPA','Dietary Habits', 'Degree', 'Financial Stress',
#        'Have you ever had suicidal thoughts ?',
#          'Family History of Mental Illness', 'Depression']

x = df.drop(columns=drop_columns)
y = df[target_column].astype(float).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# preprocess = call_preprocess(['Gender', 'Age', 'Profession', 'Work/Study Hours',
#        'CGPA','Dietary Habits', 'Degree', 'Financial Stress',
#        'Have you ever had suicidal thoughts ?',
#          'Family History of Mental Illness', 'Depression'])

preprocess = call_preprocess(['CGPA', "Depression"])

x_train_transform = preprocess.fit_transform(x_train)
x_test_transform = preprocess.transform(x_test)

poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train_transform.toarray())
x_test_poly = poly.transform(x_test_transform.toarray())

def plot_losses(model, title="Zbieżność funkcji kosztu"):
    plt.plot(model.train_loss, label="Train Loss")
    if model.val_loss:
        plt.plot(model.val_loss, label="Validation Loss")
    plt.xlabel("Epoka")
    plt.ylabel("Funkcja kosztu (Cross-Entropy)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

model = LogisticRegrGradient(lr=0.1, epochs=60, batch_size=16)

model.fit(x_train_transform.toarray(), y_train, x_test_transform.toarray(), y_test)

plot_losses(model)