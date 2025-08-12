import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eda import remove_outliers_iqr
from Data.Preparation import Preparation

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "voitures_test.csv")

df = pd.read_csv(file_path)
df = df.dropna(axis=0)
print(df)

y = df["Prix"]
X = df.drop("Prix", axis=1)

pkl_path = os.path.join(current_dir, "..", 'preparation_instance.pkl')


preparator = joblib.load(pkl_path)
       

y = np.log1p(y.astype(float))

scaled_prix = preparator.standerizers['Prix standarizer'].transform(y.values.reshape(-1, 1))

y = pd.DataFrame(scaled_prix.flatten(), columns=["Prix"])

scaled_year = preparator.standerizers['year standarizer'].transform(X[["Année-Modèle"]])

scaled_kilo = preparator.standerizers["Kilométrage standarizer"].transform(X[["Kilométrage"]])
scaled_portes = preparator.standerizers["portes standarizer"].transform(X[["Nombre de portes"]])
scaled_pf = preparator.standerizers["Puissance fiscale scaler"].transform(X[["Puissance fiscale"]])
scaled_etat = preparator.getEndoder().standrizers['etat scaler'].transform(X[["État"]])


X.loc[:, "Année-Modèle"] = scaled_year.flatten()
X.loc[:, "Kilométrage"] = scaled_kilo.flatten()
X.loc[:, "Nombre de portes"] = scaled_portes.flatten()
X.loc[:, "Puissance fiscale"] = scaled_pf.flatten().astype(int)
X.loc[:, "État"] = scaled_etat.flatten()

colonnes_cibles = [
 "Année-Modèle", "Kilométrage","Prix",
    "Nombre de portes", "Puissance fiscale", "État"
]


X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
test_df = pd.concat([X, y], axis=1)
test_df , counter = remove_outliers_iqr(test_df,colonnes_cibles)
print(counter)

train_df = preparator.get_traib_data()

x_train = train_df.drop("Prix", axis=1)
y_train = train_df["Prix"]

x_test = test_df.drop("Prix", axis=1)
y_test = test_df["Prix"]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Support Vector Regressor": SVR()
}



last_r2 = -float("inf")  
best_model = None
best_model_name = ""
trained_models = {}

print(x_train.columns.tolist())
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    if r2 > last_r2:
        best_model = model
        best_model_name = name
        last_r2 = r2

    trained_models[name] = model

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("True Prix")
    plt.ylabel("Predicted Prix")
    plt.title(f"{name} - Predicted vs True")
    plt.grid(True)
    plt.show()

    scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R²: {scores.mean():.3f}")

print(f"\nBest model is: {best_model_name} with R² Score: {last_r2:.2f}")
joblib.dump(best_model, f"{best_model_name.replace(' ', '_').lower()}_model.pkl")
