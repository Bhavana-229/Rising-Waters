import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================
# Load Dataset
# ==========================
data = pd.read_csv("dataset/flood.csv")

# Features and Target
X = data.drop("FloodProbability", axis=1)
y = data["FloodProbability"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Models
# ==========================
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

results = []

print("\n========== MODEL COMPARISON ==========\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, r2])

    print(f"{name}")
    print(f"MAE : {mae:.6f}")
    print(f"R2 Score : {r2:.6f}")
    print("-" * 35)

# ==========================
# Best Model
# ==========================
best_model = max(results, key=lambda x: x[2])

print("\n========== BEST MODEL ==========")
print("Model :", best_model[0])
print("R2 Score :", round(best_model[2], 6))