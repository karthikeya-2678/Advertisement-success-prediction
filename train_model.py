import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# load dataset
df = pd.read_csv("data/train.csv")

# drop unwanted columns
df = df.drop(["UserID"], axis=1)

# Features: Drop all target columns
X = df.drop(["netgain", "ratings", "money_back_guarantee"], axis=1)

# Targets
y_ratings = df["ratings"]
y_success = df["netgain"]
y_money = df["money_back_guarantee"]

# define categorical columns
categorical_cols = [col for col in X.columns if X[col].dtype == "object"]

# create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ],
    remainder="passthrough"
)

# create pipelines
rating_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

success_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, class_weight='balanced'))
])

money_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# train models
rating_model.fit(X, y_ratings)
success_model.fit(X, y_success)
money_model.fit(X, y_money)

# save models in a dictionary
models = {
    'rating_model': rating_model,
    'success_model': success_model,
    'money_model': money_model
}

with open("model/model.pkl", "wb") as f:
    pickle.dump(models, f)

print("Models trained and saved")