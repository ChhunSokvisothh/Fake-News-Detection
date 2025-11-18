import joblib

model = joblib.load("model/decision_tree_model.pkl")
print("Loaded successfully!")

# Check which features it expects
if hasattr(model, "feature_names_in_"):
    print("Expected input features:", model.feature_names_in_)
else:
    print("No feature names found. Model expects same number of features as in training.")
    print("Number of features expected:", getattr(model, "n_features_in_", "Unknown"))