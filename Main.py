import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Load dataset
    df = pd.read_csv("career_recommendation_dataset.csv")
    print("Data loaded. Shape:",df.shape)

    # Features (X) and Target (y)
    X = df.drop("career", axis = 1)
    y = df["career"]

    # Identify numeric & categorical columns
    numeric_features = [
                        "programming_skill",
                        "math_skill",
                        "communication_skill",
                        "logical_thinking",
                        "cgpa",
                        "projects_count",
                        "interest_coding",
                        "interest_data",
                        "interest_appdev",
                        "interest_security",
                        ]
    categorical_features = ["favorite_subject"]

    # Preprocessor: OneHotEncode categorical, pass numeric as-is
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough"
    )

    # Model: Random Forest
    model = RandomForestClassifier(n_estimators=150)

    # Create Pipeline: preprocessing + model
    crs = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # A Pipeline is like connecting multiple steps together so they work automatically in order.

    # It combines:
    #      Preprocessing(convert text to numbers)
    #      Model training(Random Forest)
    #
    # into ONE single object → clf

    # Train–Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # stratify = y means:
    #
    # 👉 Split the data into train and test in such a way that the proportion of each career
    #  class is preserved.


    print("\nData split done:")
    print("Train size:", X_train.shape[0])
    print("Test size :", X_test.shape[0])

    # Train the model
    crs.fit(X_train, y_train)
    print("\nModel training completed.")

    import joblib
    joblib.dump(crs, "career_model.pkl")
    print("Model saved!")

    # Evaluate
    y_pred = crs.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on test data: {acc:.2f}")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
