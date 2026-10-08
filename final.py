# ============================================================
# Freight Rate Prediction Challenge
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "month"
]

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment"
]



# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(file_path):
    """Load the dataset."""

    df = pd.read_csv(file_path)

    print(f"Data loaded successfully from {file_path}")
    print(f"Shape: {df.shape}")

    return df


# ============================================================
# 2. CLEAN DATA
# ============================================================

def clean_data(df):
    """Clean the dataset."""

    # Count negative weights
    negative_weights = (df["weight"] < 0).sum()
    print("Number of negative weights:", negative_weights)

    # Convert negative weights to missing values NaN
    df["weight"] = df["weight"].where(df["weight"] >= 0, np.nan)

    # Fill missing numeric values with median
    df["weight"] = df["weight"].fillna(df["weight"].median())
    df["market_index"] = df["market_index"].fillna(
        df["market_index"].median()
    )

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    print("\nMissing values:")
    print(df.isnull().sum())

    return df

import os
# ============================================================
# 3. VISUALIZATION
# ============================================================
def visualization(df):


    os.makedirs("output_png", exist_ok=True)
    # --------------------------------------------------------
    # Save Plot
    # --------------------------------------------------------

    def save_plot(filename, title):
     plt.title(title)
     plt.tight_layout()
     plt.savefig(f"output_png/{filename}.png")
     plt.close()

    # --------------------------------------------------------
    # 1. Histogram - Posted Rate
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["posted_rate"],
        bins=50,
        edgecolor="black",
        alpha=0.5
    )

    plt.xlabel("Posted Rate")
    plt.ylabel("Frequency")

    save_plot(
        "posted_rate_histogram",
        "Distribution of Posted Rate"
    )

    # --------------------------------------------------------
    # 2. Histogram - Distance
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["distance"],
        bins=50,
        edgecolor="black",
        alpha=0.5
    )

    plt.xlabel("Distance")
    plt.ylabel("Frequency")

    save_plot(
        "distance_histogram",
        "Distribution of Distance"
    )

    # --------------------------------------------------------
    # 3. Scatter - Distance vs Posted Rate
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["distance"],
        df["posted_rate"],
        alpha=0.5
    )

    plt.xlabel("Distance")
    plt.ylabel("Posted Rate")

    save_plot(
        "distance_vs_posted_rate",
        "Distance vs Posted Rate"
    )

    # --------------------------------------------------------
    # 4. Scatter - Weight vs Posted Rate
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["weight"],
        df["posted_rate"],
        alpha=0.5
    )

    plt.xlabel("Weight")
    plt.ylabel("Posted Rate")

    save_plot(
        "weight_vs_posted_rate",
        "Weight vs Posted Rate"
    )

    # --------------------------------------------------------
    # 5. Scatter - Market Index vs Posted Rate
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["market_index"],
        df["posted_rate"],
        alpha=0.5
    )

    plt.xlabel("Market Index")
    plt.ylabel("Posted Rate")

    save_plot(
        "market_index_vs_posted_rate",
        "Market Index vs Posted Rate"
    )

    # --------------------------------------------------------
    # 6. Scatter - Quote Signal vs Posted Rate
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["quote_signal"],
        df["posted_rate"],
        alpha=0.5
    )

    plt.xlabel("Quote Signal")
    plt.ylabel("Posted Rate")

    save_plot(
        "quote_signal_vs_posted_rate",
        "Quote Signal vs Posted Rate"
    )

    # --------------------------------------------------------
    # 7. Scatter - Month vs Posted Rate
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["month"],
        df["posted_rate"],
        alpha=0.5
    )

    plt.xlabel("Month")
    plt.ylabel("Posted Rate")

    plt.xticks(range(1, 13))

    save_plot(
        "month_vs_posted_rate",
        "Month vs Posted Rate"
    )

    # --------------------------------------------------------
    # 8. Boxplot - Equipment vs Posted Rate
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="equipment",
        y="posted_rate"
    )

    plt.xlabel("Equipment")
    plt.ylabel("Posted Rate")

    save_plot(
        "equipment_vs_posted_rate",
        "Posted Rate by Equipment"
    )

    # --------------------------------------------------------
    # 9. Boxplot - Month vs Posted Rate
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="month",
        y="posted_rate"
    )

    plt.xlabel("Month")
    plt.ylabel("Posted Rate")

    save_plot(
        "month_boxplot",
        "Posted Rate by Month"
    )

    # --------------------------------------------------------
    # 10. Market Index and Quote Signal by Month
    # --------------------------------------------------------

    for col in [
        "market_index",
        "quote_signal"
    ]:

        plt.figure(figsize=(10, 6))

        sns.boxplot(
            data=df,
            x="month",
            y=col
        )

        plt.xlabel("Month")
        plt.ylabel(col)

        save_plot(
         f"{col}_by_month",
         "By Month"
)

    # --------------------------------------------------------
    # 11. Correlation Heatmap
    # --------------------------------------------------------

    numeric_cols = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
        "month",
        "posted_rate"
    ]

    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True
    )

    save_plot(
        "correlation_heatmap",
        "Correlation Heatmap"
    )

    print(
        "\nVisualizations saved successfully in output_png/"
    )

# ============================================================
# 3. PREPARE DATA
# ============================================================

def prepare_data(df):
    """Separate features and target."""

    df = df.drop(columns=["load_id"])

    y = df["posted_rate"]
    X = df.drop(columns=["posted_rate"])

    return X, y

######################################
def prepare_validation_data(df):

    return df.drop(columns=["load_id"])


# ============================================================
# 4. SPLIT DATA
# ============================================================

def split_data(X, y):
    """Split data into training and validation sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nData split:")
    print(f"Training set:   {len(X_train)}")
    print(f"Test set: {len(X_test)}")


    return X_train, X_test, y_train, y_test


# ============================================================
# 5. PREPROCESSOR
# ============================================================

def build_preprocessor():
    """Build preprocessing pipeline."""

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


# ============================================================
# 6. BUILD MODELS
# ============================================================

def build_models():
    """Build all models."""

    preprocessor = build_preprocessor()

    model = {
      
        "XGBoost": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror"
            ))
        ])
    }

    return model


# ============================================================
# 7. TRAIN AND EVALUATE
# ============================================================

def train_and_evaluate(
    models,
    X_train,
    y_train,
    X_test,
    y_test
):
    """Train and evaluate all models."""

    results = []

    for name, model in models.items():

        print(f"\nTraining: {name}")

        # Cross-validation
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="neg_mean_absolute_error"
        )

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

     
    return pd.DataFrame(results)

# ============================================================
# 8. MAIN
# ============================================================

def main():

    file_path = "data/train-test.csv"
    validation_path = "data/validation.csv"

    # Load
    df = load_data(file_path)
    validation_df = load_data(validation_path)

    # Clean
    df = clean_data(df)
    validation_df = clean_data(validation_df)

    # Visualization
    visualization(df)

    # Prepare
    X, y = prepare_data(df)
    X_validation = prepare_validation_data(validation_df)

    print("\nTraining features:")
    print(X.columns.tolist())

    print("\nValidation features:")
    print(X_validation.columns.tolist())

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Build models
    model = build_models()

    # Train and evaluate
    results = train_and_evaluate(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )


    # Use XGBoost
    xgb_model = model["XGBoost"]

    # Validation predictions
    predicted_rates = xgb_model.predict(X_validation)

    final_predictions_df = pd.DataFrame({
        "load_id": validation_df["load_id"],
        "predicted_rate": predicted_rates
    })

    final_predictions_df.to_csv(
        "data/validation_predictions.csv",
        index=False
    )

    print("\nValidation prediction file created successfully!")
    print(final_predictions_df.head())
    print("Shape:", final_predictions_df.shape)
    print("Missing values:")
    print(final_predictions_df.isnull().sum())
    print("****************************************************8")
    print(results)

    return model, results, df


# ============================================================
# December Predictions
# ============================================================

def predict_december(xgb_model, train_df):
    december_path = "data/december_chart_inputs.csv"
    december_df = pd.read_csv(december_path)

    # Create model input
    X_december = december_df.copy()

    # Add features 
    X_december["pickup_lat"] = 36.99152
    X_december["pickup_lon"] = -84.99876
    X_december["delivery_lat"] = 41.31561
    X_december["delivery_lon"] = -85.36206

    dec_df = train_df[
        (train_df["pickup"] == "Lexington") &
        (train_df["delivery"] == "Fort Wayne") &
        (train_df["equipment"] == "Dry Van")
    ]

    # get median values for market_index and quote_signal from the filtered training data
    X_december["market_index"] = dec_df["market_index"].median()
    X_december["quote_signal"] = dec_df["quote_signal"].median()

    # December
    X_december["month"] = 12

    # Remove unused columns
    X_december = X_december.drop(
        columns=["predicted_rate", "date"]
    )

    # Predict
    december_predictions = xgb_model.predict(X_december)

    december_df["predicted_rate"] = december_predictions

    december_df.to_csv(december_path, index=False)

    print("\nDecember predictions saved successfully!")
    print(
        december_df[
            ["date", "predicted_rate"]
        ].to_string(index=False)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    model, results, train_df = main()

    predict_december(
        model["XGBoost"],
        train_df
    )
