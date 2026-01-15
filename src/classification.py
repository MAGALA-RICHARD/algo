from pathlib import Path
import pandas as pd

import numpy as np
from scipy import stats

data_dir = Path(__file__).parent.parent / 'data'
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv(data_dir / "data.csv")

# Optional: clean column names (safer)
df.columns = df.columns.str.strip()

# --------------------------------------------------
# 2. Define BEHAVIORAL solutions (decision-based)
#    Adjust thresholds if needed
# --------------------------------------------------
df["behavioral"] = (
        (df["R2"] > 0.95) &
        (df["RMSE"] < 200) &
        (df["RRMSE"] < 0.1) &
        (df["MAE"] < 200) &
        (df["WIA"] > 0.99) &
        (df["BIAS"].abs() < 50)
).astype(int)

print("Behavioral class balance:")
print(df["behavioral"].value_counts(normalize=True))
print()

# --------------------------------------------------
# 3. Define features (sources of iquefinality)
# --------------------------------------------------
features = [
    "algorithm",
    'method'

]

X = df[features]
y = df["behavioral"]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# --------------------------------------------------
# 4. Classification model (interpretable)
# --------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced"
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("Overall ROC-AUC scores:", auc_scores)
print("Mean ROC-AUC:", auc_scores.mean())
print()

# --------------------------------------------------
# 5. Algorithm-specific equifinality
# --------------------------------------------------
print("Algorithm-conditioned equifinality (ROC-AUC):")

alg_results = {}

for alg in df["algorithm"].unique():
    sub = df[df["algorithm"] == alg]

    # Skip if only one class
    if sub["behavioral"].nunique() < 2:
        continue

    X_sub = pd.get_dummies(sub[features], drop_first=True)
    y_sub = sub["behavioral"]

    scores = cross_val_score(
        pipeline,
        X_sub,
        y_sub,
        cv=3,
        scoring="roc_auc"
    )

    alg_results[alg] = scores.mean()
    print(f"  {alg:12s} -> ROC-AUC = {scores.mean():.3f}")

print()

# --------------------------------------------------
# 6. Feature importance (where equifinality lives)
# --------------------------------------------------
pipeline.fit(X, y)

coef = pipeline.named_steps["clf"].coef_[0]
importance = (
    pd.Series(coef, index=X.columns)
    .sort_values()
)

print("Top negative coefficients (discouraging behavioral solutions):")
print(importance.head(10))
print()

print("Top positive coefficients (encouraging behavioral solutions):")
print(importance.tail(10))
print()


def mean_ci(series, confidence=0.95):
    series = series.dropna()
    n = series.size
    mean = series.mean()
    std = series.std(ddof=1)

    if n < 2:
        return pd.Series({
            "mean": mean,
            "std": std,
            "n": n,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        })

    se = std / np.sqrt(n)
    h = stats.t.ppf((1 + confidence) / 2., n - 1) * se

    return pd.Series({
        "mean": mean,
        "std": std,
        "n": n,
        "ci_lower": mean - h,
        "ci_upper": mean + h,
    })


# --------------------------------------------------
# 7. Parameter spread among behavioral solutions
# --------------------------------------------------
beh = df[df["behavioral"] == 1]
poor = df[df["behavioral"] == 0]


def print_results(data):
    df = data.copy()
    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.float_format", "{:.4f}".format,
    ):
        print("Parameter spread among behavioral solutions (by algorithm):")
        print(
            df.groupby("algorithm")[
                [
                    "[Grain].MaximumGrainsPerCob.FixedValue",
                    "[Leaf].Photosynthesis.RUE.FixedValue",
                ]
            ].apply(lambda df: df.apply(mean_ci))
        )

        print("\nParameter spread among behavioral solutions (by method):")
        print(
            df.groupby("method")[
                [
                    "[Grain].MaximumGrainsPerCob.FixedValue",
                    "[Leaf].Photosynthesis.RUE.FixedValue",
                ]
            ].apply(lambda df: df.apply(mean_ci))
        )
