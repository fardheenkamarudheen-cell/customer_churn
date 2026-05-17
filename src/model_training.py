from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

def train_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.9
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/churn_model.pkl")

    return model, X_test, y_test