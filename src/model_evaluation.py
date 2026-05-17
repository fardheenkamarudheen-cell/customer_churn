from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return {
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "auc": round(auc, 4)
    }