import pandas as pd
import numpy as np
import joblib
import sys
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_safe(model_path, dataset_path):
    try:
        model = joblib.load(model_path)
        
        df = pd.read_csv(dataset_path, nrows=2000)
        df = df.drop(['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], axis=1, errors='ignore')
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        y_true = df['Label'].apply(lambda x: 1 if 'ddos' in str(x).lower() or 'portscan' in str(x).lower() else 0)
        X = df.drop('Label', axis=1).select_dtypes(include='number')
        
        if X.shape[1] < model.n_features_in_:
            padding = np.zeros((X.shape[0], model.n_features_in_ - X.shape[1]))
            X = np.hstack([X, padding])
        elif X.shape[1] > model.n_features_in_:
            X = X.iloc[:, :model.n_features_in_]
        
        y_pred = model.predict(X)
        
        result = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "total_records": len(y_true),
            "actual_attacks": int(y_true.sum()),
            "predicted_attacks": int(y_pred.sum())
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) == 3:
        evaluate_safe(sys.argv[1], sys.argv[2])
