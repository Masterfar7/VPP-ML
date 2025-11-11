import pandas as pd
import numpy as np
import joblib
import sys
import json
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    if len(sys.argv) != 3:
        print('{"error": "Usage: python infering.py model.pkl dataset.csv"}')
        return
    
    model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    
    try:
        model = joblib.load(model_path)
        
        df = pd.read_csv(dataset_path, nrows=2000)
        
        df = df.drop(['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], axis=1, errors='ignore')
        df = df.fillna(0)
        
        ddos_count = int(df['Label'].str.contains('ddos', case=False, na=False).sum())
        portscan_count = int(df['Label'].str.contains('portscan', case=False, na=False).sum())
        normal_count = len(df) - ddos_count - portscan_count
        
        y_true = df['Label'].apply(lambda x: 1 if 'ddos' in str(x).lower() or 'portscan' in str(x).lower() else 0)
        
        X = df.drop('Label', axis=1).select_dtypes(include='number')
        
        if X.shape[1] < model.n_features_in_:
            padding = np.zeros((X.shape[0], model.n_features_in_ - X.shape[1]))
            X = np.hstack([X, padding])
        elif X.shape[1] > model.n_features_in_:
            X = X.iloc[:, :model.n_features_in_]
        
        y_pred = model.predict(X)
        
        result = {
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "total_records": len(df),
            "actual_attacks": int(y_true.sum()),
            "predicted_attacks": int(y_pred.sum()),
            "ddos_count": ddos_count,
            "portscan_count": portscan_count,
            "normal_count": normal_count
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
