import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import sys

def load_and_balance_data(dataset_paths):
    all_data = []
    
    for path in dataset_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        print(f"Loading: {path}")
        try:
            df = pd.read_csv(path, nrows=5000)
            all_data.append(df)
            print(f"   Loaded {len(df)} records")
        except Exception as e:
            print(f"   Error: {e}")
            continue
    
    if not all_data:
        print("No data loaded!")
        return None, None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined: {len(combined_df)} total records")
    
    attacks = combined_df[combined_df['Label'].str.contains('ddos|portscan', case=False, na=False)]
    normal = combined_df[~combined_df['Label'].str.contains('ddos|portscan', case=False, na=False)]
    
    print(f"Attacks found: {len(attacks)}")
    print(f"Normal found: {len(normal)}")
    
    sample_size = min(1000, len(attacks), len(normal))
    
    if sample_size < 100:
        print("Very small dataset, using all available data")
        sample_size = min(len(attacks), len(normal))
    
    attacks_balanced = attacks.sample(n=sample_size, random_state=42)
    normal_balanced = normal.sample(n=sample_size, random_state=42)
    
    balanced_df = pd.concat([attacks_balanced, normal_balanced])
    balanced_df = balanced_df.sample(frac=1, random_state=42)  # Shuffle
    
    print(f"Balanced dataset: {len(balanced_df)} records")
    print(f"   Attacks: {sample_size}, Normal: {sample_size}")
    
    return balanced_df

def preprocess_data(df):
    cols_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    df_clean = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)
    
    for col in df_clean.select_dtypes(include=[np.number]):
        df_clean[col] = df_clean[col].clip(lower=-1e9, upper=1e9)
    
    df_clean['is_attack'] = df_clean['Label'].apply(
        lambda x: 1 if any(attack in str(x).lower() for attack in ['ddos', 'portscan']) else 0
    )
    
    X = df_clean.drop(['Label', 'is_attack'], axis=1)
    X = X.select_dtypes(include=[np.number])
    y = df_clean['is_attack']
    
    return X, y

def main():
    datasets = [
        'datasets/final_dataset.csv',
        'datasets/unbalaced_20_80_dataset.csv', 
        'datasets/balanced_50_50.csv'
    ]
    
    balanced_df = load_and_balance_data(datasets)
    if balanced_df is None:
        print(" Failed to load data")
        return
    
    X, y = preprocess_data(balanced_df)
    
    print(f"\n Final training data:")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Attacks: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Normal: {len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")
    
    if X.shape[0] < 100:
        print("Too few samples for training!")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n Training model...")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n TRAINING RESULTS:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print(f"\n Confusion Matrix:")
    print(f"           Predicted")
    print(f"           Normal  Attack")
    print(f"Actual Normal  {cm[0,0]:>5}   {cm[0,1]:>6}")
    print(f"       Attack   {cm[1,0]:>5}   {cm[1,1]:>6}")
    
    joblib.dump(model, 'trained_model.pkl')
    print(f"\n Model saved as 'trained_model.pkl'")
    print(f"Model features: {model.n_features_in_}")
    print(f"Universal training completed!")
    print(f"\n Now you can run: ./detect path/to/any/dataset.csv")

if __name__ == "__main__":
    main()
