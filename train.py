import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def main():
    all_data = []
    
    try:
        ddos_df = pd.read_csv('datasets/final_dataset_small.csv', nrows=2000)
        ddos_df = ddos_df[ddos_df['Label'].str.contains('ddos', case=False, na=False)]
        ddos_sample = ddos_df.sample(n=800, random_state=42)
        all_data.append(ddos_sample)
        print(f"   DDoS: {len(ddos_sample)} records")
    except Exception as e:
        print(f"   Error loading DDoS: {e}")

    try:
        portscan_data = []
        with open('datasets/scan_small.json', 'r') as f:
            for i, line in enumerate(f):
                if i >= 800:
                    break
                try:
                    data = json.loads(line.strip())
                    portscan_data.append(data)
                except:
                    continue
        
        portscan_df = pd.DataFrame(portscan_data)
        portscan_df['Label'] = 'PortScan'
        all_data.append(portscan_df)
        print(f"   PortScan: {len(portscan_df)} records")
    except Exception as e:
        print(f"   Error loading PortScan: {e}")

    try:
        balanced_df = pd.read_csv('datasets/balanced_50_50_small.csv', nrows=2000)
        normal_traffic = balanced_df[~balanced_df['Label'].str.contains('ddos|portscan', case=False, na=False)]
        normal_count = len(normal_traffic)
        normal_sample = normal_traffic.sample(n=normal_count, random_state=42)
        all_data.append(normal_sample)
        print(f"   Normal: {len(normal_sample)} records")
    except Exception as e:
        print(f"   Error loading normal: {e}")

    if not all_data:
        print("No data loaded!")
        return
        
    train_df = pd.concat(all_data, ignore_index=True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    
    ddos_count = train_df['Label'].str.contains('ddos', case=False, na=False).sum()
    portscan_count = train_df['Label'].str.contains('portscan', case=False, na=False).sum()
    normal_count = len(train_df) - ddos_count - portscan_count
    
    print(f"   DDoS: {ddos_count} ({ddos_count/len(train_df)*100:.1f}%)")
    print(f"   PortScan: {portscan_count} ({portscan_count/len(train_df)*100:.1f}%)")
    print(f"   Normal: {normal_count} ({normal_count/len(train_df)*100:.1f}%)")

    cols_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    df_clean = train_df.drop([col for col in cols_to_drop if col in train_df.columns], axis=1, errors='ignore')
    df_clean = df_clean.fillna(0)
    
    df_clean['is_attack'] = df_clean['Label'].apply(lambda x: 1 if any(attack in str(x).lower() for attack in ['ddos','portscan']) else 0)
    
    X = df_clean.drop(['Label', 'is_attack'], axis=1, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df_clean['is_attack']
    
    print(f"   Samples: {X.shape[0]}")
    print(f"   Attacks: {y.sum()}, Normal: {len(y)-y.sum()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100,max_depth=20,random_state=42,class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    
    joblib.dump(model, 'trained_model.pkl')
    print(f"\nModel saved as 'trained_model.pkl'")
    
if __name__ == "__main__":
    main()
