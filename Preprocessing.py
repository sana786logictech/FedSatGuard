"""
preprocess_satellite_datasets.py
Prepares WetLinks and Spache datasets for FedSatGuard experiments
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# ============================================================================
# 1. Process WetLinks Dataset
# ============================================================================

def process_wetlinks(data_path="WetLinks/data"):
    """
    Process WetLinks dataset into client-ready format with network features
    """
    print("Processing WetLinks dataset...")
    
    # Load main measurement file (adjust filename based on actual data)
    # WetLinks provides processed CSV files
    df_wet = pd.read_csv(os.path.join(data_path, "measurements.csv"))
    
    # Extract relevant features
    wet_features = pd.DataFrame({
        'timestamp': pd.to_datetime(df_wet['timestamp']),
        'client_id': df_wet['location_id'],  # 2 locations
        'rtt_ms': df_wet['rtt_avg'],
        'throughput_mbps': df_wet['download_speed'] / 1e6,
        'packet_loss_pct': df_wet['packet_loss'] * 100,
        'jitter_ms': df_wet['jitter'],
        'weather_temp': df_wet['temperature'],
        'weather_humidity': df_wet['humidity'],
        'weather_rain': df_wet['rainfall'],
        'traceroute_hops': df_wet['traceroute_hops']
    })
    
    # Create client profiles (2 clients from 2 locations)
    clients = {}
    for client_id in wet_features['client_id'].unique():
        client_data = wet_features[wet_features['client_id'] == client_id].copy()
        
        # Sort by timestamp
        client_data = client_data.sort_values('timestamp')
        
        # Create sliding windows for time-series features
        window_size = 10  # 10 consecutive measurements
        X, y = [], []
        
        for i in range(len(client_data) - window_size):
            window = client_data.iloc[i:i+window_size]
            
            # Features: RTT, throughput, loss, jitter, weather
            feature_vector = window[[
                'rtt_ms', 'throughput_mbps', 'packet_loss_pct', 
                'jitter_ms', 'weather_temp', 'weather_humidity'
            ]].values.flatten()
            
            # Label: next RTT (regression) or anomaly (binary)
            next_rtt = client_data.iloc[i+window_size]['rtt_ms']
            # Binary label: 1 if next RTT > 2*median (anomaly)
            label = 1 if next_rtt > 2 * client_data['rtt_ms'].median() else 0
            
            X.append(feature_vector)
            y.append(label)
        
        clients[f"wetlinks_client_{client_id}"] = {
            'X': np.array(X),
            'y': np.array(y),
            'metadata': {
                'location': client_data['client_id'].iloc[0],
                'n_samples': len(X)
            }
        }
    
    print(f"  Created {len(clients)} clients with WetLinks data")
    for cid, data in clients.items():
        print(f"    {cid}: {data['X'].shape[0]} samples, {data['X'].shape[1]} features")
    
    return clients


# ============================================================================
# 2. Process Spache Latency Dataset
# ============================================================================

def process_spache(data_path="spache_dataset"):
    """
    Process Spache global latency measurements
    """
    print("\nProcessing Spache dataset...")
    
    # Load latency measurements (adjust filename)
    df_spache = pd.read_csv(os.path.join(data_path, "latency_measurements.csv"))
    
    # Spache format: probe_id, continent, country, target_website, rtt_min, timestamp
    
    # Create synthetic clients based on probes
    # Group probes by continent to create 5 clients
    continent_clients = {}
    
    for continent in df_spache['continent'].unique():
        continent_data = df_spache[df_spache['continent'] == continent].copy()
        
        # Pivot to create feature vectors: each target website becomes a feature
        pivot_data = continent_data.pivot_table(
            index='timestamp',
            columns='target_website',
            values='rtt_min',
            aggfunc='mean'
        ).fillna(method='ffill').dropna()
        
        # Create sliding windows
        window_size = 5
        X, y = [], []
        
        for i in range(len(pivot_data) - window_size):
            window = pivot_data.iloc[i:i+window_size]
            
            # Features: flattened RTTs to all websites over window
            feature_vector = window.values.flatten()
            
            # Label: average RTT change (anomaly if > threshold)
            avg_rtt_change = pivot_data.iloc[i+window_size].mean() - window.iloc[-1].mean()
            label = 1 if abs(avg_rtt_change) > 20 else 0  # 20ms threshold
            
            X.append(feature_vector)
            y.append(label)
        
        continent_clients[f"spache_client_{continent}"] = {
            'X': np.array(X),
            'y': np.array(y),
            'metadata': {
                'continent': continent,
                'n_probes': continent_data['probe_id'].nunique(),
                'n_samples': len(X)
            }
        }
    
    print(f"  Created {len(continent_clients)} clients (one per continent)")
    for cid, data in continent_clients.items():
        print(f"    {cid}: {data['X'].shape[0]} samples, {data['X'].shape[1]} features")
    
    return continent_clients


# ============================================================================
# 3. Combine and Save for FedSatGuard
# ============================================================================

def prepare_for_fedsatguard(wet_clients, spache_clients, output_dir="fedsatguard_data"):
    """
    Format datasets for FedSatGuard framework
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_clients = {}
    all_clients.update(wet_clients)
    all_clients.update(spache_clients)
    
    # Split into train/test (80/20) for each client
    for client_name, client_data in all_clients.items():
        X = client_data['X']
        y = client_data['y']
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save client data
        client_dir = os.path.join(output_dir, client_name)
        os.makedirs(client_dir, exist_ok=True)
        
        np.save(os.path.join(client_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(client_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(client_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(client_dir, 'y_test.npy'), y_test)
        
        with open(os.path.join(client_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save metadata
        with open(os.path.join(client_dir, 'metadata.txt'), 'w') as f:
            for key, val in client_data['metadata'].items():
                f.write(f"{key}: {val}\n")
    
    # Create summary file
    with open(os.path.join(output_dir, 'dataset_summary.txt'), 'w') as f:
        f.write("FedSatGuard Real Satellite Datasets\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total clients: {len(all_clients)}\n\n")
        
        for client_name, client_data in all_clients.items():
            f.write(f"{client_name}:\n")
            f.write(f"  Samples: {client_data['X'].shape[0]}\n")
            f.write(f"  Features: {client_data['X'].shape[1]}\n")
            f.write(f"  Class balance: {np.mean(client_data['y']):.2%} anomalies\n")
            for key, val in client_data['metadata'].items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")
    
    print(f"\n✅ Data saved to {output_dir}/")
    return all_clients


# ============================================================================
# 4. Main Execution
# ============================================================================

if __name__ == "__main__":
    # Set paths (update these to your download locations)
    WETLINKS_PATH = "WetLinks/data"  # Change this
    SPACHE_PATH = "spache_dataset"    # Change this
    
    # Process datasets
    wet_clients = process_wetlinks(WETLINKS_PATH)
    spache_clients = process_spache(SPACHE_PATH)
    
    # Combine and save
    all_clients = prepare_for_fedsatguard(wet_clients, spache_clients)
    
    print("\n✅ Preprocessing complete!")
    print("Now you can modify your FL trainer to load these datasets")