def load_real_satellite_data(data_dir="fedsatguard_data"):
    """
    Load preprocessed real satellite datasets for FL training
    """
    import os
    import numpy as np
    
    clients = {}
    client_names = [d for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d))]
    
    for client_name in client_names:
        client_dir = os.path.join(data_dir, client_name)
        
        # Load data
        X_train = np.load(os.path.join(client_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(client_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(client_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(client_dir, 'y_test.npy'))
        
        clients[client_name] = {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
            'n_samples': len(X_train)
        }
        
        print(f"Loaded {client_name}: {len(X_train)} train, {len(X_test)} test samples")
    
    return clients