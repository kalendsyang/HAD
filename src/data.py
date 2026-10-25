import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

NSL_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'attack_type', 'difficulty',
]


def load_nsl(data_path):
    train = pd.read_csv(os.path.join(data_path, 'KDDTrain+.txt'), names=NSL_COLUMNS)
    test = pd.read_csv(os.path.join(data_path, 'KDDTest+.txt'), names=NSL_COLUMNS)
    for df in (train, test):
        df['label'] = (df['attack_type'] != 'normal').astype(int)
        df.drop(columns=['attack_type', 'difficulty'], inplace=True)
    return train, test


def load_cic(data_path, seed):
    df = pd.read_csv(os.path.join(data_path, 'all_raw_data.csv'), low_memory=False)
    df = df.drop(columns=['attack_type', 'date'], errors='ignore')
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return train_test_split(df, test_size=0.2, stratify=df['label'], random_state=seed)


def get_data(root_path, dataset, seed=2024):
    """Return min-max scaled (x_train, y_train, x_test, y_test)."""
    data_path = os.path.join(root_path, 'data', dataset)
    train, test = load_nsl(data_path) if dataset == 'nsl' else load_cic(data_path, seed)

    for col in train.select_dtypes(include='object').columns:
        encoder = LabelEncoder().fit(pd.concat([train[col], test[col]]))
        train[col] = encoder.transform(train[col])
        test[col] = encoder.transform(test[col])

    x_train = train.drop(columns='label').values.astype(np.float32)
    x_test = test.drop(columns='label').values.astype(np.float32)
    scaler = MinMaxScaler().fit(x_train)

    return (scaler.transform(x_train).astype(np.float32), train['label'].values,
            scaler.transform(x_test).astype(np.float32), test['label'].values)
