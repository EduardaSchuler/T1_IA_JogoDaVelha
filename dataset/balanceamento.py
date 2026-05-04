import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from pathlib import Path

FEAT_NAMES = ['tl','tm','tr','ml','mm','mr','bl','bm','br',
                'n_x','n_o','n_blank','threats_x','threats_o','blocked_x','blocked_o']

df = pd.concat([
    pd.read_csv('dataset_treino.csv'),
    pd.read_csv('dataset_validacao.csv'),
    pd.read_csv('dataset_teste.csv'),
])

X, y = df[FEAT_NAMES].values, df['class'].values

print("Antes:", Counter(y))

k = min(5, min(Counter(y).values()) - 1)
X_bal, y_bal = SMOTE(k_neighbors=k, random_state=42).fit_resample(X, y)

X_bal = np.clip(np.round(X_bal), 0, 2).astype(int)

print("Depois:", Counter(y_bal))

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_bal, y_bal, test_size=0.4, stratify=y_bal, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

Path('').mkdir(exist_ok=True)
for X_s, y_s, nome in [(X_tr, y_tr, 'treino'), (X_val, y_val, 'validacao'), (X_te, y_te, 'teste')]:
    pd.DataFrame(X_s, columns=FEAT_NAMES).assign(**{'class': y_s}).to_csv(f'dataset_{nome}.csv', index=False)

print("Salvo!")