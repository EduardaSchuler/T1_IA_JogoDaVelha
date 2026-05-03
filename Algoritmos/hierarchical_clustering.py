import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'dataset'
OUTPUT_DIR = BASE_DIR / 'outputs'


def _cluster_to_label_mapping(y, clusters):
    mapping = {}
    for cluster_id in np.unique(clusters):
        labels = y[clusters == cluster_id]
        if len(labels) == 0:
            mapping[cluster_id] = 0
        else:
            mapping[cluster_id] = int(pd.Series(labels).mode()[0])
    return mapping


def _centroids_from_clusters(X, clusters):
    X_arr = np.asarray(X)
    centroids = []
    for cluster_id in np.unique(clusters):
        centroids.append(X_arr[clusters == cluster_id].mean(axis=0))
    return np.vstack(centroids)


def _predict_from_centroids(X, centroids, cluster_to_label):
    X_arr = np.asarray(X)
    dists = np.linalg.norm(X_arr[:, None, :] - centroids[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    return np.array([cluster_to_label[i] for i in nearest], dtype=int)


class HierarchicalClassifier:
    def __init__(self, centroids, cluster_to_label):
        self.centroids = np.asarray(centroids)
        self.cluster_to_label = cluster_to_label

    def predict(self, X):
        return _predict_from_centroids(X, self.centroids, self.cluster_to_label)


def treinar_hierarchical():
    print("="*60)
    print("TREINAMENTO AGRUPAMENTO HIERÁRQUICO")
    print("="*60)

    try:
        treino = pd.read_csv(DATASET_DIR / 'dataset_treino.csv')
        val = pd.read_csv(DATASET_DIR / 'dataset_validacao.csv')
        teste = pd.read_csv(DATASET_DIR / 'dataset_teste.csv')
    except FileNotFoundError:
        print("Erro: Arquivos CSV não encontrados. Rode o decision_tree.py primeiro para gerar os dados.")
        return

    target_col = 'class'
    X_tr, y_tr = treino.drop(columns=[target_col]), treino[target_col]
    X_val, y_val = val.drop(columns=[target_col]), val[target_col]
    X_te, y_te = teste.drop(columns=[target_col]), teste[target_col]

    print("Testando hiperparâmetros no conjunto de validação...")
    best_score = -1
    best_params = None

    for linkage in ['ward', 'average', 'complete']:
        model = AgglomerativeClustering(n_clusters=5, linkage=linkage)
        clusters = model.fit_predict(X_tr)
        centroids = _centroids_from_clusters(X_tr, clusters)
        mapping = _cluster_to_label_mapping(y_tr.values, clusters)
        y_val_pred = _predict_from_centroids(X_val, centroids, mapping)
        acc_val = accuracy_score(y_val, y_val_pred)

        if acc_val > best_score:
            best_score = acc_val
            best_params = {'linkage': linkage}

    print(f"Melhores parâmetros encontrados: {best_params}")
    print(f"Acurácia na Validação: {best_score:.4f}")

    print("\nTreinando o modelo final usando treino + validação...")
    X_final = pd.concat([X_tr, X_val])
    y_final = pd.concat([y_tr, y_val])

    final_model = AgglomerativeClustering(n_clusters=5, linkage=best_params['linkage'])
    final_clusters = final_model.fit_predict(X_final)
    final_centroids = _centroids_from_clusters(X_final, final_clusters)
    final_mapping = _cluster_to_label_mapping(y_final.values, final_clusters)
    classifier_data = {
        'centroids': final_centroids.tolist(),
        'mapping': {int(k): int(v) for k, v in final_mapping.items()}
    }

    y_pred = _predict_from_centroids(X_te, final_centroids, final_mapping)

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)

    print("\n--- Resultados no Conjunto de Teste ---")
    print(f"Acurácia  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F-measure : {f1:.4f}")

    with open(OUTPUT_DIR / 'hierarchical_model.pkl', 'wb') as f:
        pickle.dump(classifier_data, f)

    resumo = {
        'algoritmo': 'Agrupamento Hierárquico',
        'params': best_params,
        'acuracia_teste': round(acc, 4),
        'precision_weighted': round(prec, 4),
        'recall_weighted': round(rec, 4),
        'f1_weighted': round(f1, 4)
    }
    with open(OUTPUT_DIR / 'hierarchical_results.json', 'w') as f:
        json.dump(resumo, f, indent=2)

    print("\nModelo 'hierarchical_model.pkl' salvo na pasta outputs!")


if __name__ == '__main__':
    treinar_hierarchical()
