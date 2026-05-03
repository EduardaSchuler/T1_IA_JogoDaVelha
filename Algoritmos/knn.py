import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'dataset'
OUTPUT_DIR = BASE_DIR / 'outputs'


def treinar_knn():
    print("="*60)
    print("TREINAMENTO K-NN")
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

    # Escolhendo valores típicos para k e tipos de ponderação.
    # KNN tende a funcionar bem com k ímpar em problemas de classificação multi-classe.
    for n_neighbors in [3, 5, 7, 9, 11]:
        for weights in ['uniform', 'distance']:
            for p in [1, 2]:
                knn = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    p=p,
                    n_jobs=-1
                )
                knn.fit(X_tr, y_tr)
                val_pred = knn.predict(X_val)
                acc_val = accuracy_score(y_val, val_pred)

                if acc_val > best_score:
                    best_score = acc_val
                    best_params = {
                        'n_neighbors': n_neighbors,
                        'weights': weights,
                        'p': p
                    }

    print(f"Melhores parâmetros encontrados: {best_params}")
    print(f"Acurácia na Validação: {best_score:.4f}")

    print("\nTreinando o modelo final usando treino + validação...")
    X_final = pd.concat([X_tr, X_val])
    y_final = pd.concat([y_tr, y_val])

    modelo_final = KNeighborsClassifier(
        **best_params,
        n_jobs=-1
    )
    modelo_final.fit(X_final, y_final)

    y_pred = modelo_final.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)

    print("\n--- Resultados no Conjunto de Teste ---")
    print(f"Acurácia  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F-measure : {f1:.4f}")

    with open(OUTPUT_DIR / 'knn_model.pkl', 'wb') as f:
        pickle.dump(modelo_final, f)

    resumo = {
        'algoritmo': 'k-NN',
        'params': best_params,
        'acuracia_teste': round(acc, 4),
        'precision_weighted': round(prec, 4),
        'recall_weighted': round(rec, 4),
        'f1_weighted': round(f1, 4)
    }
    with open(OUTPUT_DIR / 'knn_results.json', 'w') as f:
        json.dump(resumo, f, indent=2)

    print("\nModelo 'knn_model.pkl' salvo na pasta outputs!")


if __name__ == '__main__':
    treinar_knn()
