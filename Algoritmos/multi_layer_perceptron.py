import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, classification_report,
    precision_score, recall_score, f1_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'dataset'
OUTPUT_DIR = BASE_DIR / 'outputs'


def treinar_mlp():
    print("="*60)
    print("TREINAMENTO MLP (Multi-Layer Perceptron)")
    print("="*60)

    try:
        treino = pd.read_csv(DATASET_DIR / 'dataset_treino.csv')
        val    = pd.read_csv(DATASET_DIR / 'dataset_validacao.csv')
        teste  = pd.read_csv(DATASET_DIR / 'dataset_teste.csv')
    except FileNotFoundError:
        print("Erro: Arquivos CSV não encontrados. Rode o decision_tree.py primeiro para gerar os dados.")
        return

    target_col = 'class'
    X_tr  = treino.drop(columns=[target_col]).values
    y_tr  = treino[target_col].values
    X_val = val.drop(columns=[target_col]).values
    y_val = val[target_col].values
    X_te  = teste.drop(columns=[target_col]).values
    y_te  = teste[target_col].values

    #busca de hiperparâmetros no conjunto de validação
    print("Testando hiperparâmetros no conjunto de validação...")
    best_score  = -1
    best_params = None

    for hidden_layer_sizes in [(64,), (128,), (64, 32), (128, 64), (64, 64, 32)]:
        for learning_rate_init in [0.01, 0.05, 0.1]:
            for momentum in [0.5, 0.9]:
                mlp = MLPClassifier(
                    solver='adam',
                    hidden_layer_sizes=hidden_layer_sizes,
                    learning_rate_init=learning_rate_init,
                    momentum=momentum,
                    max_iter=2000,
                    random_state=42,
                    early_stopping=True
                )
                mlp.fit(X_tr, y_tr)
                acc_val = accuracy_score(y_val, mlp.predict(X_val))

                if acc_val > best_score:
                    best_score  = acc_val
                    best_params = {
                        'hidden_layer_sizes': hidden_layer_sizes,
                        'learning_rate_init': learning_rate_init,
                        'momentum':           momentum,
                    }

    print(f"Melhores parâmetros encontrados: {best_params}")
    print(f"Topologia: entrada -> {best_params['hidden_layer_sizes']} -> saída")
    print(f"Acurácia na Validação: {best_score:.4f}")

    #modelo final treinado em treino + validação
    print("\nTreinando o modelo final usando treino + validação...")
    X_final = np.concatenate([X_tr, X_val])
    y_final = np.concatenate([y_tr, y_val])

    modelo_final = MLPClassifier(
        solver='adam',
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        learning_rate_init=best_params['learning_rate_init'],
        momentum=best_params['momentum'],
        max_iter=2000,
        random_state=42
    )
    modelo_final.fit(X_final, y_final)

    y_pred        = modelo_final.predict(X_te)
    classes_unicas = np.unique(y_final)

    #matriz de confusão
    cm   = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_unicas)
    disp.plot(cmap='Blues')

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_te, y_pred, average='weighted', zero_division=0)

    print("\n--- Resultados no Conjunto de Teste ---")
    print(f"Acurácia  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F-measure : {f1:.4f}")
    print(classification_report(y_te, y_pred))

    with open(OUTPUT_DIR / 'mlp_model.pkl', 'wb') as f:
        pickle.dump(modelo_final, f)

    resumo = {
        'algoritmo': 'MLP',
        'params': {
            **best_params,
            'hidden_layer_sizes': list(best_params['hidden_layer_sizes'])
        },
        'acuracia_teste':     round(acc,  4),
        'precision_weighted': round(prec, 4),
        'recall_weighted':    round(rec,  4),
        'f1_weighted':        round(f1,   4)
    }
    with open(OUTPUT_DIR / 'mlp_results.json', 'w') as f:
        json.dump(resumo, f, indent=2)

    print("\nModelo 'mlp_model.pkl' salvo na pasta outputs!")


if __name__ == '__main__':
    treinar_mlp()