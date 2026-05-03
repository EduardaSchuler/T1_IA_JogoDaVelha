import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / 'dataset'
OUTPUT_DIR = BASE_DIR / 'outputs'

def treinar_random_forest():
    print("="*60)
    print("TREINAMENTO RANDOM FOREST")
    print("="*60)

    # Carrega os dados já separados
    try:
        treino = pd.read_csv(DATASET_DIR / 'dataset_treino.csv')
        val = pd.read_csv(DATASET_DIR / 'dataset_validacao.csv')
        teste = pd.read_csv(DATASET_DIR / 'dataset_teste.csv')
    except FileNotFoundError:
        print("Erro: Arquivos CSV não encontrados. Rode o decision_tree_final.py primeiro.")
        return

    # Separar features (X) e alvo (y)
    # class é a classe numérica 0..4 que o modelo deve prever
    coluna_alvo = 'class'
    X_tr, y_tr = treino.drop(columns=[coluna_alvo]), treino[coluna_alvo]
    X_val, y_val = val.drop(columns=[coluna_alvo]), val[coluna_alvo]
    X_te, y_te = teste.drop(columns=[coluna_alvo]), teste[coluna_alvo]

    # Tunagem simples: testa combinações e escolhe a melhor pela acurácia na validação
    print("Testando hiperparâmetros no conjunto de validação...")
    melhor_score = -1
    melhores_params = {}

    # Testando o número de árvores (n_estimators) e a profundidade (max_depth)
    for n_estimators in [50, 100, 150]:
        for max_depth in [5, 10, 15, None]:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,  # quantidade de árvores
                max_depth=max_depth,        # limite de profundidade (None = sem limite)
                class_weight='balanced',
                random_state=42,
                n_jobs=-1  # usa todos os núcleos do processador
            )
            rf.fit(X_tr, y_tr)
            
            # Avalia no conjunto de validação
            val_pred = rf.predict(X_val)
            acc_val = accuracy_score(y_val, val_pred)  # % de acertos
            
            if acc_val > melhor_score:
                melhor_score = acc_val
                melhores_params = {'n_estimators': n_estimators, 'max_depth': max_depth}

    print(f"Melhores parâmetros encontrados: {melhores_params}")
    print(f"Acurácia na Validação: {melhor_score:.4f}")

    # Modelo final: treina de novo usando (treino + validação) para aproveitar mais dados
    print("\nTreinando o modelo final e testando...")
    X_final_treino = pd.concat([X_tr, X_val])
    y_final_treino = pd.concat([y_tr, y_val])

    modelo_final = RandomForestClassifier(
        **melhores_params, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    modelo_final.fit(X_final_treino, y_final_treino)

    # Avaliação no conjunto de teste
    y_pred = modelo_final.predict(X_te)
    
    acc  = accuracy_score(y_te, y_pred)  # acurácia no teste
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)  # precision ponderado
    rec  = recall_score(y_te, y_pred, average='weighted', zero_division=0)     # recall ponderado
    f1   = f1_score(y_te, y_pred, average='weighted', zero_division=0)         # F1 ponderado

    print("\n--- Resultados no Conjunto de Teste ---")
    print(f"Acurácia  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F-measure : {f1:.4f}")

    with open(OUTPUT_DIR / 'random_forest_model.pkl', 'wb') as f:
        pickle.dump(modelo_final, f)
        
    resumo = {
        'algoritmo': 'Random Forest',
        'params': melhores_params,
        'acuracia_teste': round(acc, 4),
        'precision_weighted': round(prec, 4),
        'recall_weighted': round(rec, 4),
        'f1_weighted': round(f1, 4)
    }
    with open(OUTPUT_DIR / 'rf_results.json', 'w') as f:
        json.dump(resumo, f, indent=2)

    print("\nModelo 'random_forest_model.pkl' salvo na pasta outputs!")

if __name__ == "__main__":
    treinar_random_forest()