import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import random, pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / 'dataset'
OUTPUT_DIR = BASE_DIR / 'outputs'
DATASET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Convenção de nomes das posições (colunas do CSV/inputs do frontend)
# tl=top-left, tm=top-middle, tr=top-right, ml=middle-left, ...
COLS  = ['tl','tm','tr','ml','mm','mr','bl','bm','br']

# Todas as linhas possíveis de 3-em-linha no tabuleiro 3x3 (índices 0..8)
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

# Mapeamento do id numérico da classe para o nome usado em prints/relatórios.
CLASS_NAMES = {
    0: 'Tem Jogo',
    1: 'Poss. Fim de Jogo',
    2: 'Empate',
    3: 'O vence',
    4: 'X vence'
}

# Codificação do tabuleiro (entrada categórica simples -> inteiro)
ENCODE = {'b': 0, 'o': 1, 'x': 2}

# FUNÇÕES DE LÓGICA E FEATURES
def check_winner(b):
    """Retorna 'x'/'o' se houver 3-em-linha; senão None."""
    # Procura 3-em-linha de 'x' ou 'o'.
    for a,c,d in LINES:
        if b[a]==b[c]==b[d] and b[a]!='b':
            return b[a]  # encontrou um vencedor
    return None  # sem vencedor

def has_threat(b, sym):
    """True se `sym` tem uma linha com 2 peças + 1 vazio."""
    # Ameaça: linha com 2 peças do jogador + 1 vazio.
    for a,c,d in LINES:
        line = [b[a],b[c],b[d]]
        if line.count(sym)==2 and 'b' in line:
            return True  # existe pelo menos uma ameaça
    return False  # nenhuma ameaça encontrada

def classify(b):
    """Retorna a classe (0..4) do tabuleiro."""
    # Classe do tabuleiro (regras determinísticas).
    # Prioridade: vitória -> empate -> ameaça -> jogo em andamento.
    w = check_winner(b)
    if w == 'x':
        return 4  # X vence
    if w == 'o':
        return 3  # O vence
    if 'b' not in b:
        return 2  # empate (tabuleiro cheio)
    if has_threat(b, 'x') or has_threat(b, 'o'):
        return 1  # possível fim de jogo (ameaça de vitória)
    return 0  # tem jogo

def extract_features(board_dict):
    """Extrai o vetor de features (16) a partir do tabuleiro."""
    # Features (16):
    # - 9 casas codificadas (b=0, o=1, x=2)
    # - contagem de X, O e vazios
    # - contagem de "ameaças" (2 + vazio) e de "bloqueios" (2 + adversário)
    b = [board_dict[c] for c in COLS]
    enc = [ENCODE[v] for v in b]
    
    nx = b.count('x')  # quantidade de X no tabuleiro
    no = b.count('o')  # quantidade de O no tabuleiro
    nb = b.count('b')  # quantidade de casas vazias no tabuleiro
    
    tx = sum(1 for a,c,d in LINES if [b[a],b[c],b[d]].count('x')==2 and 'b' in [b[a],b[c],b[d]])  # quantas linhas têm 2 X + 1 vazio (ameaças de X)
    to = sum(1 for a,c,d in LINES if [b[a],b[c],b[d]].count('o')==2 and 'b' in [b[a],b[c],b[d]])  # quantas linhas têm 2 O + 1 vazio (ameaças de O)
    
    bx = sum(1 for a,c,d in LINES if [b[a],b[c],b[d]].count('x')==2 and [b[a],b[c],b[d]].count('o')==1)  # quantas linhas têm 2 X + 1 O (linha de X bloqueada por O)
    bo = sum(1 for a,c,d in LINES if [b[a],b[c],b[d]].count('o')==2 and [b[a],b[c],b[d]].count('x')==1)  # quantas linhas têm 2 O + 1 X (linha de O bloqueada por X)
    
    return enc + [nx, no, nb, tx, to, bx, bo]  # 16 valores

FEAT_NAMES = COLS + ['n_x','n_o','n_blank','threats_x','threats_o','blocked_x','blocked_o']

# CONSTRUÇÃO DO DATASET
if __name__ == "__main__":
    print("="*60)
    print("CONSTRUÇÃO DO DATASET")
    print("="*60)

    # Para cada classe, guardamos um conjunto de tabuleiros únicos.
    # Usar set evita duplicatas (mesmo tabuleiro gerado em jogos diferentes).
    board_set = {i: set() for i in range(5)}

    def simulate_games(n=80000):
        # Simula partidas aleatórias e guarda tabuleiros (únicos) por classe.
        for _ in range(n):
            board = ['b']*9
            turn  = 'x'
            for _ in range(9):
                empty = [i for i,c in enumerate(board) if c=='b']
                if not empty: break
                pos = random.choice(empty)
                board[pos] = turn
                cls = classify(board)
                board_set[cls].add(tuple(board))
                if cls in (2,3,4): break
                turn = 'o' if turn=='x' else 'x'

    simulate_games()

    print("Instâncias únicas geradas por classe:")
    for cls, name in CLASS_NAMES.items():
        print(f"  {name}: {len(board_set[cls])}")

    # Amostragem por classe:
    # - Pegamos até 1500 tabuleiros por classe (quando houver muitos).
    # - Para empate (classe 2), usamos todos os exemplos disponíveis.
    # Observação: isso é um "balanceamento por amostragem" simples.
    rows = []
    for cls in range(5):
        pool  = list(board_set[cls])
        limit = 1500 if cls != 2 else len(pool)
        samp  = random.sample(pool, min(limit, len(pool)))
        for b in samp:
            feats = extract_features(dict(zip(COLS, b)))
            rows.append(feats + [cls])

    all_feat_cols = FEAT_NAMES + ['class']
    df = pd.DataFrame(rows, columns=all_feat_cols)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nDataset Final Balanceado ({len(df)} instâncias):")
    for cls, name in CLASS_NAMES.items():
        print(f"  {name}: {(df['class']==cls).sum()}")

    # Divisão do dataset:
    # - 60% treino
    # - 20% validação (para escolher hiperparâmetros)
    # - 20% teste (avaliação final, "dados não vistos")
    # Usamos `stratify` para manter proporção de classes em cada divisão.
    X = df[FEAT_NAMES]
    y = df['class']
    
    X_temp, X_te, y_temp, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    for split_X, split_y, fname in [
            (X_tr, y_tr, 'dataset_treino.csv'),
            (X_val, y_val, 'dataset_validacao.csv'),
            (X_te, y_te, 'dataset_teste.csv')]:
        tmp = split_X.copy(); tmp['class'] = split_y
        tmp.to_csv(DATASET_DIR / fname, index=False)

    # TUNAGEM DE HIPERPARÂMETROS
    print("\n" + "="*60)
    print("TUNAGEM DE HIPERPARÂMETROS")
    print("="*60)

    best_score = -1
    best_params = None
    results = []

    for criterion in ['gini','entropy']:
        for max_depth in [4, 6, 8, 10, 12, None]:
            for mss in [2, 5, 10]:
                for msl in [1, 2, 5]:
                    dt = DecisionTreeClassifier(
                        criterion=criterion, max_depth=max_depth,
                        min_samples_split=mss, min_samples_leaf=msl,
                        class_weight='balanced', # Lida com o desbalanceamento naturalmente
                        random_state=42)
                    
                    dt.fit(X_tr, y_tr)
                    tr_a  = accuracy_score(y_tr, dt.predict(X_tr))
                    val_a = accuracy_score(y_val, dt.predict(X_val))
                    
                    # Penaliza overfitting:
                    # - se treino >> validação, reduzimos a pontuação
                    # - pequena margem (0.05) é tolerada
                    ov = tr_a - val_a
                    score = val_a - 0.3 * max(0, ov - 0.05)
                    
                    results.append({'criterion':criterion,'max_depth':str(max_depth),
                                     'mss':mss,'msl':msl,
                                     'tr_acc':round(tr_a,4),'val_acc':round(val_a,4),
                                     'score':round(score,4)})
                    
                    if score > best_score:
                        best_score = score
                        best_params = dict(criterion=criterion, max_depth=max_depth,
                                           min_samples_split=mss, min_samples_leaf=msl)

    print(f"Melhores parâmetros encontrados: {best_params}")

    # MODELO FINAL E TESTE
    print("\n" + "="*60)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("="*60)

    best_dt = DecisionTreeClassifier(**best_params, class_weight='balanced', random_state=42)
    # Modelo final:
    # Depois de escolher hiperparâmetros, treinamos de novo usando
    # (treino + validação) para aproveitar mais dados.
    best_dt.fit(pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val]))
    
    y_pred = best_dt.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_te,   y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_te,       y_pred, average='weighted', zero_division=0)

    print(f"\nAcurácia  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F-measure : {f1:.4f}")

    with open(OUTPUT_DIR / 'decision_tree_model.pkl','wb') as f:
        pickle.dump(best_dt, f)

    print("\nModelo salvo em 'outputs/decision_tree_model.pkl'. Pronto para o Frontend!")