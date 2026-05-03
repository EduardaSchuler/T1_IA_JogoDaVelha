import warnings
import pandas as pd
import pickle, random, json, os
import numpy as np
from flask import Flask, render_template_string, request, jsonify

"""
frontend.py  —  Jogo da Velha IA Classifier
Servidor Flask que expõe o jogo e a classificação de estado por modelos de ML.
"""

# ─── caminho dos artefatos ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ─────────────────────────────────────────────────────────────────────────────
#  REGISTRY DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────
COLS  = ['tl','tm','tr','ml','mm','mr','bl','bm','br']
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
ENCODE = {'b': 0, 'o': 1, 'x': 2}

CLASS_NAMES = {
    0: 'Tem Jogo',
    1: 'Possibilidade de Fim de Jogo',
    2: 'Empate',
    3: 'O vence',
    4: 'X vence',
}


def _winner(board: list) -> str | None:
  for a, c, d in LINES:
    if board[a] == board[c] == board[d] and board[a] != 'b':
      return board[a]
  return None


def _has_threat(board: list, sym: str) -> bool:
  for a, c, d in LINES:
    ln = [board[a], board[c], board[d]]
    if ln.count(sym) == 2 and 'b' in ln:
      return True
  return False

def _extract_features(board: list) -> list:
    """Extrai as mesmas 16 features usadas no treinamento."""
    enc = [ENCODE[v] for v in board]
    nx  = board.count('x'); no = board.count('o'); nb = board.count('b')
    tx = sum(1 for a,c,d in LINES
             if [board[a],board[c],board[d]].count('x')==2 and 'b' in [board[a],board[c],board[d]])
    to = sum(1 for a,c,d in LINES
             if [board[a],board[c],board[d]].count('o')==2 and 'b' in [board[a],board[c],board[d]])
    bx = sum(1 for a,c,d in LINES
             if [board[a],board[c],board[d]].count('x')==2 and [board[a],board[c],board[d]].count('o')==1)
    bo = sum(1 for a,c,d in LINES
             if [board[a],board[c],board[d]].count('o')==2 and [board[a],board[c],board[d]].count('x')==1)
    return enc + [nx, no, nb, tx, to, bx, bo]


class BaseClassifier:
    """Interface que todo modelo deve implementar."""
    name: str = "Base"
    description: str = ""

    def predict(self, board: list) -> int:
        raise NotImplementedError

    def is_available(self) -> bool:
        return True


class DecisionTreeClassifier_(BaseClassifier):
    name        = "Árvore de Decisão"
    description = "Modelo supervisionado e interpretável. Treinado com features do tabuleiro e balanceamento nativo de classes."
    model_file  = os.path.join(OUTPUT_DIR, "decision_tree_model.pkl")

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            with open(self.model_file, 'rb') as f:
                self._model = pickle.load(f)

    def is_available(self):
        return os.path.exists(self.model_file)

    def predict(self, board: list) -> int:
        self._load()
        feat_names = ["tl","tm","tr","ml","mm","mr","bl","bm","br",
                      "n_x","n_o","n_blank","threats_x","threats_o","blocked_x","blocked_o"]
        df = pd.DataFrame([_extract_features(board)], columns=feat_names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return int(self._model.predict(df)[0])


class KNNClassifier_(BaseClassifier):
    name        = "K-NN"
    description = "Classificador por vizinhos mais próximos usando 16 features de tabuleiro."
    model_file  = os.path.join(OUTPUT_DIR, "knn_model.pkl")

    def __init__(self):
        self._model = None

    def is_available(self):
        return os.path.exists(self.model_file)

    def predict(self, board: list) -> int:
        if self._model is None:
            with open(self.model_file, 'rb') as f:
                self._model = pickle.load(f)
        feats = _extract_features(board)
        return int(self._model.predict([feats])[0])


class MLPClassifier_(BaseClassifier):
    name        = "MLP"
    description = "Rede neural multicamadas treinada com backpropagation e solver Adam. Busca automática de topologia e hiperparâmetros via conjunto de validação."
    model_file  = os.path.join(OUTPUT_DIR, "mlp_model.pkl")

    def __init__(self):
        self._model = None

    def is_available(self):
        return os.path.exists(self.model_file)

    def predict(self, board: list) -> int:
        if self._model is None:
            with open(self.model_file, 'rb') as f:
                self._model = pickle.load(f)
        feats = np.array(_extract_features(board)).reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return int(self._model.predict(feats)[0])


class HierarchicalClassifier(BaseClassifier):
    name        = "Agrupamento Hierárquico"
    description = "Clusterização hierárquica adaptada para classificação usando centroides."
    model_file  = os.path.join(OUTPUT_DIR, "hierarchical_model.pkl")

    def __init__(self):
        self._model = None

    def is_available(self):
        return os.path.exists(self.model_file)

    def predict(self, board: list) -> int:
        if self._model is None:
            with open(self.model_file, 'rb') as f:
                self._model = pickle.load(f)
        feats = np.asarray(_extract_features(board), dtype=float)
        centroids = np.asarray(self._model['centroids'], dtype=float)
        distances = np.linalg.norm(centroids - feats, axis=1)
        nearest = int(np.argmin(distances))
        mapping = {int(k): int(v) for k, v in self._model['mapping'].items()}
        return int(mapping[nearest])


class RandomForestClassifier_(BaseClassifier):
    name        = "Random Forest"
    description = "Ensemble supervisionado de várias árvores. Geralmente mais robusto e com melhor generalização que uma única árvore."
    model_file  = os.path.join(OUTPUT_DIR, "random_forest_model.pkl")

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            with open(self.model_file, 'rb') as f:
                self._model = pickle.load(f)

    def is_available(self):
        return os.path.exists(self.model_file)

    def predict(self, board: list) -> int:
        self._load()
        import pandas as pd
        import warnings
        feat_names = ["tl","tm","tr","ml","mm","mr","bl","bm","br",
                      "n_x","n_o","n_blank","threats_x","threats_o","blocked_x","blocked_o"]
        df = pd.DataFrame([_extract_features(board)], columns=feat_names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return int(self._model.predict(df)[0])


# ── Dicionário central — adicione/remova instâncias aqui ─────────────────────
MODEL_REGISTRY: dict[str, BaseClassifier] = {
    "decision_tree": DecisionTreeClassifier_(),
    "random_forest": RandomForestClassifier_(),
    "knn":           KNNClassifier_(),
    "hierarchical":  HierarchicalClassifier(),
    "mlp":           MLPClassifier_(),
}

# ─────────────────────────────────────────────────────────────────────────────
#  LÓGICA DETERMINÍSTICA DO JOGO (ground truth)
# ─────────────────────────────────────────────────────────────────────────────
def ground_truth(board: list) -> int:
    w = _winner(board)
    if w == 'x':
        return 4
    if w == 'o':
        return 3
    if 'b' not in board:
        return 2
    if _has_threat(board, 'x') or _has_threat(board, 'o'):
        return 1
    return 0

# ─────────────────────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def index():
    models_info = [
        {
            "key":         key,
            "name":        m.name,
            "description": m.description,
            "available":   m.is_available(),
        }
        for key, m in MODEL_REGISTRY.items()
    ]

    return render_template_string(HTML_TEMPLATE,
                                  models=models_info,
                                  class_names=CLASS_NAMES)

@app.route('/api/classify', methods=['POST'])
def classify():
    data     = request.get_json()
    board    = data['board']       # list[str], len=9
    model_key = data.get('model', 'decision_tree')

    model = MODEL_REGISTRY.get(model_key)
    if model is None or not model.is_available():
        return jsonify(error="Modelo não disponível"), 400

    try:
      pred = int(model.predict(board))
      truth = ground_truth(board)
      return jsonify(
        prediction=pred,
        prediction_label=CLASS_NAMES[pred],
        ground_truth=truth,
        ground_truth_label=CLASS_NAMES[truth],
        correct=(pred == truth),
      )
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/api/models')
def list_models():
  models = [
    {"key": k, "name": m.name, "available": m.is_available()}
    for k, m in MODEL_REGISTRY.items()
  ]
  models.extend(
    [
      {"key": "model4", "name": "Modelo 4", "available": False},
      {"key": "model5", "name": "Modelo 5", "available": False},
    ]
  )
  return jsonify(models)

# ─────────────────────────────────────────────────────────────────────────────
#  HTML + CSS + JS  (template único)
# ─────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jogo da Velha — IA Classifier</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg:        #0a0e1a;
  --surface:   #111827;
  --border:    #1e2d4a;
  --accent:    #00d9ff;
  --accent2:   #ff4d6d;
  --x-color:   #ff4d6d;
  --o-color:   #00d9ff;
  --text:      #e2e8f0;
  --muted:     #64748b;
  --success:   #22c55e;
  --warning:   #f59e0b;
  --error:     #ef4444;
  --cell-size: 110px;
  --radius:    8px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Space Mono', monospace;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  overflow-x: hidden;
}

/* grid background */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background-image:
    linear-gradient(rgba(0,217,255,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,217,255,.04) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}

/* ── Header ── */
header {
  position: relative; z-index: 1;
  border-bottom: 1px solid var(--border);
  padding: 18px 32px;
  display: flex; align-items: center; justify-content: space-between;
}
header h1 {
  font-family: 'Syne', sans-serif;
  font-weight: 800; font-size: 1.3rem; letter-spacing: .08em;
  text-transform: uppercase;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.badge {
  font-size: .65rem; letter-spacing: .12em; text-transform: uppercase;
  border: 1px solid var(--border); padding: 4px 10px; border-radius: 20px;
  color: var(--muted);
}

/* ── Layout ── */
main {
  position: relative; z-index: 1;
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 360px;
  gap: 0;
  max-width: 1000px;
  margin: 0 auto;
  width: 100%;
  padding: 32px 16px;
}

/* ── Game column ── */
.game-col {
  display: flex; flex-direction: column; align-items: center;
  gap: 24px;
  padding-right: 40px;
}

.turn-indicator {
  font-size: .78rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--muted);
  display: flex; align-items: center; gap: 10px;
}
.turn-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--accent);
  animation: pulse 1.2s infinite;
}
.turn-dot.o { background: var(--o-color); }
.turn-dot.x { background: var(--x-color); }
@keyframes pulse {
  0%,100% { opacity: 1; transform: scale(1); }
  50% { opacity: .5; transform: scale(.8); }
}

/* ── Board ── */
.board-row {
  display: grid;
  grid-template-columns: 240px 1fr;
  column-gap: 24px;
  align-items: start;
  width: 100%;
  align-self: stretch;
}

.difficulty-col {
  width: 240px;
  max-width: 240px;
  margin-left: clamp(-160px, -8vw, -80px);
}

.board-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
  min-width: 0;
}

.board {
  display: grid;
  grid-template-columns: repeat(3, var(--cell-size));
  grid-template-rows: repeat(3, var(--cell-size));
  gap: 0;
  position: relative;
}

/* SVG grid lines */
.board-lines {
  position: absolute; inset: 0;
  pointer-events: none;
}

.cell {
  width: var(--cell-size); height: var(--cell-size);
  display: flex; align-items: center; justify-content: center;
  cursor: pointer;
  font-family: 'Syne', sans-serif;
  font-weight: 800; font-size: 3rem;
  position: relative;
  transition: background .15s;
  z-index: 1;
}
.cell:hover:not(.taken):not(.disabled) {
  background: rgba(0,217,255,.06);
}
.cell.taken   { cursor: default; }
.cell.disabled { cursor: not-allowed; opacity: .6; pointer-events: none; }

.cell .symbol {
  opacity: 0;
  transform: scale(.4) rotate(-20deg);
  transition: opacity .25s, transform .25s cubic-bezier(.34,1.56,.64,1);
}
.cell .symbol.show {
  opacity: 1;
  transform: scale(1) rotate(0);
}
.cell .symbol.x { color: var(--x-color); }
.cell .symbol.o { color: var(--o-color); }

/* win highlight */
.cell.win-cell::after {
  content: '';
  position: absolute; inset: 6px;
  border-radius: 6px;
  background: rgba(255,255,255,.07);
  animation: winfade .4s ease forwards;
}
@keyframes winfade { from { opacity: 0; } to { opacity: 1; } }

/* ── Controls ── */
.controls { display: flex; gap: 12px; justify-content: center; }

.btn {
  font-family: 'Space Mono', monospace;
  font-size: .75rem; letter-spacing: .1em; text-transform: uppercase;
  padding: 10px 22px;
  border-radius: var(--radius);
  border: 1px solid;
  cursor: pointer;
  transition: background .15s, transform .1s;
}
.btn:active { transform: scale(.97); }
.btn-primary {
  background: var(--accent); color: var(--bg);
  border-color: var(--accent);
}
.btn-primary:hover { background: #00b8d9; }
.btn-ghost {
  background: transparent; color: var(--muted);
  border-color: var(--border);
}
.btn-ghost:hover { border-color: var(--muted); color: var(--text); }

/* ── Side panel ── */
.side-col {
  border-left: 1px solid var(--border);
  padding-left: 32px;
  display: flex; flex-direction: column; gap: 28px;
}

.panel-section { display: flex; flex-direction: column; gap: 10px; }
.panel-label {
  font-size: .6rem; letter-spacing: .18em; text-transform: uppercase;
  color: var(--muted); border-bottom: 1px solid var(--border);
  padding-bottom: 6px;
}

/* Model selector */
.model-select {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
  font-family: 'Space Mono', monospace;
  font-size: .78rem;
  padding: 9px 12px;
  width: 100%;
  cursor: pointer;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%2364748b' stroke-width='1.5' fill='none'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 12px center;
  padding-right: 32px;
}
.model-select:disabled {
  opacity: .4; cursor: not-allowed;
}
option[disabled] { color: var(--muted); font-style: italic; }

.model-desc {
  font-size: .68rem; color: var(--muted); line-height: 1.5;
}

/* AI State card */
.state-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  display: flex; flex-direction: column; gap: 8px;
  transition: border-color .3s;
}
.state-card.correct   { border-color: var(--success); }
.state-card.incorrect { border-color: var(--error); }

.state-prediction {
  font-family: 'Syne', sans-serif;
  font-size: 1rem; font-weight: 700;
  color: var(--accent);
  min-height: 24px;
}
.state-meta {
  display: grid; grid-template-columns: auto 1fr; gap: 4px 10px;
  font-size: .68rem;
}
.state-meta .key   { color: var(--muted); }
.state-meta .val   { color: var(--text); }
.state-meta .val.ok  { color: var(--success); }
.state-meta .val.err { color: var(--error); }

.state-badge {
  display: inline-block;
  font-size: .6rem; letter-spacing: .12em; text-transform: uppercase;
  padding: 3px 8px; border-radius: 4px; align-self: flex-start;
}
.state-badge.ok  { background: rgba(34,197,94,.15); color: var(--success); }
.state-badge.err { background: rgba(239,68,68,.15);  color: var(--error); }
.state-badge.wait { background: rgba(100,116,139,.15); color: var(--muted); }

/* Game message */
.game-message {
  padding: 12px 14px;
  border-radius: var(--radius);
  font-size: .8rem; line-height: 1.5;
  border: 1px solid var(--border);
  min-height: 52px;
  color: var(--text);
  background: var(--surface);
}
.game-message.info    { border-color: var(--accent);   color: var(--accent); }
.game-message.success { border-color: var(--success);  color: var(--success); }
.game-message.error   { border-color: var(--error);    color: var(--error); }
.game-message.warning { border-color: var(--warning);  color: var(--warning); }

/* Accuracy tracker */
.accuracy-bar-wrap {
  display: flex; flex-direction: column; gap: 6px;
}
.accuracy-bar {
  height: 6px; background: var(--border); border-radius: 3px; overflow: hidden;
}
.accuracy-bar-fill {
  height: 100%; border-radius: 3px;
  background: linear-gradient(90deg, var(--success), var(--accent));
  transition: width .4s ease;
  width: 0%;
}
.accuracy-stats {
  display: flex; justify-content: space-between;
  font-size: .7rem; color: var(--muted);
}
.accuracy-value {
  font-family: 'Syne', sans-serif;
  font-size: 1.5rem; font-weight: 800;
  color: var(--text);
  letter-spacing: -.02em;
}
.stats-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.stat-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 12px;
  display: flex; flex-direction: column; gap: 2px;
}
.stat-box .s-label { font-size: .6rem; letter-spacing: .12em;
  text-transform: uppercase; color: var(--muted); }
.stat-box .s-val { font-family: 'Syne', sans-serif;
  font-size: 1.1rem; font-weight: 700; }
.stat-box.green .s-val { color: var(--success); }
.stat-box.red   .s-val { color: var(--error);   }

/* Log */
.log-list {
  max-height: 140px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 4px;
  padding-right: 4px;
}
.log-list::-webkit-scrollbar { width: 3px; }
.log-list::-webkit-scrollbar-thumb { background: var(--border); }
.log-entry {
  font-size: .65rem; color: var(--muted); line-height: 1.4;
  padding: 3px 6px;
  border-left: 2px solid var(--border);
}
.log-entry.ok  { border-color: var(--success); color: var(--text); }
.log-entry.err { border-color: var(--error);   color: var(--error); }
.log-entry.sys { border-color: var(--accent);  color: var(--accent); font-style: italic; }

/* Responsive */
@media (max-width: 700px) {
  main { grid-template-columns: 1fr; padding: 16px; }
  .game-col { padding-right: 0; }
  .board-row { grid-template-columns: 1fr; row-gap: 16px; justify-items: center; }
  .difficulty-col { width: 100%; max-width: 360px; margin-left: 0; }
  .side-col { border-left: none; border-top: 1px solid var(--border);
              padding-left: 0; padding-top: 28px; margin-top: 8px; }
  :root { --cell-size: 90px; }
}
</style>
</head>
<body>

<header>
  <h1>Jogo da Velha ✕ IA</h1>
  <span class="badge">Classificador de Estado</span>
</header>

<main>
  <!-- ── Coluna do jogo ── -->
  <div class="game-col">

    <div class="board-row">

      <!-- Dificuldade (separado do painel lateral) -->
      <div class="difficulty-col">
        <div class="panel-section">
          <div class="panel-label">Dificuldade (jogada do O)</div>
          <select class="model-select" id="difficultySelect" onchange="onDifficultyChange()">
            <option value="default" selected>Padrão (aleatório)</option>
            <option disabled>────────</option>
            <option value="easy">Fácil</option>
            <option value="medium">Médio</option>
            <option value="hard">Difícil</option>
            <option value="impossible">Impossível</option>
          </select>
          <div class="model-desc" id="difficultyDesc">
            <span class="d-desc" data-key="default" style="display:block">O joga aleatoriamente</span>
            <span class="d-desc" data-key="easy" style="display:none">O faz jogadas fracas e evita lances óbvios</span>
            <span class="d-desc" data-key="medium" style="display:none">O tenta vencer quando possível, caso contrário, joga aleatório</span>
            <span class="d-desc" data-key="hard" style="display:none">O tenta vencer, bloquear ameaças e prefere centro/cantos</span>
            <span class="d-desc" data-key="impossible" style="display:none">O joga de forma ótima (minimax). Não perde.</span>
          </div>
        </div>
      </div>

      <div class="board-area">
        <div class="turn-indicator">
          <div class="turn-dot" id="turnDot"></div>
          <span id="turnText">Aguardando...</span>
        </div>

        <!-- Tabuleiro -->
        <div class="board" id="board">
          <!-- linhas SVG -->
          <svg class="board-lines" viewBox="0 0 330 330" xmlns="http://www.w3.org/2000/svg">
            <line x1="110" y1="10"  x2="110" y2="320" stroke="#1e2d4a" stroke-width="1.5"/>
            <line x1="220" y1="10"  x2="220" y2="320" stroke="#1e2d4a" stroke-width="1.5"/>
            <line x1="10"  y1="110" x2="320" y2="110" stroke="#1e2d4a" stroke-width="1.5"/>
            <line x1="10"  y1="220" x2="320" y2="220" stroke="#1e2d4a" stroke-width="1.5"/>
          </svg>
          <!-- células injetadas via JS -->
        </div>

        <div class="controls">
          <button class="btn btn-primary" onclick="newGame()">↺ Nova Partida</button>
          <button class="btn btn-ghost"   onclick="resetScore()">Zerar Score</button>
        </div>
      </div>
    </div>

  </div>

  <!-- ── Painel lateral ── -->
  <div class="side-col">

    <!-- Modelo -->
    <div class="panel-section">
      <div class="panel-label">Modelo de IA</div>
      <select class="model-select" id="modelSelect" onchange="onModelChange()">
        {% for m in models %}
        <option value="{{ m.key }}"
          {% if not m.available %}disabled{% endif %}>
          {{ m.name }}{% if not m.available %} (indisponível){% endif %}
        </option>
        {% endfor %}
      </select>
      <div class="model-desc" id="modelDesc">
        {% for m in models %}
        <span class="mdesc" data-key="{{ m.key }}"
          style="display:{{ 'block' if loop.first else 'none' }}">
          {{ m.description }}
        </span>
        {% endfor %}
      </div>
    </div>

    <!-- Estado predito -->
    <div class="panel-section">
      <div class="panel-label">Classificação da IA</div>
      <div class="state-card" id="stateCard">
        <div class="state-prediction" id="statePred">—</div>
        <div class="state-meta" id="stateMeta">
          <span class="key">Gabarito:</span><span class="val" id="stateGT">—</span>
          <span class="key">Resultado:</span><span class="val" id="stateResult">—</span>
        </div>
        <span class="state-badge wait" id="stateBadge">Aguardando jogada</span>
      </div>
    </div>

    <!-- Mensagem do jogo -->
    <div class="panel-section">
      <div class="panel-label">Status do Jogo</div>
      <div class="game-message" id="gameMsg">Faça sua jogada para começar.</div>
    </div>

    <!-- Acurácia -->
    <div class="panel-section">
      <div class="panel-label">Acurácia da IA — sessão atual</div>
      <div class="accuracy-value" id="accValue">—</div>
      <div class="accuracy-bar-wrap">
        <div class="accuracy-bar"><div class="accuracy-bar-fill" id="accBar"></div></div>
        <div class="accuracy-stats">
          <span id="accFrac">0/0 predições</span>
          <span id="accPct"></span>
        </div>
      </div>
      <div class="stats-grid">
        <div class="stat-box green">
          <span class="s-label">Acertos</span>
          <span class="s-val" id="statHits">0</span>
        </div>
        <div class="stat-box red">
          <span class="s-label">Erros</span>
          <span class="s-val" id="statErrors">0</span>
        </div>
      </div>
    </div>

    <!-- Log -->
    <div class="panel-section">
      <div class="panel-label">Log de Jogadas</div>
      <div class="log-list" id="logList"></div>
    </div>

  </div>
</main>

<script>
// ─── Estado do jogo ─────────────────────────────────────────────────────────
const POS_NAMES = ['topo-esq','topo-cen','topo-dir',
                   'meio-esq','centro','meio-dir',
                   'baixo-esq','baixo-cen','baixo-dir'];
const CLASS_NAMES = {{ class_names | tojson }};
const LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];

let board       = Array(9).fill('b');
let gameActive  = false;
let humanTurn   = true;   // humano é X, computador é O
let totalHits   = 0;
let totalErrors = 0;
let currentModel = document.getElementById('modelSelect').value;
let currentDifficulty = document.getElementById('difficultySelect')?.value || 'default';

// ─── Inicialização ───────────────────────────────────────────────────────────
function buildBoard() {
  const el = document.getElementById('board');
  // manter o SVG de fundo
  const svg = el.querySelector('svg');
  el.innerHTML = '';
  el.appendChild(svg);
  for (let i = 0; i < 9; i++) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    cell.dataset.idx = i;
    cell.innerHTML = `<span class="symbol" id="sym-${i}"></span>`;
    cell.addEventListener('click', () => humanMove(i));
    el.appendChild(cell);
  }
}

function newGame() {
  board      = Array(9).fill('b');
  gameActive = true;
  humanTurn  = true;
  buildBoard();
  setTurn('x');
  setMsg('Sua vez — jogue em qualquer posição.', 'info');
  resetStateCard();
  log('── nova partida iniciada ──', 'sys');
}

function resetScore() {
  totalHits = 0; totalErrors = 0;
  updateAccuracy();
  log('── score zerado ──', 'sys');
}

// ─── Turno humano ────────────────────────────────────────────────────────────
async function humanMove(idx) {
  if (!gameActive || !humanTurn || board[idx] !== 'b') return;
  applyMove(idx, 'x');
  humanTurn = false;
  // Atualiza a UI imediatamente: agora é a vez do computador.
  setTurn('o');
  await classify();
  if (!gameActive) return;
  setTimeout(computerMove, 600);
}

// ─── Turno computador ────────────────────────────────────────────────────────
async function computerMove() {
  if (!gameActive) return;
  const empty = board.map((v,i) => v==='b' ? i : -1).filter(i => i>=0);
  if (!empty.length) return;

  const idx = chooseComputerMove(board, currentDifficulty);
  applyMove(idx, 'o');
  humanTurn = true;
  await classify();
  if (!gameActive) return;
  setTurn('x');
}

// ─── Dificuldade: escolha de jogada do computador (O) ───────────────────────
function chooseComputerMove(b, difficulty) {
  const empty = b.map((v,i) => v==='b' ? i : -1).filter(i => i>=0);
  if (!empty.length) return -1;

  // Padrão (comportamento atual): aleatório
  if (difficulty === 'default') {
    return empty[Math.floor(Math.random() * empty.length)];
  }

  const oWins = winningMoves(b, 'o');
  const xWins = winningMoves(b, 'x');

  if (difficulty === 'easy') {
    // Evita ganhar e evita bloquear (se houver alternativas), para ficar bem fraco.
    const avoid = new Set([...oWins, ...xWins]);
    const candidates = empty.filter(i => !avoid.has(i));
    const pool = candidates.length ? candidates : empty;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  if (difficulty === 'medium') {
    // Se pode vencer, vence; senão aleatório.
    if (oWins.length) return oWins[Math.floor(Math.random() * oWins.length)];
    return empty[Math.floor(Math.random() * empty.length)];
  }

  if (difficulty === 'hard') {
    // Vence se possível; senão bloqueia ameaça imediata; senão heurística simples.
    if (oWins.length) return oWins[Math.floor(Math.random() * oWins.length)];
    if (xWins.length) return xWins[Math.floor(Math.random() * xWins.length)];

    if (b[4] === 'b') return 4; // centro
    const corners = [0,2,6,8].filter(i => b[i] === 'b');
    if (corners.length) return corners[Math.floor(Math.random() * corners.length)];
    return empty[Math.floor(Math.random() * empty.length)];
  }

  // Impossível: minimax ótimo
  if (difficulty === 'impossible') {
    return minimaxBestMove(b, 'o');
  }

  // Fallback
  return empty[Math.floor(Math.random() * empty.length)];
}

function winningMoves(b, sym) {
  const empty = b.map((v,i) => v==='b' ? i : -1).filter(i => i>=0);
  const wins = [];
  for (const idx of empty) {
    const bb = b.slice();
    bb[idx] = sym;
    if (winnerSymbol(bb) === sym) wins.push(idx);
  }
  return wins;
}

function winnerSymbol(b) {
  for (const [a,c,d] of LINES) {
    if (b[a] !== 'b' && b[a] === b[c] && b[a] === b[d]) return b[a];
  }
  return null;
}

function terminalState(b) {
  const w = winnerSymbol(b);
  if (w) return w;
  if (!b.includes('b')) return 'draw';
  return null;
}

function minimaxBestMove(b, sym) {
  const memo = new Map();
  let bestScore = -Infinity;
  let bestMove = -1;
  const empty = b.map((v,i) => v==='b' ? i : -1).filter(i => i>=0);
  for (const idx of empty) {
    const bb = b.slice();
    bb[idx] = sym;
    const score = minimax(bb, 'x', memo);
    if (score > bestScore) {
      bestScore = score;
      bestMove = idx;
    }
  }
  return bestMove >= 0 ? bestMove : empty[0];
}

function minimax(b, player, memo) {
  const term = terminalState(b);
  // Score simples e exato (evita problemas de cache com poda):
  //  1  = vitória do O
  //  0  = empate
  // -1  = vitória do X
  if (term === 'o') return 1;
  if (term === 'x') return -1;
  if (term === 'draw') return 0;

  const key = b.join('') + '|' + player;
  if (memo.has(key)) return memo.get(key);

  const empty = b.map((v,i) => v==='b' ? i : -1).filter(i => i>=0);

  let value;
  if (player === 'o') {
    value = -Infinity;
    for (const idx of empty) {
      const bb = b.slice();
      bb[idx] = 'o';
      value = Math.max(value, minimax(bb, 'x', memo));
    }
  } else {
    value = Infinity;
    for (const idx of empty) {
      const bb = b.slice();
      bb[idx] = 'x';
      value = Math.min(value, minimax(bb, 'o', memo));
    }
  }

  memo.set(key, value);
  return value;
}

// ─── Aplica jogada no board ──────────────────────────────────────────────────
function applyMove(idx, sym) {
  board[idx] = sym;
  const sp = document.getElementById(`sym-${idx}`);
  sp.textContent = sym.toUpperCase();
  sp.className = `symbol ${sym}`;
  requestAnimationFrame(() => sp.classList.add('show'));
  document.querySelectorAll('.cell')[idx].classList.add('taken');
}

// ─── Classificação via API ───────────────────────────────────────────────────
async function classify() {
  const res = await fetch('/api/classify', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ board, model: currentModel })
  });
  if (!res.ok) { log('Erro na API.', 'err'); return; }
  const data = await res.json();

  const pred    = data.prediction;
  const gt      = data.ground_truth;
  const correct = data.correct;

  // Atualizar card
  document.getElementById('statePred').textContent = data.prediction_label;
  document.getElementById('stateGT').textContent   = data.ground_truth_label;

  const card   = document.getElementById('stateCard');
  const badge  = document.getElementById('stateBadge');
  const resEl  = document.getElementById('stateResult');

  if (correct) {
    card.className  = 'state-card correct';
    badge.className = 'state-badge ok'; badge.textContent = '✓ Correto';
    resEl.className = 'val ok'; resEl.textContent = 'Acerto';
    totalHits++;
    log(`✓ Predição: "${data.prediction_label}" — correto`, 'ok');
  } else {
    card.className  = 'state-card incorrect';
    badge.className = 'state-badge err'; badge.textContent = '✗ Incorreto';
    resEl.className = 'val err'; resEl.textContent = 'Erro';
    totalErrors++;
    log(`✗ Predição: "${data.prediction_label}" | Gabarito: "${data.ground_truth_label}"`, 'err');
  }
  updateAccuracy();

  // ── Lógica de controle do jogo (enunciado) ───────────────────────────────
  const reallyOver = (gt >= 2);        // 2,3,4 = fim real
  const modelSaysOver = (pred >= 2);   // modelo acha que acabou

  if (reallyOver && !modelSaysOver) {
    // IA não detectou o fim → encerrar o jogo
    gameActive = false;
    highlightWinner(gt);
    setMsg(
      `⚠ IA não detectou o fim (disse: "${data.prediction_label}").\n`+
      `Jogo encerrado. Resultado real: ${data.ground_truth_label}.`,
      'error'
    );
    disableBoard();
  } else if (!reallyOver && modelSaysOver) {
    // IA detectou fim incorretamente → continuar jogando
    setMsg(
      `⚠ IA detectou fim incorretamente ("${data.prediction_label}").\n`+
      `Jogo continua — real: "${data.ground_truth_label}".`,
      'warning'
    );
  } else if (reallyOver && modelSaysOver) {
    // ambos concordam: fim de jogo
    gameActive = false;
    highlightWinner(gt);
    let emoji = gt === 4 ? '🎉' : gt === 3 ? '🤖' : '🤝';
    setMsg(`${emoji} ${data.ground_truth_label}! IA classificou corretamente.`, 'success');
    disableBoard();
  }
}

// ─── Utilitários de UI ───────────────────────────────────────────────────────
function highlightWinner(gt) {
  if (gt === 2) return; // empate, sem linha vencedora
  const sym = gt === 4 ? 'x' : 'o';
  for (const [a,b,c] of LINES) {
    if (board[a]===sym && board[b]===sym && board[c]===sym) {
      [a,b,c].forEach(i => document.querySelectorAll('.cell')[i].classList.add('win-cell'));
      break;
    }
  }
}

function disableBoard() {
  document.querySelectorAll('.cell').forEach(c => c.classList.add('disabled'));
}

function setTurn(sym) {
  const dot  = document.getElementById('turnDot');
  const text = document.getElementById('turnText');
  dot.className = `turn-dot ${sym}`;
  text.textContent = sym === 'x' ? 'Sua vez (X)' : 'Computador jogando (O)...';
}

function setMsg(txt, type='') {
  const el = document.getElementById('gameMsg');
  el.textContent = txt;
  el.className = `game-message ${type}`;
}

function resetStateCard() {
  document.getElementById('statePred').textContent   = '—';
  document.getElementById('stateGT').textContent     = '—';
  document.getElementById('stateResult').textContent = '—';
  document.getElementById('stateResult').className   = 'val';
  document.getElementById('stateCard').className     = 'state-card';
  const badge = document.getElementById('stateBadge');
  badge.className = 'state-badge wait'; badge.textContent = 'Aguardando jogada';
}

function updateAccuracy() {
  const total = totalHits + totalErrors;
  const pct   = total > 0 ? (totalHits / total * 100) : 0;
  document.getElementById('accValue').textContent = total > 0 ? `${pct.toFixed(1)}%` : '—';
  document.getElementById('accBar').style.width   = `${pct}%`;
  document.getElementById('accFrac').textContent  = `${total} predições`;
  document.getElementById('accPct').textContent   = total > 0 ? `${pct.toFixed(1)}%` : '';
  document.getElementById('statHits').textContent   = totalHits;
  document.getElementById('statErrors').textContent = totalErrors;
}

function log(msg, type='') {
  const list  = document.getElementById('logList');
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  const ts = new Date().toLocaleTimeString('pt-BR',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  entry.textContent = `[${ts}] ${msg}`;
  list.prepend(entry);
  if (list.children.length > 40) list.lastChild.remove();
}

function onModelChange() {
  currentModel = document.getElementById('modelSelect').value;
  document.querySelectorAll('.mdesc').forEach(el => {
    el.style.display = el.dataset.key === currentModel ? 'block' : 'none';
  });
  log(`Modelo alterado para: ${currentModel}`, 'sys');
}

function onDifficultyChange() {
  currentDifficulty = document.getElementById('difficultySelect').value;
  document.querySelectorAll('.d-desc').forEach(el => {
    el.style.display = el.dataset.key === currentDifficulty ? 'block' : 'none';
  });
  log(`Dificuldade alterada para: ${currentDifficulty}`, 'sys');
}

// ─── Start ───────────────────────────────────────────────────────────────────
buildBoard();
newGame();
</script>
</body>
</html>
"""

if __name__ == '__main__':
    print("="*55)
    print("  Jogo da Velha — IA Classifier")
    print("="*55)
    print()
    print("Modelos disponíveis:")
    for key, m in MODEL_REGISTRY.items():
        status = "✓" if m.is_available() else "✗ (arquivo .pkl não encontrado)"
        print(f"  [{status}] {m.name}")
    print()
    port = int(os.environ.get("PORT", "5000"))
    print(f"Acesse: http://localhost:{port}")
    print()
    app.run(debug=True, port=port)