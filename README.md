# T1 - Tic Tac Toe com ML
#### Grupo E: João Henrique Pires Bergallo, João Vitor Lichston Machado, Larissa Oliveira da Silva e Maria Eduarda da Silveira Schüler

## Como rodar

### 1) Dependências

Com o ambiente virtual ativo (ex.: `.venv`), instale as dependências do trabalho:

Linux e Mac: `python3 -m pip install -r T1_IA_JogoDaVelha/requirements.txt`

Windows: `python -m pip install -r T1_IA_JogoDaVelha/requirements.txt`

### 2) Treinar e gerar artefatos (modelos/gráficos)

O treino gera os CSVs em `T1_IA_JogoDaVelha/dataset/` e salva o modelo + gráficos em `T1_IA_JogoDaVelha/outputs/`.

Linux e Mac: `python3 T1_IA_JogoDaVelha/decision_tree_final.py`

Windows: `python T1_IA_JogoDaVelha/decision_tree_final.py`

### 3) Subir o frontend (Flask)

Linux e Mac:  `python3 T1_IA_JogoDaVelha/frontend.py`

Windows:  `python T1_IA_JogoDaVelha/frontend.py`

Acesse: `http://localhost:5000`

### Rodar tudo junto (treina se faltar o modelo e inicia o servidor)

Linux e Mac: `python3 T1_IA_JogoDaVelha/run_all.py`

Windows: `python T1_IA_JogoDaVelha/run_all.py`

Para forçar re-treino:

Linux e Mac: `python3 T1_IA_JogoDaVelha/run_all.py --train`

Windows: `python T1_IA_JogoDaVelha/run_all.py --train`

### Enunciado
Neste primeiro trabalho prático da disciplina, você vai construir um sistema de IA para o jogo
da velha em um tabuleiro clássico 3x3. O objetivo da IA não é ser um dos players, mas sim
verificar o estado de jogo. A seguir serão descritas as etapas do trabalho.

<img width="560" height="160" alt="image" src="https://github.com/user-attachments/assets/7e7a8cbf-5d40-4fa6-810d-42d059c313d0" />

Objetivo: A IA que você implementará deve receber como entrada o estado atual de
um tabuleiro do jogo da velha e classificar esse estado em:
- Tem jogo
- Possibilidade de Fim de Jogo
- Empate
- O vence
- X vence

<img width="304" height="116" alt="image" src="https://github.com/user-attachments/assets/4fd2dba1-2ce8-4845-8e81-4bddb0ae0490" />
