# T1 - Tic Tac Toe com ML

Grupo E: Joao Henrique Pires Bergallo, Larissa Oliveira da Silva e Maria Eduarda da Silveira Schuler

## Objetivo do Projeto

O trabalho implementa uma IA para classificar estados de um tabuleiro classico 3x3 de jogo da velha. A IA nao joga como player principal; ela avalia o estado atual do jogo e retorna uma das cinco classes exigidas no enunciado:

| Classe | Descricao |
|---|---|
| 0 - Tem Jogo | O jogo ainda esta em andamento, sem ameaca imediata de termino. |
| 1 - Possibilidade de Fim de Jogo | Existe pelo menos uma linha com duas pecas iguais e uma casa vazia. |
| 2 - Empate | O tabuleiro esta cheio e nao ha vencedor. |
| 3 - O vence | O jogador O completou uma linha, coluna ou diagonal. |
| 4 - X vence | O jogador X completou uma linha, coluna ou diagonal. |

## Estrutura do Projeto

```text
T1_IA_JogoDaVelha/
|-- algoritmos/
|   |-- decision_tree.py
|   |-- random_forest.py
|   |-- knn.py
|   |-- hierarchical_clustering.py
|   |-- multi_layer_perceptron.py
|   |-- frontend.py
|   |-- run_all.py
|-- dataset/
|   |-- tic-tac-toe.data
|   |-- tic-tac-toe.names
|   |-- dataset_treino.csv
|   |-- dataset_validacao.csv
|   |-- dataset_teste.csv
|-- outputs/
|   |-- *_model.pkl
|   |-- *_results.json
|-- tests/
|   |-- test_frontend_requisitos.py
|-- README.md
|-- Documentacao/
|   |-- requirements.txt
```

## Dataset e Preparacao

O dataset de referencia foi o Tic-Tac-Toe Endgame da UCI, armazenado na pasta `dataset` como `tic-tac-toe.data`, `tic-tac-toe.names` e `Index`. Esse dataset original descreve estados finais possiveis, mas o problema do trabalho exige tambem estados intermediarios como `Tem Jogo` e `Possibilidade de Fim de Jogo`. Por isso, o dataset final usado no treinamento foi adequado por simulacao de partidas e regras deterministicas.

Passos executados:

1. Foram simuladas partidas aleatorias de jogo da velha.
2. Cada estado gerado foi classificado pela regra deterministica do projeto.
3. Estados repetidos foram removidos usando conjunto de tabuleiros unicos.
4. As casas foram codificadas como `b=0`, `o=1`, `x=2`.
5. Foram adicionadas features derivadas: quantidade de X, O, casas vazias, ameacas de X/O e linhas bloqueadas de X/O.
6. Foi feita amostragem por classe para reduzir desbalanceamento quando havia muitas amostras disponiveis.
7. A classe `Empate` permaneceu pequena porque ha menos estados unicos gerados que atendem a essa condicao.

Distribuicao final documentada nos CSVs:

| Classe | Total |
|---|---:|
| Tem Jogo | 677 |
| Possibilidade de Fim de Jogo | 1500 |
| Empate | 16 |
| O vence | 316 |
| X vence | 626 |
| **Total** | **3135** |

## Divisao do Dataset

A divisao fisica foi feita em tres arquivos CSV fixos, mantendo os mesmos conjuntos para todos os algoritmos:

| Arquivo | Uso | Amostras | Observacao |
|---|---|---:|---|
| `dataset_treino.csv` | Treinamento | 1881 | 60% do dataset final |
| `dataset_validacao.csv` | Escolha de hiperparametros | 627 | 20% do dataset final |
| `dataset_teste.csv` | Avaliacao final | 627 | 20% do dataset final |

A divisao usa `random_state=42` e estratificacao para manter a proporcao das classes. Depois de escolher os melhores parametros na validacao, os modelos finais sao treinados com treino + validacao e avaliados no teste.

## Features Utilizadas

O vetor de entrada possui 16 features:

| Feature | Descricao |
|---|---|
| `tl`, `tm`, `tr`, `ml`, `mm`, `mr`, `bl`, `bm`, `br` | Posicoes do tabuleiro codificadas como 0, 1 ou 2. |
| `n_x` | Quantidade de X no tabuleiro. |
| `n_o` | Quantidade de O no tabuleiro. |
| `n_blank` | Quantidade de casas vazias. |
| `threats_x` | Linhas com 2 X e 1 vazio. |
| `threats_o` | Linhas com 2 O e 1 vazio. |
| `blocked_x` | Linhas com 2 X bloqueadas por O. |
| `blocked_o` | Linhas com 2 O bloqueadas por X. |

## Solucoes de IA

Foram implementados os 5 algoritmos solicitados. Todos usam os mesmos arquivos fisicos de treino, validacao e teste.

| Modelo | Parametros escolhidos | Acuracia | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| MLP | `hidden_layer_sizes=(64,)`, `learning_rate_init=0.01`, `momentum=0.5` | 97.93% | 98.02% | 97.93% | 97.96% |
| Random Forest | `n_estimators=50`, `max_depth=15` | 98.41% | 98.48% | 98.41% | 98.34% |
| k-NN | `n_neighbors=11`, `weights=distance`, `p=1` | 95.06% | 95.21% | 95.06% | 94.98% |
| Arvore de Decisao | `criterion=entropy`, `max_depth=7`, `min_samples_split=2`, `min_samples_leaf=2` | 91.46% | 91.52% | 91.46% | 91.35% |
| Agrupamento Hierarquico | `linkage=ward`, `n_clusters=5` | 52.95% | 42.19% | 52.95% | 46.65% |

### Justificativa dos Algoritmos

- **MLP (Multi-Layer Perceptron):** rede neural supervisionada que aprende relacoes nao lineares entre as features. A topologia escolhida foi entrada com 16 features, uma camada oculta com 64 neuronios e saida multiclasse para as 5 classes.
- **Random Forest:** conjunto de arvores de decisao. Reduz a variancia de uma arvore isolada e apresentou otimo desempenho para o problema.
- **k-NN:** classifica um estado comparando sua proximidade com exemplos conhecidos. Foi ajustado com diferentes valores de `k`, ponderacao e distancia.
- **Arvore de Decisao:** modelo interpretavel baseado em regras. Foi ajustado com criterio, profundidade e limites minimos de amostras para evitar overfitting.
- **Agrupamento Hierarquico:** algoritmo originalmente nao supervisionado. Foi adaptado para classificacao mapeando clusters para a classe majoritaria e usando centroides para predizer novos exemplos; por isso teve desempenho inferior aos supervisionados.

O melhor modelo pelos resultados de teste foi a **MLP**, seguida de perto pela **Random Forest**. Para uso no frontend, ambos sao adequados; a MLP foi a melhor em acuracia/F1, enquanto a Random Forest tambem e robusta e simples de explicar.

## Frontend

O frontend foi implementado em Flask no arquivo `algoritmos/frontend.py`. Ele atende aos requisitos principais:

- Permite uma partida 3x3 entre humano e computador.
- O humano joga como X.
- O computador joga como O e possui modo padrao aleatorio.
- A cada jogada, o frontend chama `/api/classify` e consulta o modelo selecionado.
- Exibe a classificacao da IA, o gabarito deterministico, acerto/erro e status do jogo.
- Contabiliza acertos, erros e acuracia da sessao atual.
- Permite selecionar entre os modelos disponiveis: Arvore de Decisao, Random Forest, k-NN, Agrupamento Hierarquico e MLP.

### Regra Especial do Enunciado

O requisito foi verificado no codigo do frontend:

- Se o jogo realmente acabou, mas a IA nao detectou fim (`reallyOver && !modelSaysOver`), o frontend encerra a partida e informa o erro da IA.
- Se a IA detectou fim incorretamente, mas o jogo ainda nao acabou (`!reallyOver && modelSaysOver`), o frontend registra o erro e continua a partida.
- Se ambos concordam que acabou, a partida e encerrada normalmente.

Portanto, a regra solicitada esta implementada e foi coberta por teste automatizado de presenca dessa logica no template.

## Criterios de Aceite

| ID | Criterio | Como validar |
|---|---|---|
| CA01 | O sistema classifica as cinco classes do enunciado. | Rodar testes automatizados de `ground_truth` e testar exemplos manuais no frontend. |
| CA02 | O dataset esta documentado e separado fisicamente. | Verificar `dataset_treino.csv`, `dataset_validacao.csv`, `dataset_teste.csv` e a secao Dataset deste README. |
| CA03 | Todos os 5 algoritmos exigidos existem e possuem artefatos. | Rodar `.venv\Scripts\python.exe -B algoritmos\check_models.py` ou `.venv\Scripts\python.exe -m unittest discover -s tests`. |
| CA04 | As metricas exigidas estao registradas. | Conferir `outputs/*_results.json` e a tabela de resultados deste README. |
| CA05 | O frontend permite humano vs maquina. | Subir Flask e jogar uma partida. |
| CA06 | A IA informa estado do jogo a cada turno. | Observar painel de classificacao apos jogadas do humano e do computador. |
| CA07 | O frontend contabiliza acertos, erros e acuracia. | Observar painel de acuracia durante a partida. |
| CA08 | A regra especial de fim incorreto/nao detectado esta implementada. | Conferir testes automatizados e a funcao `classify()` no frontend. |
| CA09 | O projeto roda com o ambiente virtual e dependencias declaradas. | Instalar requirements e executar testes/servidor. |

## Testes Manuais Recomendados

1. Subir o frontend com `.venv\Scripts\python.exe algoritmos\frontend.py`.
2. Acessar `http://localhost:5000`.
3. Confirmar que os 5 modelos aparecem disponiveis no seletor.
4. Jogar uma partida normal e verificar se a classificacao muda a cada turno.
5. Conferir se acertos, erros e acuracia sao atualizados apos cada predicao.
6. Selecionar modelos diferentes e repetir algumas jogadas.
7. Forcar um cenario de vitoria de X ou O e verificar se o jogo encerra quando a IA tambem detecta fim.
8. Usar um modelo menos preciso, como Agrupamento Hierarquico, para observar casos de erro e confirmar se o fluxo especial continua/encerra conforme o gabarito real.

## Testes Automatizados

Foi adicionada uma suite `unittest` em `tests/test_frontend_requisitos.py` cobrindo:

- Classificacao deterministica das cinco classes.
- Nomes das classes exigidas no enunciado.
- Registro e disponibilidade dos cinco modelos.
- Resposta da API `/api/classify` com predicao, gabarito e acerto.
- Presenca da regra especial de controle do jogo no frontend.

Para executar:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```

## Como Rodar

Instalar dependencias:

```powershell
.venv\Scripts\python.exe -m pip install -r "Documentação\requirements.txt"
```

Treinar todos os modelos e iniciar o frontend:

```powershell
.venv\Scripts\python.exe algoritmos\run_all.py --train
```

Iniciar apenas o frontend usando modelos ja existentes:

```powershell
.venv\Scripts\python.exe algoritmos\frontend.py
```

Acessar no navegador:

```text
http://localhost:5000
```

## Ferramentas de IA Utilizadas

Foram usadas ferramentas de IA como apoio ao desenvolvimento e documentacao, especialmente Claude e NotebookLM. Elas foram utilizadas para auxiliar na organizacao textual, revisao de requisitos e apoio na explicacao dos algoritmos, mantendo a implementacao e validacao no codigo do projeto.

## Conclusao Resumida

O projeto atende ao objetivo principal: classificar estados do jogo da velha em cinco classes, comparar cinco solucoes de IA e usar o melhor resultado dentro de um frontend interativo. A principal adaptacao foi transformar o dataset original, voltado a estados finais, em um conjunto mais adequado ao problema com estados intermediarios. O maior desafio foi lidar com o desbalanceamento natural das classes, principalmente `Empate`, e adaptar Agrupamento Hierarquico para uma tarefa supervisionada. Como ganho, o trabalho consolidou o fluxo completo de ML: preparacao de dados, divisao fisica, treinamento, validacao, teste, comparacao de modelos e integracao com frontend.

