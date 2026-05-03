from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

def run(cmd: list[str], cwd: Path) -> None:
    subprocess.check_call(cmd, cwd=str(cwd))

def main() -> int:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lista de Modelos do Projeto
    modelos = [
        {"script": "decision_tree.py", "pkl": "decision_tree_model.pkl"},
        {"script": "random_forest.py", "pkl": "random_forest_model.pkl"},
        {"script": "knn.py", "pkl": "knn_model.pkl"},
        {"script": "multi_layer_perceptron.py", "pkl": "mlp_model.pkl"},
        {"script": "hierarchical_clustering.py", "pkl": "hierarchical_model.pkl"}
    ]

    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--train", action="store_true", help="força re-treinar todos os modelos")
    g.add_argument(
        "--skip-train",
        action="store_true",
        help="não treina nada e tenta iniciar o servidor direto",
    )
    args = parser.parse_args()

    # Passa por todos os modelos registrados acima
    for modelo in modelos:
        model_path = output_dir / modelo["pkl"]
        script_path = base_dir / modelo["script"]

        # Se pedimos para treinar, ou se o arquivo .pkl não existe
        if args.train or (not args.skip_train and not model_path.exists()):
            # Verifica se o script de treino já foi criado por você
            if script_path.exists():
                print(f"\n[run_all] Treinando modelo via '{modelo['script']}'...")
                try:
                    run([sys.executable, str(script_path)], cwd=base_dir)
                except subprocess.CalledProcessError:
                    print(f"[Erro] Ocorreu um problema ao rodar {modelo['script']}. O servidor continuará com os outros.")
            else:
                if args.train: # Só avisa se o cara forçou o treino
                    print(f"[Aviso] Script '{modelo['script']}' não encontrado. Ignorando.")

    # Iniciar o Frontend
    print("\n[run_all] Iniciando servidor Flask...")
    
    # Ele tenta rodar o frontend.py
    frontend_script = "frontend.py"
        
    try:
        run([sys.executable, str(base_dir / frontend_script)], cwd=base_dir)
    except KeyboardInterrupt:
        print("\n[run_all] Servidor encerrado pelo usuário.")
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())