"""run_all.py — Treina (se necessário) e sobe o frontend Flask.

Uso:
  python3 run_all.py            # treina apenas se faltar o .pkl e inicia o servidor
  python3 run_all.py --train    # força re-treinar e inicia o servidor
  python3 run_all.py --skip-train  # não treina (erro se faltar modelo) e inicia o servidor
"""

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

    model_path = output_dir / "decision_tree_model.pkl"

    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--train", action="store_true", help="força re-treinar o modelo")
    g.add_argument(
        "--skip-train",
        action="store_true",
        help="não treina (falha se o .pkl não existir)",
    )
    args = parser.parse_args()

    if args.train or (not args.skip_train and not model_path.exists()):
        print(f"[run_all] Treinando modelo (saída em {output_dir})...")
        run([sys.executable, str(base_dir / "decision_tree_final.py")], cwd=base_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            "Modelo não encontrado em outputs/. Rode com --train ou execute decision_tree_final.py."
        )

    print("[run_all] Iniciando servidor Flask em http://localhost:5000 ...")
    run([sys.executable, str(base_dir / "frontend.py")], cwd=base_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
