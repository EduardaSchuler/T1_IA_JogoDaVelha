import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "algoritmos"))

from frontend import CLASS_NAMES, MODEL_REGISTRY, ground_truth


class FrontendRequisitosTest(unittest.TestCase):
    def test_ground_truth_classifica_estados_base(self):
        casos = [
            (["b", "b", "b", "b", "x", "b", "b", "b", "b"], 0),
            (["x", "x", "b", "b", "o", "b", "b", "b", "b"], 1),
            (["x", "o", "x", "x", "o", "o", "o", "x", "x"], 2),
            (["o", "o", "o", "x", "x", "b", "b", "b", "b"], 3),
            (["x", "x", "x", "o", "o", "b", "b", "b", "b"], 4),
        ]

        for board, esperado in casos:
            with self.subTest(board=board):
                self.assertEqual(ground_truth(board), esperado)

    def test_class_names_cobrem_as_cinco_classes_do_enunciado(self):
        self.assertEqual(
            CLASS_NAMES,
            {
                0: "Tem Jogo",
                1: "Possibilidade de Fim de Jogo",
                2: "Empate",
                3: "O vence",
                4: "X vence",
            },
        )

    def test_todos_os_cinco_modelos_estao_registrados_e_disponiveis(self):
        self.assertEqual(
            set(MODEL_REGISTRY),
            {"decision_tree", "random_forest", "knn", "hierarchical", "mlp"},
        )
        indisponiveis = [key for key, model in MODEL_REGISTRY.items() if not model.is_available()]
        self.assertEqual(indisponiveis, [])

    def test_api_classify_retorna_predicao_gabarito_e_acerto(self):
        from frontend import app

        board = ["x", "x", "x", "o", "o", "b", "b", "b", "b"]
        with app.test_client() as client:
            response = client.post(
                "/api/classify",
                json={"board": board, "model": "mlp"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["ground_truth"], 4)
        self.assertEqual(data["ground_truth_label"], "X vence")
        self.assertIn("prediction", data)
        self.assertIn("prediction_label", data)
        self.assertIn("correct", data)

    def test_template_contem_regra_especial_do_enunciado(self):
        from frontend import HTML_TEMPLATE

        self.assertIn("reallyOver && !modelSaysOver", HTML_TEMPLATE)
        self.assertIn("!reallyOver && modelSaysOver", HTML_TEMPLATE)
        self.assertIn("Jogo encerrado. Resultado real", HTML_TEMPLATE)
        self.assertIn("Jogo continua", HTML_TEMPLATE)


if __name__ == "__main__":
    unittest.main()
