"""
Treina o Naive Bayes (CategoricalNB) com a base "Predict Students' Dropout and
Academic Success" (UCI, id 697), avalia e salva o modelo em .joblib.

Uso:
    python train_model.py                       # lê dados/data.csv
    python train_model.py --csv outro.csv       # outro caminho
    python train_model.py --baixar              # baixa via ucimlrepo (precisa de internet)
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

from ml_common import CLASSES, construir_pipeline, preparar_dataset

BASE = Path(__file__).parent


def carregar_csv(caminho: Path, baixar: bool) -> pd.DataFrame:
    if baixar and not caminho.exists():
        from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo
        ds = fetch_ucirepo(id=697)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(caminho, sep=";", index=False)
        print(f"Base baixada e salva em {caminho}")
    if not caminho.exists():
        raise SystemExit(
            f"CSV não encontrado em {caminho}.\n"
            "Baixe a base em https://archive.ics.uci.edu/dataset/697 e coloque o "
            "data.csv em dados/, ou rode com --baixar.")
    return pd.read_csv(caminho, sep=";")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(BASE / "dados" / "data.csv"))
    ap.add_argument("--saida", default=str(BASE / "modelo"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=1.0, help="suavização de Laplace")
    ap.add_argument("--baixar", action="store_true")
    a = ap.parse_args()

    df = carregar_csv(Path(a.csv), a.baixar)
    X, y = preparar_dataset(df)
    print(f"Observações usadas: {len(X)} | Preditoras: {X.shape[1]}")
    print("Distribuição das classes:\n", y.value_counts().to_string(), "\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=a.test_size, random_state=a.seed, stratify=y)

    modelo = construir_pipeline(alpha=a.alpha).fit(X_tr, y_tr)
    pred = modelo.predict(X_te)

    metricas = {
        "acuracia": accuracy_score(y_te, pred),
        "precisao_macro": precision_score(y_te, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_te, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_te, pred, average="macro", zero_division=0),
        "matriz_confusao": {"classes": CLASSES,
                            "valores": confusion_matrix(y_te, pred, labels=CLASSES).tolist()},
        "relatorio": classification_report(y_te, pred, labels=CLASSES,
                                           output_dict=True, zero_division=0),
        "n_treino": len(X_tr), "n_teste": len(X_te),
    }
    print(classification_report(y_te, pred, labels=CLASSES, zero_division=0))
    print("Matriz de confusão (linhas = real, colunas = previsto):")
    print(pd.DataFrame(metricas["matriz_confusao"]["valores"],
                       index=CLASSES, columns=CLASSES))

    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, saida / "modelo_nb.joblib")
    (saida / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nModelo salvo em {saida / 'modelo_nb.joblib'}")


if __name__ == "__main__":
    main()
