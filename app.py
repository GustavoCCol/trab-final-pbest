"""
API back-end (Flask) do classificador de estudantes.

Fluxo: formulário -> POST /api/prever -> validação -> Pipeline Naive Bayes
       -> JSON com a classe prevista e as probabilidades.

Rodar:  python app.py          (http://127.0.0.1:5000)
"""
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from jinja2 import TemplateNotFound

import ml_common as ml  # necessário para o joblib reconstruir o Pipeline

BASE = Path(__file__).parent
MODELO_PATH = Path(os.environ.get("MODELO_PATH", BASE / "modelo" / "modelo_nb.joblib"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "")  # ex.: "*" se o front rodar em outra porta

app = Flask(__name__, static_folder="static", template_folder="templates")
app.json.ensure_ascii = False  # acentos legíveis no JSON

# --------------------------------------------------------------------------
# Carregamento do modelo (uma vez, na inicialização)
# --------------------------------------------------------------------------
modelo = None
erro_modelo = None
try:
    modelo = joblib.load(MODELO_PATH)
except FileNotFoundError:
    erro_modelo = f"Modelo não encontrado em {MODELO_PATH}. Rode: python train_model.py"
except Exception as e:  # versão do scikit-learn diferente, arquivo corrompido etc.
    erro_modelo = f"Falha ao carregar o modelo: {e}"


# --------------------------------------------------------------------------
# Validação
# --------------------------------------------------------------------------
def _numero(valor):
    """Converte '13,0' / '13.5' / 13 em float; levanta ValueError se inválido."""
    if isinstance(valor, bool) or valor is None:
        raise ValueError
    if isinstance(valor, str):
        valor = valor.strip().replace(",", ".")
    n = float(valor)
    if math.isnan(n) or math.isinf(n):
        raise ValueError
    return n


def validar(dados: dict):
    """Retorna (registro_para_o_modelo, erros). `erros` é {campo: mensagem}."""
    erros, reg = {}, {}

    for campo, (mn, mx, inteiro) in ml.LIMITES_NUMERICOS.items():
        bruto = dados.get(campo)
        if bruto is None or (isinstance(bruto, str) and not bruto.strip()):
            erros[campo] = "Campo obrigatório."
            continue
        try:
            n = _numero(bruto)
        except (ValueError, TypeError):
            erros[campo] = "Informe um número válido."
            continue
        if inteiro and n != int(n):
            erros[campo] = "Informe um número inteiro."
        elif not (mn <= n <= mx):
            erros[campo] = f"O valor deve estar entre {mn:g} e {mx:g}."
        else:
            reg[campo] = int(n) if inteiro else n

    for campo, opcoes in ml.OPCOES.items():
        bruto = dados.get(campo)
        if bruto is None or (isinstance(bruto, str) and not bruto.strip()):
            erros[campo] = "Campo obrigatório."
            continue
        valor = ml.resolver_opcao(campo, bruto)
        if valor is None:
            erros[campo] = f"Valor inválido. Opções: {', '.join(opcoes)}."
        else:
            reg[campo] = valor

    # Campos do formulário que não existem na base (não entram no modelo)
    renda = dados.get("renda_familiar")
    if renda not in (None, ""):
        try:
            if _numero(renda) < 0:
                erros["renda_familiar"] = "A renda não pode ser negativa."
        except (ValueError, TypeError):
            erros["renda_familiar"] = "Informe um número válido."

    return reg, erros


# --------------------------------------------------------------------------
# Rotas
# --------------------------------------------------------------------------
@app.get("/")
def index():
    try:
        return render_template("index.html")
    except TemplateNotFound:
        return jsonify(mensagem="API no ar. Coloque o front em templates/index.html.",
                       endpoints=["GET /api/saude", "GET /api/opcoes", "POST /api/prever"])


@app.get("/api/saude")
def saude():
    return jsonify(status="ok" if modelo is not None else "sem_modelo",
                   modelo_carregado=modelo is not None, detalhe=erro_modelo)


@app.get("/api/opcoes")
def opcoes():
    """Valores aceitos em cada campo (o front pode montar os <select> a partir daqui)."""
    return jsonify(opcoes=ml.OPCOES, limites={
        k: {"min": v[0], "max": v[1], "inteiro": v[2]}
        for k, v in ml.LIMITES_NUMERICOS.items()})


@app.post("/api/prever")
@app.post("/prever")  # alias, caso o front use a rota curta
def prever():
    if modelo is None:
        return jsonify(erro="Modelo indisponível.", detalhe=erro_modelo), 503

    dados = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if not isinstance(dados, dict) or not dados:
        return jsonify(erro="Envie os dados em JSON ou como formulário."), 400

    registro, erros = validar(dados)
    if erros:
        return jsonify(erro="Dados inválidos.", detalhes=erros), 422

    X = pd.DataFrame([registro])[ml.FEATURES]
    probs = modelo.predict_proba(X)[0]
    classes = list(modelo.classes_)
    i = int(probs.argmax())
    previsto = classes[i]

    return jsonify(
        classe=previsto,
        rotulo=ml.ROTULOS_CLASSES.get(previsto, previsto),
        probabilidade=round(float(probs[i]), 4),
        probabilidades={c: round(float(p), 4) for c, p in zip(classes, probs)},
        entrada=registro,
    )


# --------------------------------------------------------------------------
# Erros em JSON e CORS opcional
# --------------------------------------------------------------------------
@app.errorhandler(404)
def nao_encontrado(_):
    return jsonify(erro="Rota não encontrada."), 404


@app.errorhandler(405)
def metodo_invalido(_):
    return jsonify(erro="Método não permitido."), 405


@app.errorhandler(500)
def erro_interno(_):
    return jsonify(erro="Erro interno ao processar a requisição."), 500


@app.after_request
def cors(resp):
    if CORS_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = CORS_ORIGINS
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
