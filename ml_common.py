"""
Código compartilhado entre o treinamento (train_model.py) e a API (app.py).

Aqui ficam:
  * as categorias aceitas em cada campo do formulário;
  * a conversão dos códigos numéricos da base da UCI para essas categorias;
  * a construção do Pipeline (pré-processamento + Naive Bayes).

IMPORTANTE: o arquivo .joblib referencia funções deste módulo. Por isso o
app.py precisa importar este mesmo arquivo (já faz isso).
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

# --------------------------------------------------------------------------
# Alvo
# --------------------------------------------------------------------------
CLASSES = ["Dropout", "Enrolled", "Graduate"]
ROTULOS_CLASSES = {
    "Dropout": "Evasão / Desistência",
    "Enrolled": "Matriculado / Em Curso",
    "Graduate": "Concluinte / Formado",
}

# --------------------------------------------------------------------------
# Campos do formulário e categorias aceitas
# --------------------------------------------------------------------------
NIVEIS_ESCOLARIDADE = [
    "Ensino Fundamental",
    "Ensino Médio",
    "Ensino Superior",
    "Pós-Graduação",
    "Desconhecido",
]
CURSOS_FORM = [
    "Engenharia Informática / Computação",
    "Gestão / Administração",
    "Enfermagem",
    "Agronomia",
    "Jornalismo e Comunicação",
    "Serviço Social",
]
CURSO_OUTRO = "Outro"  # só existe no treino (cursos da base que não estão no formulário)

# Opções aceitas pela API em cada campo categórico.
OPCOES = {
    "genero": ["Masculino", "Feminino"],
    "nacionalidade": ["Nativo / Local", "Estrangeiro"],
    "deslocado": ["Não", "Sim"],
    "qualificacao_mae": NIVEIS_ESCOLARIDADE,
    "qualificacao_pai": NIVEIS_ESCOLARIDADE,
    "inadimplente": ["Não", "Sim"],
    "mensalidades_em_dia": ["Sim", "Não"],
    "bolsista": ["Não", "Sim"],
    "curso": CURSOS_FORM,
    "turno": ["Diurno", "Noturno"],
}

# Sinônimos extras aceitos (além do próprio rótulo). Chave = texto normalizado.
SINONIMOS = {
    "nacionalidade": {
        "estrangeiro": "Estrangeiro",
        "internacional": "Estrangeiro",
        "estrangeirointernacional": "Estrangeiro",
        "nativo": "Nativo / Local",
        "local": "Nativo / Local",
    },
    "turno": {"noite": "Noturno", "dia": "Diurno", "vespertino": "Diurno"},
    "curso": {
        "engenhariainformatica": "Engenharia Informática / Computação",
        "gestao": "Gestão / Administração",
        "administracao": "Gestão / Administração",
        "jornalismo": "Jornalismo e Comunicação",
    },
}
SIM_NAO = {
    "sim": "Sim", "s": "Sim", "true": "Sim", "1": "Sim", "yes": "Sim", "y": "Sim",
    "nao": "Não", "n": "Não", "false": "Não", "0": "Não", "no": "Não",
}

# Campos numéricos: (mínimo, máximo, inteiro?)
LIMITES_NUMERICOS = {
    "idade": (14, 100, True),        # idade na matrícula
    "aprovadas": (0, 30, True),      # disciplinas aprovadas no 1º semestre
    "media": (0.0, 20.0, False),     # média das notas (escala 0-20)
}

# Campos que o formulário envia mas que NÃO existem na base da UCI.
# São validados (se vierem), porém não entram no modelo.
CAMPOS_FORA_DO_MODELO = ["renda_familiar", "ocupacao"]

COLUNAS_NUM = ["idade", "aprovadas", "media"]
COLUNAS_CAT = list(OPCOES.keys())
FEATURES = COLUNAS_NUM + COLUNAS_CAT

# Categorias usadas pelo OrdinalEncoder (curso ganha "Outro" para o treino).
CATEGORIAS_MODELO = {c: list(v) for c, v in OPCOES.items()}
CATEGORIAS_MODELO["curso"] = CURSOS_FORM + [CURSO_OUTRO]

# Faixas (bins) das variáveis numéricas, já que o CategoricalNB trabalha
# com variáveis categóricas. np.digitize devolve o índice da faixa.
CORTES = {
    "idade": [19, 21, 26, 36],             # <19 | 19-20 | 21-25 | 26-35 | 36+
    "aprovadas": [1, 3, 5, 6, 7],          # 0 | 1-2 | 3-4 | 5 | 6 | 7+
    "media": [0.5, 11.0, 12.0, 13.0, 14.0, 15.0],  # 0(sem nota) | <11 | ... | 15+
}


def normalizar(texto) -> str:
    """minúsculas, sem acentos e só letras/números ('Não' -> 'nao')."""
    s = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolver_opcao(campo: str, valor):
    """Devolve o rótulo canônico de `valor` para o campo, ou None se inválido."""
    if valor is None or isinstance(valor, (dict, list)):
        return None
    chave = normalizar(valor)
    if not chave:
        return None
    for opcao in OPCOES[campo]:
        if normalizar(opcao) == chave:
            return opcao
    if campo in SINONIMOS and chave in SINONIMOS[campo]:
        return SINONIMOS[campo][chave]
    if set(OPCOES[campo]) == {"Sim", "Não"} and chave in SIM_NAO:
        return SIM_NAO[chave]
    return None


# --------------------------------------------------------------------------
# Base da UCI (id 697) -> colunas do formulário
# --------------------------------------------------------------------------
_QUALIF = {
    "Pós-Graduação": {4, 5, 43, 44},
    "Ensino Superior": {2, 3, 6, 39, 40, 41, 42},
    "Ensino Médio": {1, 9, 10, 12, 13, 14, 18, 20, 22, 25, 27, 31, 33},
    "Ensino Fundamental": {11, 19, 26, 29, 30, 37, 38},
    "Desconhecido": {34, 35, 36},
}
_COD_QUALIF = {cod: nivel for nivel, cods in _QUALIF.items() for cod in cods}

_CURSOS = {
    9119: "Engenharia Informática / Computação",
    9147: "Gestão / Administração", 9991: "Gestão / Administração",
    9500: "Enfermagem",
    9003: "Agronomia",
    9773: "Jornalismo e Comunicação",
    9238: "Serviço Social", 8014: "Serviço Social",
}


def _col(df, nome):
    """Acha a coluna ignorando espaços/tabs no nome (a UCI tem um '\\t' perdido)."""
    for c in df.columns:
        if str(c).strip().lower() == nome.lower():
            return c
    raise KeyError(f"Coluna '{nome}' não encontrada no CSV.")


def preparar_dataset(df: pd.DataFrame):
    """Converte o DataFrame bruto da UCI em (X, y) com as colunas do formulário."""
    sim_nao = lambda s: s.map({1: "Sim", 0: "Não"})  # noqa: E731
    X = pd.DataFrame({
        "idade": pd.to_numeric(df[_col(df, "Age at enrollment")]),
        "aprovadas": pd.to_numeric(df[_col(df, "Curricular units 1st sem (approved)")]),
        "media": pd.to_numeric(df[_col(df, "Curricular units 1st sem (grade)")]),
        "genero": df[_col(df, "Gender")].map({1: "Masculino", 0: "Feminino"}),
        "nacionalidade": df[_col(df, "Nacionality")].map(
            lambda c: "Nativo / Local" if c == 1 else "Estrangeiro"),
        "deslocado": sim_nao(df[_col(df, "Displaced")]),
        "qualificacao_mae": df[_col(df, "Mother's qualification")].map(_COD_QUALIF),
        "qualificacao_pai": df[_col(df, "Father's qualification")].map(_COD_QUALIF),
        "inadimplente": sim_nao(df[_col(df, "Debtor")]),
        "mensalidades_em_dia": sim_nao(df[_col(df, "Tuition fees up to date")]),
        "bolsista": sim_nao(df[_col(df, "Scholarship holder")]),
        "curso": df[_col(df, "Course")].map(_CURSOS).fillna(CURSO_OUTRO),
        "turno": df[_col(df, "Daytime/evening attendance")].map({1: "Diurno", 0: "Noturno"}),
    })
    # códigos de escolaridade fora da tabela viram "Desconhecido"
    for c in ("qualificacao_mae", "qualificacao_pai"):
        X[c] = X[c].fillna("Desconhecido")
    y = df[_col(df, "Target")].astype(str).str.strip()
    ok = X.notna().all(axis=1) & y.isin(CLASSES)
    return X[ok][FEATURES].reset_index(drop=True), y[ok].reset_index(drop=True)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def discretizar(X):
    """Transforma idade, aprovadas e média em índices de faixa (inteiros)."""
    X = np.asarray(X, dtype=float)
    cols = [np.digitize(X[:, i], CORTES[nome]) for i, nome in enumerate(COLUNAS_NUM)]
    return np.column_stack(cols)


def para_inteiro(X):
    return np.asarray(X).astype(int)


def construir_pipeline(alpha: float = 1.0) -> Pipeline:
    pre = ColumnTransformer([
        ("num", FunctionTransformer(discretizar), COLUNAS_NUM),
        ("cat", OrdinalEncoder(
            categories=[CATEGORIAS_MODELO[c] for c in COLUNAS_CAT]), COLUNAS_CAT),
    ])
    n_categorias = [len(CORTES[c]) + 1 for c in COLUNAS_NUM] + \
                   [len(CATEGORIAS_MODELO[c]) for c in COLUNAS_CAT]
    return Pipeline([
        ("pre", pre),
        ("inteiro", FunctionTransformer(para_inteiro)),
        ("nb", CategoricalNB(alpha=alpha, min_categories=n_categorias)),
    ])
