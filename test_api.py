"""Testes da API (rode com: python test_api.py). Precisa do modelo treinado."""
import app as appmod

c = appmod.app.test_client()

OK = {
    "idade": "19", "genero": "Masculino", "nacionalidade": "Nativo / Local",
    "deslocado": "Não", "qualificacao_mae": "Ensino Médio",
    "qualificacao_pai": "Ensino Médio", "ocupacao": "Apenas Estudante",
    "renda_familiar": "2500", "inadimplente": "Não", "mensalidades_em_dia": "Sim",
    "bolsista": "Não", "curso": "Engenharia Informática / Computação",
    "turno": "Diurno", "aprovadas": "5", "media": "13,0",
}


def check(nome, cond):
    print(("OK   " if cond else "FALHA"), nome)
    assert cond, nome


r = c.get("/api/saude"); check("saude", r.json["modelo_carregado"])
r = c.get("/api/opcoes"); check("opcoes", "curso" in r.json["opcoes"])

r = c.post("/api/prever", json=OK)
check("previsão com os dados da tela (JSON)", r.status_code == 200
      and r.json["classe"] in ("Dropout", "Enrolled", "Graduate")
      and abs(sum(r.json["probabilidades"].values()) - 1) < 1e-3)
print("     ->", r.json["classe"], r.json["probabilidades"])

r = c.post("/prever", data=OK)
check("previsão via form-urlencoded + rota curta", r.status_code == 200)

r = c.post("/api/prever", json={**OK, "idade": 19, "aprovadas": 5, "media": 13.5})
check("números nativos em JSON", r.status_code == 200)

r = c.post("/api/prever", json={**OK, "idade": "abc", "media": "25", "curso": "Medicina",
                                "aprovadas": "2,5", "bolsista": "talvez"})
check("422 com vários erros", r.status_code == 422 and
      set(r.json["detalhes"]) == {"idade", "media", "curso", "aprovadas", "bolsista"})

faltando = {k: v for k, v in OK.items() if k != "turno"}
r = c.post("/api/prever", json=faltando)
check("campo obrigatório ausente", r.status_code == 422 and "turno" in r.json["detalhes"])

r = c.post("/api/prever", json={**OK, "renda_familiar": "-1"})
check("renda negativa", r.status_code == 422)

r = c.post("/api/prever", json={**OK, "idade": "nan", "media": "inf"})
check("nan/inf rejeitados", r.status_code == 422)

r = c.post("/api/prever", json=[1, 2]); check("JSON que não é objeto -> 400", r.status_code == 400)
r = c.post("/api/prever", data="{}", content_type="application/json")
check("corpo vazio -> 400", r.status_code == 400)
r = c.get("/api/prever"); check("GET em rota POST -> 405 JSON", r.status_code == 405)
r = c.get("/nada"); check("404 JSON", r.status_code == 404)
print("\nTodos os testes passaram.")
