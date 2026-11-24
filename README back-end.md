# Back-end – Classificação de Estudantes (Naive Bayes + Flask)

Base: [Predict Students' Dropout and Academic Success (UCI #697)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
· 4.424 observações · alvo: `Dropout`, `Enrolled`, `Graduate`.

## Como rodar

```bash
pip install -r requirements.txt
# coloque o data.csv da UCI em dados/  (ou use: python train_model.py --baixar)
python train_model.py        # treina, mostra as métricas e salva modelo/modelo_nb.joblib
python app.py                # sobe em http://127.0.0.1:5000
python test_api.py           # testes da API
```
Treine e rode a API no mesmo ambiente (mesma versão do scikit-learn).
O front fica em `templates/index.html` (+ `static/`) e é servido em `/`.
Se o front rodar em outra porta: `CORS_ORIGINS="*" python app.py`.

## Endpoints

| Método | Rota | Função |
|---|---|---|
| GET | `/` | Serve o `index.html` |
| GET | `/api/saude` | Verifica se o modelo foi carregado |
| GET | `/api/opcoes` | Valores aceitos em cada campo e limites numéricos |
| POST | `/api/prever` (alias `/prever`) | Valida, prevê e devolve o resultado |

### Requisição (JSON ou formulário)

| Campo | Tela | Regra |
|---|---|---|
| `idade` | Idade na Matrícula | inteiro 14–100 |
| `genero` | Gênero | Masculino / Feminino |
| `nacionalidade` | Nacionalidade | Nativo / Local / Estrangeiro |
| `deslocado` | Mora Fora de Casa | Sim / Não |
| `qualificacao_mae`, `qualificacao_pai` | Qualificação | Ensino Fundamental / Médio / Superior / Pós-Graduação / Desconhecido |
| `inadimplente` | Inadimplente | Sim / Não |
| `mensalidades_em_dia` | Mensalidades em Dia | Sim / Não |
| `bolsista` | Bolsista | Sim / Não |
| `curso` | Curso | os 6 cursos do formulário |
| `turno` | Turno | Diurno / Noturno |
| `aprovadas` | Disciplinas Aprovadas (1º Sem.) | inteiro 0–30 |
| `media` | Média das Notas | 0–20 (aceita `13,0` ou `13.0`) |
| `renda_familiar`, `ocupacao` | Renda / Ocupação | opcionais, **não entram no modelo** (ver abaixo) |

Os textos aceitam variações de maiúsculas e acentos (`nao`, `NÃO`).

### Resposta 200
```json
{
  "classe": "Graduate",
  "rotulo": "Concluinte / Formado",
  "probabilidade": 0.6626,
  "probabilidades": {"Dropout": 0.0091, "Enrolled": 0.3283, "Graduate": 0.6626},
  "entrada": { "...": "dados já validados e normalizados" }
}
```
Erros: `422` com `{"erro": "Dados inválidos.", "detalhes": {"campo": "mensagem"}}`,
`400` para corpo malformado, `503` se o modelo não estiver carregado.

## Decisões de modelagem (para o relatório)

- **Modelo:** `CategoricalNB` dentro de um `Pipeline`, de modo que a API envia os dados brutos do formulário.
- **Variáveis numéricas** (idade, aprovadas, média) são divididas em faixas, pois o modelo trabalha com categorias (faixas em `ml_common.CORTES`).
- **Escolaridade dos pais:** os 34 códigos da UCI foram agrupados nos 5 níveis da tela.
- **Curso:** só 6 dos 17 cursos da base estão na tela; os demais viram a categoria `Outro` no treino (não selecionável).
- **Campos sem correspondência na base:** *Renda Média Familiar* e *Ocupação do Estudante* não existem no dataset (ele só tem indicadores macroeconômicos e ocupação dos pais). Por isso a API os valida, mas não os usa. Se a equipe quiser que influenciem a previsão, é preciso trocá-los por variáveis da base ou escolher outro dataset.
