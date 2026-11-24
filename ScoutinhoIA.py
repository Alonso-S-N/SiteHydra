import os
import time

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import types

load_dotenv()

CHAVE_API = os.getenv("GOOGLE_API_KEY")
if not CHAVE_API:
    raise ValueError("API KEY não encontrada no .env")

ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "http://127.0.0.1:5501")
ORIGENS_PERMITIDAS = [origem.strip() for origem in ALLOWED_ORIGIN.split(",")]

client = genai.Client(api_key=CHAVE_API)

print("Iniciando Scoutinho")

# Carrega o arquivo de referência das equipes uma única vez ao iniciar
CAMINHO_DOC = os.path.join(os.path.dirname(__file__), "..", "Historico", "historico_scoutinho.md")
try:
    with open(CAMINHO_DOC, "r", encoding="utf-8") as f:
        DADOS_EQUIPES = f.read()
except FileNotFoundError:
    DADOS_EQUIPES = "Base de dados não encontrada. Responda com boas práticas gerais de eventos STEAM."

SYSTEM_INSTRUCTION = f"""
Você é Quimera, agente de IA oficial da Hydra#9163.
Você deve analisar os 4 parâmetros básicos informados pelo usuário (Quantitativo de participantes, Formato da ação STEAM, Público-alvo e Habilidade principal a desenvolver) e estruturar o planejamento completo do evento com base nas experiências reais abaixo.

BASE DE REFERÊNCIA DE EQUIPES REAIS:
---
{DADOS_EQUIPES}
---

REGRAS OBRIGATÓRIAS:
- Seja conciso, direto, claro e evite ambiguidades ou respostas vagas.
- Forneça respostas detalhadas cruzando com os dados reais das equipes da base acima (adaptando conceitos de robótica para o universo STEAM quando necessário).
- Informe detalhadamente como cada passo deve ser feito, utilizando exemplos práticos.
- Destaque os pontos mais importantes de forma objetiva.
- não cite as equipes da base, apenas use os dados para embasar suas respostas.

ESTRUTURA OBRIGATÓRIA DA RESPOSTA:
1. **Análise de Viabilidade da Habilidade:** Avalie e diga claramente logo no início se é possível desenvolver a habilidade escolhida pelo usuário dentro do formato do evento, público-alvo e quantitativo de participantes informados. Se houver algum conflito (ex: tentar ensinar programação avançada em uma mostra rápida aberta ao público ou para crianças muito novas sem treino prévio), aponte o porquê e explique como adaptar a atividade para que funcione.
2. **Cronograma & Capacitação Prévia:** Defina o tempo ideal de planejamento (em meses) e se exige treinamento prévio dos participantes.
3. **Equipe Organizadora (Staff) & Dinâmica:** Indique quantas pessoas são necessárias na organização, como dividir os setores e como estruturar as atividades/pontuação.
4. **Orçamento, Financiamento & Maiores Custos:** Recomende como captar recursos e liste os principais centros de custo esperados.
5. **Alertas de Risco & Gargalos Reais:** Cite equipes da base que fizeram eventos semelhantes e alerte sobre os imprevistos que elas enfrentaram (infraestrutura, tomadas/rede, licenças, regulamento).
6. **Engajamento & Avaliação:** Sugira como engajar o público e como medir se a habilidade foi realmente desenvolvida.
"""

# Ordem de preferência: tenta o mais barato primeiro; se der 503, pula para o próximo
MODELOS_FALLBACK = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash"
]


def executar_agente(pergunta):
    ultimo_erro = ""

    for modelo in MODELOS_FALLBACK:
        for tentativa in range(2):  # Tenta até 2 vezes em cada modelo
            try:
                config_args = {"system_instruction": SYSTEM_INSTRUCTION}
                if "3." in modelo:
                    config_args["thinking_config"] = types.ThinkingConfig(
                        thinking_level="minimal"
                    )

                response = client.models.generate_content(
                    model=modelo,
                    contents=pergunta,
                    config=types.GenerateContentConfig(**config_args)
                )
                return response.text

            except Exception as e:
                ultimo_erro = str(e)
                if "503" in ultimo_erro or "429" in ultimo_erro:
                    time.sleep(1.5)
                    continue
                return f"Erro no processamento da IA: {ultimo_erro}"

    return "O sistema está com alta demanda no momento. Por favor, aguarde alguns segundos e envie novamente."


app = Flask(__name__)
CORS(app, origins=ORIGENS_PERMITIDAS)


@app.route('/perguntar', methods=['POST'])
def perguntar():
    dados = request.json or {}
    pergunta = dados.get('pergunta')
    if not pergunta:
        return jsonify({"resposta": "Pergunta vazia."}), 400
    resposta = executar_agente(pergunta)
    return jsonify({"resposta": resposta})


if __name__ == '__main__':
    porta = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=porta, debug=False)