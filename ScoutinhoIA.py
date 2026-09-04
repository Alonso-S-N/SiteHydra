import os
import time
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

load_dotenv()
CHAVE_API = os.getenv("GOOGLE_API_KEY")

if not CHAVE_API:
    raise ValueError("API KEY não encontrada no .env")

client = genai.Client(api_key=CHAVE_API)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Iniciando Scoutinho")

SYSTEM_INSTRUCTION = """
Você é Scoutinho, agente de IA oficial da Hydra#9163.
Você deve analisar com base nas perguntas e respostas do usuario, organizar um campeonato/evento e fornecer informações detalhadas sobre.

REGRAS:
- Seja conciso e direto, fornecendo informações claras e objetivas.
- Forneça respostas detalhadas com base nos dados das equipes.
- Informe detalhadamente sobre como cada passo deve ser feito e utilize exemplos.
"""

def executar_agente(pergunta):
    for tentativa in range(3):
        try:
            chat = client.chats.create(
                model='gemini-2.5-flash',
                config={'system_instruction': SYSTEM_INSTRUCTION}
            )
            response = chat.send_message(pergunta)
            return response.text
        except Exception as e:
            if "503" in str(e) and tentativa < 2:
                time.sleep(2 * (tentativa + 1))
                continue
            return f"Erro no processamento da IA: {str(e)}"

app = Flask(__name__)
CORS(app)

@app.route('/perguntar', methods=['POST'])
def perguntar():
    dados = request.json or {}
    pergunta = dados.get('pergunta')
    
    if not pergunta:
        return jsonify({"resposta": "Pergunta vazia."}), 400

    resposta = executar_agente(pergunta)
    return jsonify({"resposta": resposta})

if __name__ == '__main__':
  
    app.run(port=5000, debug=False)