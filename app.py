from flask import Flask, render_template, request, jsonify, send_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, WhisperForConditionalGeneration
import requests
import librosa
import numpy as np
import io
import base64
import soundfile as sf
import os
from werkzeug.utils import secure_filename
import tempfile
import subprocess

# Gemini imports
import google.generativeai as genai

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Gemini API Key
GEMINI_API_KEY = 'YOUR_GEMIN_API_KEY' # CHANGE THIS
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load Qwen LLM
llm_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", torch_dtype="auto")
llm_model.to(device)
llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")

# Whisper
whisper_processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")

# Kokoro TTS
try:
    from kokoro import KPipeline
    kokoro_pipeline = KPipeline(lang_code='a')
    KOKORO_AVAILABLE = True
except ImportError:
    print("Kokoro not available.")
    KOKORO_AVAILABLE = False




# Chat histories
chat_sessions = {
    "qwen": {},
    "gemini": {}
}

SERPER_API_KEY = "d40da60a91f82f5cb0193ebc871b72beabdf257f"

def clean_context(history):
    return "\n".join([f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}" for msg in history])

def needs_web_search(user_input):
    keywords = ["who is", "latest", "current", "news", "today", "now", "weather", "price", "score"]
    return any(kw in user_input.lower() for kw in keywords)

def search_web(query):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY}
    payload = {"q": query}
    try:
        res = requests.post(url, headers=headers, json=payload)
        results = res.json()
        if "organic" not in results:
            return "No relevant web results found."
        return "\n".join([f"- {r.get('title', '')}: {r.get('snippet', '')}" for r in results["organic"][:3]])
    except Exception as e:
        return f"Web search error: {str(e)}"

def generate_qwen_response(user_input, chat_history):
    context = clean_context(chat_history)
    if needs_web_search(user_input):
        search_results = search_web(user_input)
        system_prompt = (
            "You are a helpful and concise AI assistant. You must answer only in English. "
            "If you are unsure or lack enough information, clearly say so. "
            "Use the following search results to inform your answer."
        )
        user_prompt = (
            f"Search Results:\n{search_results}\n\nConversation History:\n{context}\n\nUser: {user_input}\nAssistant:"
        )
    else:
        system_prompt = (
            "You are a helpful and concise AI assistant. You must answer only in English. "
            "If you are unsure or lack enough information, clearly say so."
        )
        user_prompt = f"Conversation History:\n{context}\n\nUser: {user_input}\nAssistant:"

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    model_inputs = llm_tokenizer([text], return_tensors="pt").to(device)

    generated_ids = llm_model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_new_tokens=512,
        eos_token_id=llm_tokenizer.eos_token_id
    )
    generated_ids = [output_ids[len(model_inputs["input_ids"][0]):] for output_ids in generated_ids]
    raw_output = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    cleaned = raw_output.strip()
    if cleaned.lower().startswith("user:"):
        cleaned = cleaned.partition("User:")[2].strip()
    return cleaned
def generate_gemini_response(user_input, chat_history):

    context = clean_context(chat_history)

    if needs_web_search(user_input):
        search_results = search_web(user_input)
        system_prompt = (
            "You are a helpful and concise AI assistant.  "
            "If you are unsure or lack enough information, clearly say so. "
            "Use the following search results to inform your answer."
        )
        instruction = (
            "Respond in plain English without using any formatting symbols like asterisks or Markdown. "
            "Avoid using **bold** or *italic* styles."
        )
        full_prompt = (
            f"{system_prompt + instruction}\n\nSearch Results:\n{search_results}\n\nConversation History:\n{context}\n\nUser: {user_input}\nAssistant:"
        )
    else:
        system_prompt = (
            "You are a helpful and concise AI assistant. "
            "If you are unsure or lack enough information, clearly say so. "
        )

        instruction = (
            "Respond in plain English without using any formatting symbols like asterisks or Markdown. "
            "Avoid using **bold** or *italic* styles."
        )
        full_prompt = f"{system_prompt + instruction}\n\nPrevious conversation:\n{context}\n\nUser: {user_input}\nAssistant:"

  
    response = gemini_model.generate_content([full_prompt])
    return response.text.strip()
  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    session_id = data.get('session_id', 'default')
    model = data.get('model', 'qwen')

    if model not in chat_sessions:
        return jsonify({'error': f'Invalid model: {model}'}), 400

    if session_id not in chat_sessions[model]:
        chat_sessions[model][session_id] = []

    chat_history = chat_sessions[model][session_id]

    try:
        if model == 'gemini':
            if not gemini_model:
                raise Exception("Gemini API not available or not configured.")
            response = generate_gemini_response(user_input, chat_history)
        else:
            response = generate_qwen_response(user_input, chat_history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})

    return jsonify({'response': response, 'session_id': session_id})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        audio_data = request.files.get('audio')
        if not audio_data:
            return jsonify({'error': 'No audio data received'}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            audio_data.save(tmp_file.name)
        converted_path = tmp_file.name + "_converted.wav"
        subprocess.run(["ffmpeg", "-i", tmp_file.name, "-ar", "16000", "-ac", "1", converted_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        audio, sr = librosa.load(converted_path, sr=16000)
        inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="pt", language="en")
        generated_ids = whisper_model.generate(input_features=inputs.input_features)
        transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        os.unlink(tmp_file.name)
        os.unlink(converted_path)

        return jsonify({'transcription': transcription})
    except Exception as e:
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    try:
        if not KOKORO_AVAILABLE:
            return jsonify({'error': 'Kokoro TTS not available'}), 500

        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        generator = kokoro_pipeline(text, voice='af_heart')
        for _, _, audio in generator:
            buffer = io.BytesIO()
            sf.write(buffer, audio, 24000, format='WAV')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return jsonify({'audio': audio_base64, 'sample_rate': 24000})
    except Exception as e:
        return jsonify({'error': f'Speech synthesis failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
