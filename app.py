from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import speech_recognition as sr
from gtts import gTTS
import os
import base64
import tempfile
import requests
import json
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Hugging Face API configuration
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Language code mapping
LANGUAGE_CODES = {
    'en': 'eng_Latn',
    'es': 'spa_Latn',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
    'it': 'ita_Latn',
    'pt': 'por_Latn',
    'nl': 'nld_Latn',
    'pl': 'pol_Latn',
    'ru': 'rus_Cyrl',
    'ja': 'jpn_Jpan',
    'zh': 'zho_Hans',
    'ar': 'ara_Arab',
    'ko': 'kor_Hang',
    'hi': 'hin_Deva',
    'tr': 'tur_Latn'
}

@app.route('/translate', methods=['POST'])
def translate():
    temp_audio_path = None
    try:
        data = request.get_json()
        logger.debug("Received translation request")
        
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        text = data.get('text')
        target_lang = data.get('target_lang', 'es')
        source_lang = data.get('source_lang', 'en')
        
        # Convert to NLLB language codes
        nllb_source = LANGUAGE_CODES.get(source_lang, 'eng_Latn')
        nllb_target = LANGUAGE_CODES.get(target_lang, 'spa_Latn')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        logger.debug(f"Translating from {nllb_source} to {nllb_target}")

        # Prepare the payload for Hugging Face API
        payload = {
            "inputs": text,
            "parameters": {
                "src_lang": nllb_source,
                "tgt_lang": nllb_target,
                "max_length": 512
            }
        }

        # Make request to Hugging Face API
        logger.debug(f"Sending request to Hugging Face API: {payload}")
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Hugging Face API error: {response.text}")
            return jsonify({'error': f"Translation API error: {response.text}"}), 500

        # Extract translation from response
        translation_response = response.json()
        logger.debug(f"Received translation response: {translation_response}")
        
        if isinstance(translation_response, list):
            translation = translation_response[0].get('translation_text', '')
        else:
            translation = translation_response.get('translation_text', '')
        
        if not translation:
            return jsonify({'error': 'No translation generated'}), 500

        # Generate audio for translated text
        tts = gTTS(text=translation, lang=target_lang)
        
        # Create temporary file
        temp_audio_path = tempfile.mktemp(suffix='.mp3')
        
        # Save audio to file
        tts.save(temp_audio_path)
        
        # Read the file and convert to base64
        with open(temp_audio_path, 'rb') as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        
        return jsonify({
            'translation': translation,
            'audio': audio_base64,
            'source_lang': nllb_source,
            'target_lang': nllb_target
        })
    
    except Exception as e:
        logger.error(f"Error in translation: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up the temporary file in the finally block
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok', 'message': 'Server is running'})

if __name__ == '__main__':   
    app.run(debug=True)

