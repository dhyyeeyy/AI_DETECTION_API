import base64
import io
import numpy as np
import librosa
import soundfile as sf
import torch
import shap
import joblib
import whisper

from fastapi import FastAPI, Header
from transformers import Wav2Vec2Model

app = FastAPI()

SECRET_KEY = "sk_test_123456789"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load models ONCE (very important) ----
print("Loading models...")

model_name = "facebook/wav2vec2-large-xlsr-53"
wav2vec_model = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)
wav2vec_model.eval()

xgb_model = joblib.load("hybrid_model.pkl")
scaler = joblib.load("feature_hybrid_scaler.pkl")

print("All models loaded!")
def split_audio(y, sr=16000, chunk_sec=4):
    chunk_len = sr * chunk_sec
    chunks = []

    for i in range(0, len(y), chunk_len):
        chunk = y[i:i+chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk)

    return chunks

def wav2vec_embedding_chunked_from_waveform(y, sr=16000):
    chunks = split_audio(y, sr=sr, chunk_sec=4)
    chunk_embeds = []

    for chunk in chunks:
        audio_tensor = torch.tensor(chunk, dtype=torch.float32)\
                            .unsqueeze(0)\
                            .to(DEVICE)

        with torch.no_grad():
            out = wav2vec_model(audio_tensor).last_hidden_state

        embed = out.mean(dim=1).cpu().numpy().squeeze()
        chunk_embeds.append(embed)

    return np.mean(np.vstack(chunk_embeds), axis=0)

def flcc_like_from_waveform(y, sr=16000, n_mfcc=20):
    cc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mean = np.mean(cc, axis=1)
    std = np.std(cc, axis=1)
    delta = librosa.feature.delta(cc).mean(axis=1)
    return np.concatenate([mean, std, delta])   # ~60 dims

def extract_hybrid_features(y, sr=16000):
    wv = wav2vec_embedding_chunked_from_waveform(y, sr)
    flcc = flcc_like_from_waveform(y, sr)
    return np.concatenate([wv, flcc])

def analyze_audio(audio_base64: str):
    # --- Decode Base64 to waveform ---
    audio_bytes = base64.b64decode(audio_base64)
    audio_buffer = io.BytesIO(audio_bytes)

    y, sr = librosa.load(audio_buffer, sr=16000, mono=True)

    features = extract_hybrid_features(y, sr)
    features = features.reshape(1, -1)

    # --- XGBoost prediction ---
    y_prob = xgb_model.predict_proba(features)[0]
    threshold = 0.66
    prob_ai = float(y_prob[1])

    if prob_ai > threshold:
        classification = "AI_GENERATED"
    else:
        classification = "HUMAN"

    confidence_score = round(abs(prob_ai - 0.5) * 2, 2)

    # --- SHAP explanation ---
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(features)
    sv = np.abs(shap_values).flatten()

    N_MFCC = 60
    mfcc_shap = np.mean(sv[:N_MFCC])
    wav2vec_shap = np.mean(sv[N_MFCC:])

    if classification == "AI_GENERATED":
        if mfcc_shap > wav2vec_shap:
            explanation = (
                "Unnatural spectral smoothness and stable pitch patterns "
                "suggest synthetic speech."
            )
        else:
            explanation = (
                "Deep acoustic patterns resemble known AI-generated voices."
            )
    else:
        explanation = (
            "Natural variations in tone, rhythm, and timbre are consistent "
            "with human speech."
        )

    return {
        "classification": classification,
        "confidenceScore": confidence_score,
        "language": mapped_language,
        "transcript": transcript,
        "explanation": explanation,
        "prob_ai": prob_ai
    }

# -----------------------------------------------------------
@app.post("/api/voice-detection")
def detect_voice(data: dict, x_api_key: str = Header(None)):

    if x_api_key != SECRET_KEY:
        return {
            "status": "error",
            "message": "Invalid API key or malformed request"
        }

    try:
        audio_b64 = data.get("audioBase64")
        if not audio_b64:
            raise ValueError("Missing audioBase64")

        result = analyze_audio(audio_b64)
        mapped_language = data["language"]

        return {
            "status": "success",
            "language": mapped_language,
            "classification": result["classification"],
            "confidenceScore": result["confidenceScore"],
            "explanation": result["explanation"]
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
