import asyncio
import torch
import soundfile as sf
import io
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer
# sys.path.append("/home/r0b0t1x/ML/STS/builtmodels/parler_tts_main")
# from parler_tts import ParlerTTSForConditionalGeneration
import requests
import tempfile

app = FastAPI()

# OpenAI Whisper API Key
OPENAI_API_KEY = "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"

# Load Parler TTS Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "parler-tts/parler-tts-mini-v1.1"
model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live speech translation."""
    await websocket.accept()
    try:
        while True:
            audio_chunk = await websocket.receive_bytes()

            # Save chunk as WAV file
            with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_wav:
                temp_wav.write(audio_chunk)
                temp_wav.flush()

                # Whisper STT
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
                response = requests.post(
                    "https://api.openai.com/v1/audio/translations",
                    headers=headers, 
                    files={"file": open(temp_wav.name, "rb")}
                )
                
                result = response.json()
                text = result.get("text", "").strip()

                if not text:
                    continue  # Skip if no speech detected

                # Generate Speech (TTS)
                input_ids = tokenizer("A neutral speaker delivers a clear and natural speech.", return_tensors="pt").input_ids.to(device)
                prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

                generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
                audio_arr = generation.cpu().numpy().squeeze()


                # Save generated speech
                output_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(output_wav.name, audio_arr, model.config.sampling_rate)

                # Send back the translated text and audio
                await websocket.send_json({"text": text, "audio_url": f"http://localhost:8000/{output_wav.name}"})

    except WebSocketDisconnect:
        print("WebSocket disconnected")