import sounddevice as sd
import numpy as np
import torch
import sys
sys.path.append("/home/r0b0t1x/ML/STS/builtmodels/parler_tts_main")
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from threading import Thread
from googletrans import Translator

whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

# audio_path = "/home/r0b0t1x/ML/STS/audioformodel/story.wav"
sample_rate = 16000
duration = 5

language_descriptions = {
    "en": "A clear and natural English speaker.",
    "fr": "Un locuteur français fluide et naturel.",
    "es": "Un hablante español con pronunciación clara.",
    "de": "Ein deutscher Sprecher mit ruhiger Stimme.",
    "zh": "一个清晰普通话发音的中文说话者。",
    "hi": "एक स्पष्ट और मधुर हिंदी वक्ता।",
    "ja": "明瞭な発音の日本語話者。",
}

# #This is the batch recording
# def record_audio():
#     print("recording")
#     audio = sd.rec(int(duration * sample_rate), samplerate = sample_rate, channels = 1, dtype = np.float32)
#     sd.wait()
#     print("recording finished")
#     return audio.squeeze()

# This si for streaming input
def stream_audio():
    print("listening! speak now...")
    with sd.InputStream(samplerate = sample_rate, channels = 1, dtype = "float32") as stream:
        while True:
            audio_chunk = stream.read(int(duration * sample_rate))[0].squeeze()
            yield audio_chunk

def transcribe(audio_chunk, target_language="hi"):
    print(f"transcribing in {target_language}...")

    input_features = processor(
        audio_chunk, 
        sampling_rate = sample_rate, 
        return_tensors = "pt"
    ).input_features
    attention_mask = torch.ones_like(input_features)
    predicted_ids = whisper_model.generate(input_features, attention_mask=attention_mask)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"transribed ({target_language}) : {transcription}")

    return transcription

    # result = whisper_model.transcribe(audio, fp16=torch.cuda.is_available(), language = target_language)
    # print(f"transcription ({target_language}): {result['text']}")
    # return result['text']


def translating(transcription):
    translator = Translator()
    translated_text = translator.translate(transcription, dest = "hi")
    print("translated:", translated_text.text)
    return translated_text

torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model_name = "parler-tts/parler-tts-mini-v1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(torch_device, dtype=torch_dtype)

sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

def generate_audio(text, target_language = "hi"):
    play_steps = int(frame_rate * 0.2)
    description = language_descriptions.get(target_language, language_descriptions["hi"])

    streamer = ParlerTTSStreamer(model, device = torch_device, play_steps= play_steps)

    inputs = tokenizer(text = description, return_tensors="pt", return_attention_mask = True).to(torch_device)
    prompt = tokenizer(text = text, return_tensors="pt", return_attention_mask = True).to(torch_device)

    # if inputs.attention_mask is None:
    #     print("⚠️ Fixing missing `inputs.attention_mask`")
    #     inputs.attention_mask = torch.ones_like(inputs.input_ids)

    # if prompt.attention_mask is None:
    #     print("⚠️ Fixing missing `prompt.attention_mask`")
    #     prompt.attention_mask = torch.ones_like(prompt.input_ids)


    # print(f"🛠 Final Inputs Attention Mask: {inputs.attention_mask}")
    # print(f"🛠 Final Prompt Attention Mask: {prompt.attention_mask}")

    generation_kwargs= dict(
        input_ids = inputs.input_ids,
        prompt_input_ids=prompt.input_ids,
        attention_mask=inputs.attention_mask,
        # prompt_attention_mask=prompt.attention_mask,
        streamer=streamer,
        do_sample=True,
        temperature=0.2,
        min_new_tokens=30,
    )

    print("🚀 Running Parler-TTS with generation_kwargs:")
    for key, value in generation_kwargs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: {value}")


    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

# def generate_audio(text, target_language="hi"):
#     play_steps = int(frame_rate * 0.5)  # ✅ Optimized for better streaming
#     stride = play_steps // 10  # ✅ Overlap audio chunks for smoother playback
#     description = language_descriptions.get(target_language, language_descriptions["hi"])

#     streamer = ParlerTTSStreamer(model, device=torch_device, play_steps=play_steps, stride=stride)

#     # ✅ Tokenize inputs
#     inputs = tokenizer(text=description, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(torch_device)
#     prompt = tokenizer(text=text, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(torch_device)

#     # ✅ Ensure `attention_mask` exists
#     inputs["attention_mask"] = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"], dtype=torch.int64, device=torch_device))
#     prompt["attention_mask"] = prompt.get("attention_mask", torch.ones_like(prompt["input_ids"], dtype=torch.int64, device=torch_device))

#     # ✅ Convert to standard dictionary
#     generation_kwargs = {
#         "input_ids": inputs["input_ids"],
#         "attention_mask": inputs["attention_mask"],  
#         "prompt_input_ids": prompt["input_ids"],
#         "prompt_attention_mask": prompt["attention_mask"],
#         "streamer": streamer,
#         "do_sample": True,
#         "temperature": 0.8,  # ✅ Balance between variety & coherence
#         "top_k": 50,  # ✅ Prevents random gibberish output
#         "top_p": 0.95,  # ✅ Improves natural flow
#         "min_new_tokens": 40,  # ✅ Ensures speech isn't too short
#         "max_new_tokens": 100,  # ✅ Allows for complete, natural sentences
#     }

#     print("🚀 Running Parler-TTS with final generation_kwargs:")
#     for key, value in generation_kwargs.items():
#         if isinstance(value, torch.Tensor):
#             print(f"{key}: shape={value.shape}, dtype={value.dtype}")
#         else:
#             print(f"{key}: {value}")

#     # ✅ Start in a separate thread to allow streaming
#     thread = Thread(target=model.generate, kwargs=generation_kwargs)
#     thread.start()



    # for audio_chunk in streamer:
    #     if audio_chunk.shape[0] == 0: break

    #     sd.play(audio_chunk.squeeze(), samplerate=sampling_rate)
    #     sd.wait()
    #     # yield sampling_rate, audio_chunk

    for audio_chunk in streamer:
        print(f"🔊 Received audio chunk of shape: {audio_chunk.shape}")  # ✅ Debugging output

        if audio_chunk.shape[0] == 0:
            print("⚠️ Warning: Empty audio chunk received. Skipping playback.")
            continue  # ✅ Avoids playing an empty buffer

        sd.play(audio_chunk.squeeze(), samplerate=sampling_rate)
        sd.wait()  # ✅ Ensure playback completes before next chunk


    # for audio_chunk in streamer:
    #     if audio_chunk.shape[0] == 0:
    #         print("⚠️ Warning: Empty audio chunk received. Skipping playback.")
    #         break  # ✅ Avoids playing an empty buffer

    # sd.play(audio_chunk.squeeze(), samplerate=sampling_rate)
    # sd.wait()  # ✅ Ensure playback completes before next chunk

    # ✅ Add small delay to prevent ALSA corruption
    import time
    time.sleep(0.5)  # Allow buffer flush before freeing memory




def speech_to_speech(target_language = "hi"):
    audio_stream = stream_audio()

    for audio_chunk in audio_stream:
        text = transcribe(audio_chunk, target_language)
        translated_text = translating(text)
        Thread(target=generate_audio, args=(translated_text.text, target_language)).start()

    # while true:
    #     audio = record_audio()
    #     text = transcribe(audio)
    #     print(f"speaking: {text}")

    #     for(sr, chunk) in generate_audio(text):
    #         sd.play(chunk.cpu().numpy().squeeze(), samplerate = sr)
    #         sd.wait()

speech_to_speech("hi")