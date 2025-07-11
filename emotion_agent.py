from transformers import pipeline
from collections import defaultdict
import whisper
import ffmpeg

# Load Whisper locally
transcriber = whisper.load_model("tiny")

# Load emotion classifier
emotion_model = pipeline(
    "text-classification", 
    model="bhadresh-savani/distilbert-base-uncased-emotion", 
    top_k=None
)

# Custom emotion labels
emotion_labels = ['happy', 'angry', 'frustrated', 'confused', 'sad', 'surprised', 'neutral', 'hopeful', 'bored']

def preprocess_audio(input_path, output_path="cleaned.wav"):
    (
        ffmpeg
        .input(input_path)
        .output(output_path, ar='16000', ac='1', format='wav', af='dynaudnorm')
        .run(overwrite_output=True)
    )
    return output_path

# TEMP translation fix
def translate_text_auto(text, src_lang):
    return text

def classify_emotion(text):
    result = emotion_model(text)[0]
    mapped = dict.fromkeys(emotion_labels, 0.0)

    for item in result:
        label = item['label'].lower()
        score = item['score']

        if label in ['joy']:
            mapped['happy'] += score
        elif label in ['anger', 'annoyance']:
            mapped['angry'] += score
        elif label in ['disgust', 'disappointment']:
            mapped['frustrated'] += score
        elif label in ['confusion', 'realization']:
            mapped['confused'] += score
        elif label in ['sadness']:
            mapped['sad'] += score
        elif label in ['surprise']:
            mapped['surprised'] += score
        elif label in ['neutral']:
            mapped['neutral'] += score
        elif label in ['caring', 'excitement']:
            mapped['hopeful'] += score
        elif label in ['boredom']:
            mapped['bored'] += score

    return mapped

def analyze_call(audio_path):
    # Reset per call
    emotion_totals = defaultdict(float)
    utterance_count = 0
    emotion_history = []
    full_transcript = []

    print("🎧 Preprocessing & transcribing...")
    clean_path = preprocess_audio(audio_path)
    result = transcriber.transcribe(clean_path)
    lang = result.get("language", "en")
    print(f"🌐 Detected language: {lang}")
    print("✅ Transcription complete.")

    for segment in result['segments']:
        text = segment['text']
        full_transcript.append(text)
        print(f"\n🗣️: {text}")
        translated = translate_text_auto(text, lang)
        emotions = classify_emotion(translated)
        print(f"🎭 Emotions: {emotions}")

        for k, v in emotions.items():
            emotion_totals[k] += v
        utterance_count += 1

        emotion_history.append({
            "text": text,
            "translated": translated,
            "emotions": emotions
        })

    total = sum(emotion_totals.values())
    final_percentages = {k: round((v / total) * 100, 2) if total > 0 else 0.0 for k, v in emotion_totals.items()}

    print("\n📊 Final Emotion Percentages:")
    print(final_percentages)

    transcript_string = "\n".join(full_transcript)

    return emotion_history, final_percentages, transcript_string
