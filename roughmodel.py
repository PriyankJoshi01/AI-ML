from transformers import pipeline

# Download and save the model locally
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
pipe = pipeline("text-classification", model=model_name, top_k=None)
pipe.save_pretrained("./local_emotion_model")
