from transformers import pipeline


def init_emotions_model():
    classifier = pipeline(
        task="text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None)

    return classifier
