from fastapi import FastAPI, Response
from pydantic_settings import BaseSettings, SettingsConfigDict
import pandas as pd

from src.yt_api import get_comments
from src.models import init_emotions_model


class Settings(BaseSettings):
    YT_API_KEY: str
    PRED_BATCH_SIZE: int = 512
    MAX_COMMENT_SIZE: int = 300
    model_config = SettingsConfigDict(env_file='.env')


settings = Settings()
app = FastAPI(title='social-stat')

emotions_clf = init_emotions_model()


@app.get('/')
def home():
    return 'social-stat'


@app.get('/predict')
def predict(video_id):
    # Get comments
    comments = get_comments(
        video_id,
        settings.MAX_COMMENT_SIZE,
        settings.YT_API_KEY
    )
    comments_df = pd.DataFrame(comments)

    # Predict emotions in batches
    text_list = comments_df['text_display'].to_list()
    batch_size = settings.PRED_BATCH_SIZE
    text_batches = [text_list[i:i + batch_size]
                    for i in range(0, len(text_list), batch_size)]
    preds = []
    for batch in text_batches:
        preds.extend(emotions_clf(batch))

    # Add predictions to DataFrame
    preds_df = []
    for pred in preds:
        pred_dict = {}
        for emotion in pred:
            pred_dict[emotion['label']] = emotion['score']
        preds_df.append(pred_dict)
    preds_df = pd.DataFrame(preds_df)
    comments_df = pd.concat([comments_df, preds_df], axis=1)

    # Return DataFrame as a JSON file
    return Response(
        content=comments_df.to_json(orient='records'),
        media_type='application/json')
