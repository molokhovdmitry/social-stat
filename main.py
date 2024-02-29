from fastapi import FastAPI
from yt_api import get_comments
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    YT_API_KEY: str
    model_config = SettingsConfigDict(env_file='.env')


settings = Settings()
app = FastAPI(title='social-stat')


YT_API_KEY = settings.YT_API_KEY


@app.get('/')
def home():
    return 'social-stat'


@app.post('/predict')
def predict(video_id):
    return get_comments(video_id, YT_API_KEY)
