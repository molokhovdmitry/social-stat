# social-stat
API application for social network analysis.

# Endpoints
## Get /predict/{video_id}
Returns `pandas` DataFrame with all `commentThreads` of a `YouTube` video with emotion scores estimated by [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions).
<details>
<summary>All DataFrame columns:</summary>

- comment_id
- video_id
- channel_id
- author_display_name
- text_original
- text_display
- published_at
- updated_at
- like_count
- can_reply
- total_reply_count
- neutral
- approval
- annoyance
- disapproval
- realization
- admiration
- disappointment
- excitement
- disgust
- confusion
- joy
- anger
- optimism
- amusement
- gratitude
- surprise
- sadness
- fear
- curiosity
- love
- embarrassment
- desire
- caring
- pride
- relief
- grief
- remorse
- nervousness

</details>

# Installation and Running
```
git clone https://github.com/molokhovdmitry/social-stat
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload
```
