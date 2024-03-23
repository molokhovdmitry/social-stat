import os
from dotenv import load_dotenv
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE

from yt_api import YouTubeAPI
from maps import lang_map


# Load app settings
load_dotenv()
YT_API_KEY = os.getenv('YT_API_KEY')
MAX_COMMENT_SIZE = int(os.getenv('MAX_COMMENT_SIZE'))
PRED_BATCH_SIZE = int(os.getenv('PRED_BATCH_SIZE'))
LANG_DETECTION_CONF = float(os.getenv('LANG_DETECTION_CONF'))


@st.cache_resource
def init_emotions_model():
    classifier = pipeline(
        task="text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None)

    return classifier


@st.cache_resource
def init_embedding_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model


@st.cache_resource
def init_lang_model():
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    pipe = pipeline("text-classification", model=model_ckpt)
    return pipe


def predict_emotions(df, clf):
    """
    Predicts emotions for every `text_original` in a DataFrame `df` with a
    classifier `clf`.
    Returns a DataFrame with emotion columns.
    """
    # Predict emotions in batches
    text_list = df['text_original'].to_list()
    batch_size = PRED_BATCH_SIZE
    text_batches = [text_list[i:i + batch_size]
                    for i in range(0, len(text_list), batch_size)]
    preds = [comment_emotions
             for text_batch in text_batches
             for comment_emotions in clf(text_batch)]

    # Add predictions to DataFrame
    preds_df = pd.DataFrame([{emotion['label']: emotion['score']
                            for emotion in pred} for pred in preds])
    df = pd.concat([df, preds_df], axis=1)

    return df


def detect_languages(df, clf):
    """
    Detects languages for every `text_original` in a DataFrame `df` with a
    classifier `clf`. Takes the language with the highest score.
    Returns a DataFrame with `predicted_language` column.
    """
    # Detect languages in batches
    text_list = df['text_original'].to_list()
    batch_size = PRED_BATCH_SIZE
    text_batches = [text_list[i:i + batch_size]
                    for i in range(0, len(text_list), batch_size)]
    preds = [batch_preds[0]['label']
             if batch_preds[0]['score'] > LANG_DETECTION_CONF
             else None
             for text_batch in text_batches
             for batch_preds in clf(text_batch, top_k=1, truncation=True)]

    # Add predictions to DataFrame
    df['predicted_language'] = preds

    return df


def emotion_dist_plot(df, emotion_cols):
    """
    Creates an emotion distribution plotly figure from `df` DataFrame
    and `emotion_cols` and returns it.
    """
    fig = px.bar(df[emotion_cols].sum().sort_values(ascending=False))
    fig.update_layout(title_text="Emotion Distribution",
                      width=2000)

    return fig


def nmf_plots(df,
              nmf_components,
              tfidf_max_features,
              tfidf_stop_words='english'
              ):
    """
    Converts all `text_original` values of `df` DataFrame to TF-IDF features
    and performs Non-negative matrix factorization on them.

    Returns a tuple of the modified DataFrame with NMF values and a list of
    plotly figures (`df`, [plotly figures]).
    """
    # Convert to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features,
                                 stop_words=tfidf_stop_words)
    embeddings = vectorizer.fit_transform(df['text_original'])

    # Get feature_names (words) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Perform NMF
    nmf = NMF(n_components=nmf_components)
    nmf_embeddings = nmf.fit_transform(embeddings).T
    topic_cols = [f'topic_{topic_num+1}'
                  for topic_num in range(nmf_components)]

    # Add NMF values to the DataFrame
    for i, col in enumerate(topic_cols):
        df[col] = nmf_embeddings[i]

    # Get word values for every topic
    word_df = pd.DataFrame(
        nmf.components_.T,
        columns=topic_cols,
        index=feature_names
    )

    # Plot word distributions of each topic
    topic_words_fig = make_subplots(
        rows=1, cols=nmf_components,
        subplot_titles=topic_cols)

    for i, col in enumerate(topic_cols):
        topic_words = word_df[col].sort_values(ascending=False)
        top_topic_words = topic_words[:top_words_in_topic]
        topic_words_fig.add_trace(go.Bar(y=top_topic_words.index,
                                         x=top_topic_words.values,
                                         orientation='h',
                                         base=0),
                                  row=1, col=i+1)
    topic_words_fig.update_layout(title_text="Topic Word Distributions",
                                  showlegend=False)

    # Plot topic contribution for the dataset
    for col in topic_cols:
        df[col + '_cumsum'] = df[col].cumsum()
    for col in topic_cols:
        cumsum_sum = df[[col + '_cumsum' for col in topic_cols]].sum(axis=1)
        df[col + '_percentage'] = df[col + '_cumsum'] / cumsum_sum
    contributions_fig = stacked_area_plot(
        x=df['published_at'],
        y_list=[df[f'topic_{i+1}_percentage'] for i in range(nmf_components)],
        names=topic_cols)

    return df, [topic_words_fig, contributions_fig]


def tsne_plots(df, encoder, emotion_cols, color_emotion, tsne_perplexity):
    """
    Encodes all `text_original` values of `df` DataFrame with `encoder`,
    uses t-SNE algorithm for visualization on these embeddings and on
    predicted emotions if they were predicted.
    """
    # Encode and add embeddings to the DataFrame
    embeddings = encoder.encode(df['text_original'])
    embedding_cols = [f'embedding_{i+1}' for i in range(embeddings.shape[1])]
    df = pd.concat([df, pd.DataFrame(embeddings, columns=embedding_cols)],
                   axis=1)

    # t-SNE
    TSNE_COMPONENTS = 2
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
    )

    # Also use predicted emotions
    if emotion_cols:
        tsne_cols = embedding_cols + emotion_cols
        color = color_emotion
        hover_data = ['first_emotion', 'second_emotion', 'text_original']
    else:
        tsne_cols = embedding_cols
        color = None
        hover_data = 'text_original'

    tsne_results = tsne.fit_transform(df[tsne_cols])
    tsne_results = pd.DataFrame(
        tsne_results,
        columns=[f'tsne_{i+1}' for i in range(TSNE_COMPONENTS)]
    )

    df = pd.concat([df, tsne_results], axis=1)

    # 2D Visualization
    fig2d = px.scatter(
        df,
        x='tsne_1',
        y='tsne_2',
        color=color,
        hover_data=hover_data
    )
    fig2d.update_layout(
        title_text="t-SNE Visualization"
    )

    # 3D Visualization with date as the third axis
    fig3d = px.scatter_3d(
        df,
        x='published_at',
        y='tsne_1',
        z='tsne_2',
        color=color,
        hover_data=hover_data
    )
    fig3d.update_layout(
        title_text="t-SNE Visualization Over Time"
    )

    return df, [fig2d, fig3d]


def stacked_area_plot(x, y_list, names):
    """Creates plotly stacked area plot. Returns a figure of that plot."""
    fig = go.Figure()
    for y, name in zip(y_list, names):
        fig.add_trace(go.Scatter(
            x=x, y=y*100,
            mode='lines',
            line=dict(width=0.5),
            stackgroup='one',
            name=name,
        ))

    fig.update_layout(
        showlegend=True,
        xaxis_type='category',
        yaxis=dict(
            type='linear',
            range=[0, 100],
            ticksuffix='%')
        )

    fig.update_layout(title_text="Topic Contribution")

    return fig


def add_top_2_emotions(row):
    emotions = row[emotion_cols].sort_values(ascending=False)
    row['first_emotion'] = emotions.index[0]
    row['second_emotion'] = emotions.index[1]
    return row


st.set_page_config(layout='wide')
st.title("Social-Stat")

# Load models
emotions_clf = init_emotions_model()
sentence_encoder = init_embedding_model()
lang_model = init_lang_model()

# Init YouTube API
yt_api = YouTubeAPI(
    api_key=YT_API_KEY,
    max_comment_size=MAX_COMMENT_SIZE
)

# Input form
with st.form(key='input'):
    video_id = st.text_input("Video ID")

    # Emotions
    emotions_checkbox = st.checkbox(
        "Predict Emotions",
        value=True,
    )

    # NMF
    nmf_checkbox = st.checkbox(
        "Non-Negative Matrix Factorization",
        value=True,
    )

    nmf_components = st.slider(
        "Topics (NMF Components)",
        min_value=2,
        max_value=20,
        value=8,
        step=1,
    )

    tfidf_max_features = st.select_slider(
        "Words (TF-IDF Vectorizer Max Features)",
        options=list(range(10, 501)) + [None],
        value=100,
    )

    top_words_in_topic = st.slider(
        "Top Topic Words",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )

    # t-SNE
    tsne_checkbox = st.checkbox(
        "t-SNE Visualization",
        value=True,
    )

    tsne_perplexity = st.slider(
        "t-SNE Perplexity",
        min_value=5,
        max_value=50,
        value=10,
        step=1,
    )

    tsne_color_emotion = st.selectbox(
        "Emotion For The Plot Color",
        options=['first_emotion', 'second_emotion']
    )

    # Language Map
    map_checkbox = st.checkbox(
        "Language Map",
        value=True,
    )

    submit = st.form_submit_button("Analyze")


if submit:
    # Get comments
    try:
        bad_id = False
        comments = yt_api.get_comments(video_id)
    except KeyError:
        st.write("Video not found.")
        bad_id = True

    if not bad_id:
        plots = []

        # Convert to pandas DataFrame and sort by publishing date
        df = pd.DataFrame(comments).sort_values('published_at')

        emotion_cols = []
        if emotions_checkbox:
            # Predict emotions
            df = predict_emotions(df, emotions_clf)
            emotion_cols = list(df.columns[11:])

            # Get emotion distribution figure
            plots.append(emotion_dist_plot(df, emotion_cols))

            # Get top 2 emotions
            df = df.apply(add_top_2_emotions, axis=1)

        if map_checkbox:
            df = detect_languages(df, lang_model)
            plots.append(lang_map(df))

        if nmf_checkbox:
            # NMF
            df, nmf_figs = nmf_plots(df, nmf_components, tfidf_max_features)
            plots.extend(nmf_figs)

        if tsne_checkbox:
            # t-SNE visualization
            df, tsne_figs = tsne_plots(df,
                                       sentence_encoder,
                                       emotion_cols,
                                       tsne_color_emotion,
                                       tsne_perplexity)
            plots.extend(tsne_figs)

        # Draw the plots
        for i, plot in enumerate(plots):
            st.plotly_chart(
                plot, sharing='streamlit',
                theme='streamlit',
                use_container_width=True)

        # Show the final DataFrame
        st.dataframe(df)
