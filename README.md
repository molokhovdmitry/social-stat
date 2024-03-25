---
title: Social-Stat
emoji: 🦕
colorFrom: indigo
colorTo: pink
sdk: streamlit
sdk_version: 1.29.0
app_file: src/app.py
pinned: false
---


# Social-Stat: A Streamlit Web App for Social Media Analysis
Streamlit web application for social network analysis.

[Hugging Face Space](https://huggingface.co/spaces/molokhovdmitry/social-stat)

![social-stat](social-stat.gif)
## Features

- **Emotion Prediction**: Utilizes a text classification model to predict emotions in video comments.
- **Topic Modeling**: Applies Non-negative Matrix Factorization (NMF) to identify and visualize the main topics discussed in the comments.
- **t-SNE Visualization**: Provides a 2D and 3D visualization of the comment data, highlighting patterns and clusters.
- **Language Detection**: Detects the language of comments to understand the global reach of the video and visualizes the distribution of languages on a **plotly Choropleth** map.

## How to Use

1. **Enter a YouTube Video URL or ID**: Input the URL or ID of the YouTube video.
2. **Select Analysis Options**: Choose whether to predict emotions, perform NMF, visualize with t-SNE, and display a language map.
3. **Adjust Parameters**: Customize the analysis by adjusting parameters such as the number of NMF components, t-SNE perplexity.
4. **Analyze**: Click the "Analyze" button to start the analysis.


# Installation and Running
```
git clone https://github.com/molokhovdmitry/social-stat
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run src/app.py
```
