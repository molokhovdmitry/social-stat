import json
import pandas as pd
import plotly.express as px

# Language codes predicted by language detection model
LANG_CODES = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'it', 'ja',
              'nl', 'pl', 'pt', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']

COUNTRY_TO_LANG_CODE = {
    'Algeria': 'ar',
    'Chad': 'ar',
    'Djibouti': 'ar',
    'Egypt': 'ar',
    'Iraq': 'ar',
    'Jordan': 'ar',
    'Kuwait': 'ar',
    'Lebanon': 'ar',
    'Libya': 'ar',
    'Mali': 'ar',
    'Mauritania': 'ar',
    'Morocco': 'ar',
    'Oman': 'ar',
    'Palestine': 'ar',
    'Qatar': 'ar',
    'Saudi Arabia': 'ar',
    'Somalia': 'ar',
    'Sudan': 'ar',
    'Syria': 'ar',
    'Tunisia': 'ar',
    'United Arab Emirates': 'ar',
    'Yemen': 'ar',
    'Bulgaria': 'bg',
    'Germany': 'de',
    'Greece': 'el',
    'Cyprus': 'el',
    'United States of America': 'en',
    'Ireland': 'en',
    'United Kingdom': 'en',
    'Canada': 'en',
    'Australia': 'en',
    'Mexico': 'es',
    'Mexico': 'es',
    'Colombia': 'es',
    'Spain': 'es',
    'Argentina': 'es',
    'Peru': 'es',
    'Venezuela': 'es',
    'Chile': 'es',
    'Guatemala': 'es',
    'Ecuador': 'es',
    'Bolivia': 'es',
    'Cuba': 'es',
    'Dominican Rep.': 'es',
    'Honduras': 'es',
    'Paraguay': 'es',
    'El Salvador': 'es',
    'Nicaragua': 'es',
    'Costa Rica': 'es',
    'Panama': 'es',
    'Uruguay': 'es',
    'Guinea': 'es',
    'France': 'fr',
    'India': 'hi',
    'Italy': 'it',
    'Japan': 'ja',
    'Netherlands': 'nl',
    'Belgium': 'nl',
    'Poland': 'pl',
    'Portugal': 'pt',
    'Russia': 'ru',
    'Uganda': 'sw',
    'Kenya': 'sw',
    'Tanzania': 'sw',
    'Thailand': 'th',
    'Turkey': 'tr',
    'Pakistan': 'ur',
    'Vietnam': 'vi',
    'China': 'zh'
}


def lang_map(df):
    with open('data/countries.geo.json') as f:
        countries = json.load(f)
    country_list = [country['properties']['name']
                    for country in dict(countries)['features']]
    LANG_CODES = df.value_counts('predicted_language')

    countries_data = []
    lang_count_data = []
    lang_code_data = []
    for country in country_list:
        if country in COUNTRY_TO_LANG_CODE:
            country_lang = COUNTRY_TO_LANG_CODE[country]
            if country_lang in LANG_CODES.index:
                countries_data.append(country)
                lang_count = LANG_CODES.loc[COUNTRY_TO_LANG_CODE[country]]
                lang_count_data.append(lang_count)
                lang_code_data.append(country_lang)
    lang_df = pd.DataFrame({
        'country': countries_data,
        'count': lang_count_data,
        'lang_code': lang_code_data
    })

    fig = px.choropleth(
        lang_df,
        geojson=countries,
        locations='country',
        locationmode='country names',
        color='count',
        color_continuous_scale=[
            [0, "rgb(45,45,48)"],
            [0.33, "rgb(116,173,209)"],
            [0.66, "rgb(255,255,0)"],
            [1, "rgb(255,94,5)"]
        ],
        scope='world',
        hover_data=['lang_code'],
        labels={'count': "Language Count"},
        template='plotly_dark'
    )
    fig.update_geos(showcountries=True)
    fig.update_layout(
        title_text="Language Map",
        margin={"r": 0, "t": 20, "l": 0, "b": 0}
    )

    return fig
