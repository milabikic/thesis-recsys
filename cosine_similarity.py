# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:48:18 2022

@author: Mila Bikić
"""

import pandas as pd

informacije = pd.read_csv("podaci/tmdb_5000_credits.csv")
filmovi = pd.read_csv("podaci/tmdb_5000_movies.csv")

informacije_preimenovano = informacije.rename(index = str, columns = {"movie_id": "id"})
filmovi_spojeno = filmovi.merge(informacije_preimenovano, on="id")
filmovi_ocisceno = filmovi_spojeno.drop(columns=["homepage", "title_x", "title_y", "status", "production_countries"])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_mjera = TfidfVectorizer(min_df = 3,  max_features = None,
            strip_accents = "unicode", analyzer = "word", 
            token_pattern = r"\w{1,}",
            ngram_range=(1, 3),
            stop_words = "english")
tfidf_matrica = tfidf_mjera.fit_transform(filmovi_ocisceno["overview"].values.astype('U'))

from sklearn.metrics.pairwise import sigmoid_kernel
sig = sigmoid_kernel(tfidf_matrica, tfidf_matrica)

indeksi = pd.Series(filmovi_ocisceno.index, index=filmovi_ocisceno["original_title"]).drop_duplicates()

def preporuka(title, sig=sig):
    idx = indeksi[title]
    sig_rezultat = list(enumerate(sig[idx]))
    sig_rezultat = sorted(sig_rezultat, key = lambda x: x[1], reverse=True)
    sig_rezultat = sig_rezultat[1:11]
    film_indeksi = [i[0] for i in sig_rezultat]
    return filmovi_ocisceno["original_title"].iloc[film_indeksi]