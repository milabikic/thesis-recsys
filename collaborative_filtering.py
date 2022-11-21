# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 07:03:52 2022

@author: Mila Bikić
"""


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
filmovi = pd.read_csv("podaci/movies.csv")
ocjene = pd.read_csv("podaci/ratings.csv")

print(f"Dimenzije movies.csv: {filmovi.shape}\nDimenzije ratings.csv: {ocjene.shape}")

podaci = [filmovi, ocjene]

for dokument in podaci:
    print(dokument.head(10))
    
filmovi = filmovi.merge(ocjene, on = "movieId", how = "left")
print(filmovi.head(10))

ocjene = pd.DataFrame(filmovi.groupby("title")["rating"].mean())
print(ocjene.head(10))

ocjene["Total Rating"] = pd.DataFrame(filmovi.groupby("title")["rating"].count())
print(ocjene.head(10))

film_korisnik = filmovi.pivot_table(index = 'userId', columns = 'title',values = 'rating')
print(film_korisnik.head(10))

ocjene_nula = film_korisnik.copy().fillna(0)

print(ocjene_nula.head(10))

matrica_slicnosti = cosine_similarity(ocjene_nula, ocjene_nula)
matrica_slicnosti_tablica = pd.DataFrame(matrica_slicnosti, index = film_korisnik.index, columns = film_korisnik.index)

def izracun_ocjena(naziv, id_korisnik):
    if naziv in film_korisnik:
        kosinusna_vrijednost = matrica_slicnosti_tablica[id_korisnik]
        vrijednost_ocjena = film_korisnik[naziv]
        indeks_neocjenjen = vrijednost_ocjena[vrijednost_ocjena.isnull()].index
        vrijednost_ocjena = vrijednost_ocjena.dropna()
        kosinusna_vrijednost = kosinusna_vrijednost.drop(indeks_neocjenjen)
        ocjene_filma = np.dot(vrijednost_ocjena, kosinusna_vrijednost)/kosinusna_vrijednost.sum()
    else:
        return 2.5
    return ocjene_filma

nazivFilmaLista = list(film_korisnik.columns.values)

def top_10_preporuka(id_korisnik):
    rjecnik_filmovi = {}
    index = 0
    for film in nazivFilmaLista: #vraca li iloc po redu?
        if ocjene_nula.iloc[id_korisnik][index] == 0:
            rjecnik_filmovi[film] = izracun_ocjena(film, id_korisnik)
        index += 1
    sortirano = sorted(rjecnik_filmovi.items(), key=lambda x: x[1], reverse=True)[:10]
    return sortirano