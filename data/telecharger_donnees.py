"""Télécharge le jeu Bank Marketing (UCI) si le fichier CSV est absent.

Usage : python data/telecharger_donnees.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

CIBLE = Path(__file__).resolve().parent / "bank-additional-full.csv"
URL_UCI = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def main():
    if CIBLE.exists():
        print(f"Déjà présent : {CIBLE}")
        return
    print("Téléchargement depuis l'UCI...")
    archive = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL_UCI).read()))
    # L'archive UCI contient un second zip : bank-additional.zip
    interne = zipfile.ZipFile(io.BytesIO(archive.read("bank-additional.zip")))
    CIBLE.write_bytes(interne.read("bank-additional/bank-additional-full.csv"))
    print(f"Enregistré : {CIBLE}")


if __name__ == "__main__":
    main()
