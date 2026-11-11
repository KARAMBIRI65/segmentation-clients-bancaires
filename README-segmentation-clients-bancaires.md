# Segmentation de clients bancaires

Regrouper les clients d'une banque en groupes qui se ressemblent.

Les données viennent du jeu Bank Marketing, 41 188 clients d'une banque portugaise appelés pour un placement.

J'ai utilisé K-means et gardé 5 groupes.

Les taux d'acceptation vont de 4,9 % à 31,4 % selon le groupe. Les étudiants et les retraités acceptent le plus.

K-means n'est pas idéal ici car beaucoup de variables sont du type oui ou non.

Pour le lancer, `pip install -r requirements.txt` puis `python -m src.segmentation`.

Issa Aziz Karambiri
