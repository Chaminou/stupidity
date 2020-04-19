
# Stupidity

## Installation

1. Installer open_spiel https://github.com/deepmind/open_spiel
2. Remplacer le fichier open_spiel/python/algorithms/dqn.py par celui fourni dans ce depot
3. Les autres dépendances du dépot sont dans requirements.txt

## Utilisation

### parameters.py

Ce fichier permet de choisir les parametres communs aux 3 autres scripts

Choix du reseaux de neurones et du jeu
Chemin des agents et du ladder sauvegardé

### generateur.py

Permet de générer les agents, de les versionner et de les entrainer.

### ranker.py

Permet de creer une population de jean jacques, d'en ajuster les parametres, de les faire s'affronter puis de les sauvegarder

### plotter.py

Permet de visualiser ce que l'on veut sur le ladder de jean jacques



