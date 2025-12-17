# Markov Chain Sampling Methods for Dirichlet Process Mixture Models

Rapport PGM sur l'article **"Markov Chain Sampling Methods for Dirichlet Process Mixture Models"** de Neal, R. M. (1998).

## Installation

```bash
# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Scripts disponibles

### 1. Validation sur l'exemple du papier
```bash
python run_paper_example.py
```
Compare les résultats de l'algorithme 8 avec le Tableau 1 de Neal (1998).

### 2. Analyse de dimension
```bash
python run_dimension_analysis.py
```
Teste la scalabilité des algorithmes en fonction de la dimension des données.

### 3. Estimation par diffusion
```bash
python -m src.estimate.main
```
Entraîne un modèle de diffusion transformer pour l'estimation des paramètres.

## Algorithmes implémentés

- **Algorithme 1**: Gibbs sampling de base
- **Algorithme 5**: Metropolis-Hastings avec R propositions
- **Algorithme 8**: Méthode avec paramètres auxiliaires (m=1, 2, 30)
