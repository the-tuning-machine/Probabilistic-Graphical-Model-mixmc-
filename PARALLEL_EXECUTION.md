# Exécution Parallèle

Le code a été modifié pour supporter l'**exécution parallèle** des calculs sur différentes dimensions, ce qui accélère considérablement l'analyse.

## 🚀 Comment lancer

### Prérequis

Installer les dépendances (une seule fois) :

```bash
# Option 1 : Installation directe
pip3 install numpy scipy matplotlib

# Option 2 : Avec environnement virtuel (recommandé)
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib
```

### Test rapide (recommandé)

Pour vérifier que tout fonctionne :

```bash
python3 test_parallel.py
```

Ce script teste l'exécution parallèle sur 3 dimensions (10, 20, 50) avec 2 algorithmes. Durée : ~1-2 minutes.

### Analyse complète

Pour l'analyse complète sur toutes les dimensions :

```bash
python3 run_dimension_analysis.py
```

Ce script :
- Teste 9 dimensions : [2, 4, 8, 16, 32, 64, 128, 256, 512]
- Compare 6 algorithmes : Alg3, Alg5, Alg7, Alg8_m1, Alg8_m2, Alg8_m30
- **S'exécute en parallèle** sur tous les CPUs disponibles
- Génère le graphique : `results/iid_time_vs_dimension.png`

Durée estimée : Dépend du nombre de CPUs (avec 8 CPUs : ~10-15 minutes)

## ⚙️ Comment ça marche

### Parallélisation

Le code utilise `multiprocessing.Pool` pour traiter **chaque dimension en parallèle** :

```python
plot_autocorr_time_vs_dimension(
    dimensions=[10, 20, 50, 100, 200],
    n_jobs=-1  # -1 = utilise tous les CPUs disponibles
)
```

Options pour `n_jobs` :
- `-1` : Utilise tous les CPUs (recommandé)
- `4` : Utilise 4 CPUs
- `1` : Exécution séquentielle (pas de parallélisation)

### Calcul du temps i.i.d.

Pour chaque dimension et chaque algorithme, le code calcule :

```
τ = 1 / (1 - ρ)                    # Temps d'autocorrélation
temps_2_iid = 2 × τ × temps_par_iter  # Temps pour 2 échantillons indépendants
```

où :
- `ρ` = autocorrélation de lag-1 (sur k ou θ₁)
- `temps_par_iter` = temps moyen par itération MCMC en ms

## 📊 Résultats

Les résultats sont sauvegardés dans `results/` :

- `iid_time_vs_dimension.png` : Graphique principal
- `results_dim{n}.pkl` : Résultats détaillés pour chaque dimension
- `results_dim{n}.json` : Résumé en JSON

## 🔧 Personnalisation

Vous pouvez modifier `run_dimension_analysis.py` pour :

### Changer les dimensions testées

```python
dimensions = [10, 50, 100, 500]  # Vos dimensions
```

### Changer les algorithmes

```python
algorithms = ['Alg7', 'Alg8_m2']  # Seulement 2 algos rapides
```

### Ajuster les itérations

```python
iid_times_k = plot_autocorr_time_vs_dimension(
    dimensions=dimensions,
    n_iter=2000,   # Plus d'itérations = meilleure précision
    burn_in=200,
    n_jobs=-1
)
```

### Désactiver la parallélisation

```python
n_jobs=1  # Exécution séquentielle
```

## 📈 Exemple de sortie

```
================================================================================
Dimension Analysis: Time for 2 i.i.d. samples vs Data Dimension
================================================================================

Using 8 parallel processes

================================================================================
Testing dimension: 64
================================================================================
Running Algorithm 3...
Running Algorithm 5...
...

Time for 2 i.i.d. samples (ms) - autocorrelation on k:
--------------------------------------------------------------------------------
Algorithm      n=2          n=4          n=8          n=16         ...
Alg3           45.23        89.12        178.45       356.89       ...
Alg5           52.34        104.67       209.34       418.68       ...
...
```

## 🐛 Troubleshooting

### Erreur "ModuleNotFoundError: No module named 'scipy'"

Installez les dépendances :
```bash
pip3 install scipy matplotlib numpy
```

### Problème avec multiprocessing sur Windows

Sur Windows, assurez-vous que le code est dans un bloc `if __name__ == "__main__":` (déjà fait).

### Utilisation excessive de RAM

Réduisez le nombre de processus parallèles :
```python
n_jobs=4  # Au lieu de -1
```

Ou réduisez le nombre d'itérations :
```python
n_iter=500  # Au lieu de 1000
```

## 💡 Conseils

1. **Commencez par le test** : `python3 test_parallel.py`
2. **Surveillez l'utilisation CPU** : Vous devriez voir tous vos CPUs à ~100%
3. **Pour les grandes dimensions** : Considérez réduire le nombre d'algorithmes
4. **Sauvegarde** : Les résultats intermédiaires sont sauvegardés au fur et à mesure

## 🎯 Performance

Gain de temps avec parallélisation (exemple avec 8 CPUs) :

| Exécution | Temps estimé |
|-----------|--------------|
| Séquentielle (n_jobs=1) | ~80 minutes |
| Parallèle (n_jobs=-1, 8 CPUs) | ~10-15 minutes |

**Accélération : ~5-6x**
