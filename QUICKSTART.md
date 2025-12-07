# Démarrage Rapide

## Étape 1 : Installer les dépendances

```bash
pip3 install numpy scipy matplotlib
```

Si vous avez l'erreur "externally-managed-environment", utilisez un environnement virtuel :

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib
```

## Étape 2 : Tester l'installation

```bash
python3 test_parallel.py
```

✅ Si ça marche, vous verrez :
- Le nombre de CPUs détectés
- Les algorithmes qui tournent en parallèle
- Un graphique généré dans `results/`

## Étape 3 : Lancer l'analyse complète

```bash
python3 run_dimension_analysis.py
```

Cela va :
- Tester 9 dimensions : [2, 4, 8, 16, 32, 64, 128, 256, 512]
- Comparer 6 algorithmes en parallèle
- Générer `results/iid_time_vs_dimension.png`

**Durée estimée :** 10-15 minutes (avec 8 CPUs)

## Alternative : Reproduire les résultats du papier

Si vous voulez d'abord vérifier que les algorithmes reproduisent le papier de Neal (1998) :

```bash
python3 run_paper_example.py
```

Cela teste les 8 algorithmes sur les 9 points de données du papier et compare avec la Table 1.

## 📊 Résultats

Tous les résultats sont dans `results/` :
- `iid_time_vs_dimension.png` : **Graphique principal (temps i.i.d. vs dimension)**
- `results_dim{n}.pkl` : Résultats détaillés
- `results_dim{n}.json` : Résumés

## 🔧 Personnalisation rapide

Pour tester seulement quelques dimensions rapidement, éditez `run_dimension_analysis.py` ligne 25 :

```python
dimensions = [10, 50, 100]  # Au lieu de [2, 4, 8, ..., 512]
```

Et ligne 28 pour limiter les algorithmes :

```python
algorithms = ['Alg7', 'Alg8_m2']  # Au lieu de tous les 6
```

## ❓ Problèmes ?

Consultez `PARALLEL_EXECUTION.md` pour plus de détails et le troubleshooting.
