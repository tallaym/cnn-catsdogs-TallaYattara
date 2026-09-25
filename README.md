# CNN "from scratch" vs Transfer Learning — Cats vs Dogs

## Objectif

Comparer un CNN entraîné from scratch et un modèle en transfert d'apprentissage
(ResNet50 pré-entraîné sur ImageNet) sur la classification binaire chats/chiens, et
mesurer l'impact du transfert learning sur la convergence, la performance et la
robustesse.

## Reproduire les expériences

1. `pip install -r requirements.txt` (Python ≥ 3.9 recommandé)
2. Télécharger `Cat_Dog_data` et le placer à la racine du repo, avec la structure
   décrite ci-dessous.
3. Vérifier que `training_utils.py` est dans le **même dossier** que `notebook.ipynb`
   (le notebook fait `from training_utils import *`).
4. Lancer Jupyter (`jupyter notebook`) ou ouvrir le notebook dans Colab/Kaggle, puis
   **exécuter toutes les cellules dans l'ordre, de haut en bas** (`Run All`) — les
   cellules dépendent des variables définies plus haut (`data_loaders`, `n_classes`,
   `classes`, modèles, etc.), un ordre différent cassera l'exécution.
5. Les 4 checkpoints (`best_scratch_adam.pt`, `best_scratch_sgd.pt`,
   `tl_adam_finetune.pt`, `tl_sgd_finetune.pt`) et les métriques par epoch sont générés
   automatiquement pendant l'exécution ; les résultats de test et matrices de confusion
   s'affichent en sortie des cellules correspondantes.

**Note sur la reproductibilité exacte :** le seed (42) garantit une initialisation des
poids et un split train/valid identiques à chaque exécution. En revanche,
`torch.backends.cudnn.benchmark = True` et le mixed precision (AMP), activés dans
`training_utils.py` pour accélérer l'entraînement sur GPU, introduisent une légère
non-déterminisme au niveau des opérations CUDA. Les résultats reproduits seront donc
très proches de ceux rapportés ci-dessous (mêmes tendances, écarts de l'ordre du
dixième de point de pourcentage), mais pas strictement identiques bit-à-bit.

## Environnement

```bash
pip install -r requirements.txt
```

`requirements.txt` minimal :
```
torch
torchvision
torchmetrics
matplotlib
numpy
pandas
seaborn
tqdm
```

**GPU** : entraînement effectué sur GPU (vérifié via `torch.cuda.is_available()` → `True`,
device `cuda`, GPU T4). Le code bascule automatiquement sur CPU si aucun GPU n'est
disponible (beaucoup plus lent : voir Limites).

## Organisation des données

Dataset : Cats vs Dogs (même corpus que vu en cours). **Non versionné sur GitHub**
(voir `.gitignore`).

Télécharger et placer sous la racine du projet :
```
Cat_Dog_data/
├─ train/
│  ├─ cat/
│  └─ dog/
└─ test/
   ├─ cat/
   └─ dog/
```

Un split **train/validation (85/15)** est fait par-dessus `train/` via
`make_data_loaders(..., valid_size=0.15)`, nécessaire pour le suivi valid loss/acc et le
scheduler `ReduceLROnPlateau`. `test/` reste strictement séparé, utilisé uniquement pour
l'évaluation finale.

## Reproductibilité

Seed fixé à `42` (`random`, `numpy`, `torch`, `torch.cuda`) dans `training_utils.py`.

## Entraînement

Ouvrir `notebook.ipynb` et exécuter les cellules dans l'ordre. Hyperparamètres clés par
expérience (modifiables directement dans la cellule correspondante) :

| Expérience | Optimiseur | LR | Epochs | Batch size | Dropout | Batch Norm | Scheduler |
|---|---|---|---|---|---|---|---|
| From scratch | Adam | 1e-3 | 20 | 32 | 0.4 (avant la tête dense) | après chaque conv (×4) | ReduceLROnPlateau |
| From scratch | SGD (momentum 0.9) | 1e-2 | 20 | 32 | 0.4 | après chaque conv (×4) | ReduceLROnPlateau |
| Transfer learning | Adam | 5e-3 (tête) → 5e-5 (fine-tune) | 5 + 10 | 32 | 0.4 (tête) | héritée du backbone ResNet50 | ReduceLROnPlateau |
| Transfer learning | SGD (momentum 0.9) | 1e-2 (tête) → 1e-4 (fine-tune) | 5 + 10 | 32 | 0.4 (tête) | héritée du backbone ResNet50 | ReduceLROnPlateau |

Le LR de départ pour le CNN from scratch a été choisi via `lr_finder` (rampe
exponentielle de LR + repérage du point de divergence sur la courbe loss/LR).

**Pourquoi Dropout + BatchNorm, et où :**
- **BatchNorm** après chaque couche conv, avant l'activation : stabilise la
  distribution des activations dès le début de l'entraînement pour le modèle from
  scratch (poids initialisés aléatoirement, pas de pré-entraînement).
- **Global Average Pooling** avant la tête dense, plutôt qu'un flatten complet : réduit
  drastiquement le nombre de paramètres de la tête, limite l'overfitting.
- **Dropout (0.4)** juste avant la couche de sortie finale, sur les deux modèles :
  limite l'overfitting sur un jeu de données de taille modeste, sans perturber
  l'extraction de features des couches convolutionnelles.

Transfer learning en 2 phases : phase 1 = tête seule entraînée, backbone gelé ; phase 2
= fine-tuning de l'ensemble avec un LR réduit d'un facteur 100.

## Évaluation / rechargement du modèle

```python
model.load_state_dict(torch.load("best_scratch_adam.pt"))  # ou tl_adam_finetune.pt, etc.
test_loss, preds, actuals = test_model(data_loaders["test"], model, loss)
```

Checkpoints locaux (non versionnés) : `best_scratch_adam.pt`, `best_scratch_sgd.pt`,
`tl_adam_finetune.pt`, `tl_sgd_finetune.pt`.

## Résultats (test set, 2500 images)

| Modèle | Test Loss | Test Accuracy |
|---|---|---|
| From scratch + Adam | 0.3865 | 83.64% (2091/2500) |
| From scratch + SGD | 0.4836 | 77.44% (1936/2500) |
| Transfer learning + Adam (fine-tuné) | 0.0276 | 98.76% (2469/2500) |
| Transfer learning + SGD (fine-tuné) | 0.0265 | **99.00%** (2475/2500) |

Matrices de confusion (test set) :

| | Scratch+Adam | Scratch+SGD | TL+Adam | TL+SGD |
|---|---|---|---|---|
| Cat → Cat | 1031 | 908 | 1242 | 1236 |
| Cat → Dog (erreur) | 219 | 342 | 8 | 14 |
| Dog → Dog | 1060 | 1028 | 1227 | 1239 |
| Dog → Cat (erreur) | 190 | 222 | 23 | 11 |

Nombre de paramètres entraînables : CNN from scratch ≈ **390K** (grâce au Global
Average Pooling) vs ResNet50 ≈ **25M**.

**Analyse.** Le transfert learning surclasse nettement le CNN from scratch (~98-99%
contre ~77-84% d'accuracy), avec en plus une loss de test 10 à 15 fois plus faible. Ceci
confirme l'intérêt du transfert learning sur un dataset de taille modeste : le backbone
ResNet50, pré-entraîné sur ImageNet (qui contient déjà des images de chats et de
chiens), fournit des features visuelles déjà pertinentes, alors que le CNN from scratch
doit tout apprendre — des contours de bas niveau jusqu'à la sémantique de la tâche —
avec beaucoup moins de données et de paramètres disponibles.

Entre les deux optimiseurs, Adam converge plus vite que SGD sur peu d'epochs (utile
pour le from scratch), mais SGD+momentum, avec plus d'epochs et une convergence plus
progressive, rattrape voire dépasse légèrement Adam en transfer learning (99.00% vs
98.76%). Le CNN from scratch montre par ailleurs une valid loss plus instable d'une
epoch à l'autre (pics visibles aux epochs 11-12 et 19 pour Adam) — signe d'un modèle
plus sensible au bruit d'entraînement sur un petit dataset, alors que le transfer
learning converge de façon nettement plus régulière.

Les erreurs résiduelles du transfer learning (8 à 23 images sur 2500 selon le modèle)
sont vraisemblablement des cas ambigus : angle de prise de vue inhabituel, occlusion
partielle de l'animal, ou races dont l'apparence chevauche l'autre classe.

## Limites & pistes d'amélioration

- Le `lr_finder` utilise un SGD sans momentum en interne, ce qui donne une estimation
  approximative du LR optimal pour Adam ou un SGD avec momentum — à affiner si plus de
  temps disponible.
- Le fine-tuning du transfer learning dégèle tout le backbone d'un coup ; un dégel
  progressif par blocs (discriminative fine-tuning, LR différent par couche) serait
  plus robuste sur un dataset encore plus petit.
- Pas d'early stopping formel (seul le meilleur checkpoint est conservé) : le nombre
  d'epochs a été fixé empiriquement plutôt que déterminé par un critère d'arrêt.
- Entraînement dépendant de la disponibilité GPU gratuite (Google Colab) : temps
  d'entraînement multiplié par 20-30x en cas de bascule CPU, ce qui a contraint le
  nombre d'expériences supplémentaires (ex. comparaison fine de plusieurs LR) réalisées
  dans le temps imparti.
