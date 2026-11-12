# Détection de fraude bancaire — Projet MLOps

Pipeline MLOps complet pour la détection de fraude sur cartes bancaires
(classification binaire, dataset [Kaggle creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).

> **État actuel : séances 1 & 2** — reproductibilité, versioning des données (DVC),
> entraînement PyTorch, tracking MLflow, Model Registry, stockage des modèles sur MinIO (S3).
> Prochaines étapes : CI/CD (S3), API FastAPI (S4), Kubernetes (S5), monitoring (S6), réentraînement (S7).

---

## Architecture (dev)

```
                 ┌──────────────────────── docker compose ────────────────────────┐
                 │                                                                 │
 python -m       │   ┌──────────────┐  métadonnées   ┌─────────────┐              │
 fraud_detection ├──▶│ MLflow server│───────────────▶│ PostgreSQL  │              │
 .train          │   │  :5000       │                └─────────────┘              │
                 │   │ tracking +   │  artefacts     ┌─────────────────────────┐  │
                 │   │ registry     │───────────────▶│ MinIO (S3)  :9000/:9001 │  │
                 │   └──────────────┘                │  bucket mlflow/ (modèles)│  │
 dvc push/pull ──┼──────────────────────────────────▶│  bucket dvc/   (données) │  │
                 │                                   └─────────────────────────┘  │
                 └─────────────────────────────────────────────────────────────────┘
```

- **PostgreSQL** stocke les runs, paramètres, métriques et le registre de modèles.
- **MinIO** (compatible S3) stocke les modèles et graphiques (bucket `mlflow`) et les données versionnées par DVC (bucket `dvc`).
- Le serveur MLflow tourne en mode `--serve-artifacts` : les scripts d'entraînement (et plus tard l'API) passent par lui pour lire/écrire les artefacts, ils n'ont pas besoin des identifiants S3.

## Structure du repo

```
├── src/fraud_detection/
│   ├── config.py      # chargement de params.yaml, seed (reproductibilité)
│   ├── data.py        # téléchargement Kaggle, validation, split stratifié
│   ├── model.py       # MLP PyTorch (normalisation intégrée au modèle)
│   ├── evaluate.py    # métriques adaptées au déséquilibre, graphiques
│   ├── registry.py    # promotion du modèle (alias @champion)
│   └── train.py       # entraînement + logging MLflow + enregistrement
├── tests/             # tests unitaires + test d'intégration MLflow
├── docker/mlflow/     # image du serveur MLflow
├── scripts/setup_dvc.sh
├── docker-compose.yml # environnement de dev
├── dvc.yaml           # pipeline DVC : prepare -> train
├── params.yaml        # TOUS les hyperparamètres (source unique)
├── requirements*.txt  # dépendances épinglées
└── Makefile           # raccourcis (make help)
```

## Démarrage rapide

Prérequis : Docker Desktop, Python 3.11–3.13, Git, un compte Kaggle (clé API).

```bash
# 1. Environnement Python
python -m venv .venv && source .venv/bin/activate
make install

# 2. Variables d'environnement (remplir KAGGLE_USERNAME / KAGGLE_KEY)
cp .env.example .env

# 3. Stack de dev : MinIO + PostgreSQL + MLflow
make up
#   MLflow UI     -> http://localhost:5000
#   MinIO console -> http://localhost:9001  (minioadmin / minioadmin123)

# 4. Données : init DVC, téléchargement Kaggle, versioning sur MinIO
git init            # si le repo n'est pas déjà cloné
make dvc-setup
make data           # download + dvc add + dvc push
git add data/raw/creditcard.csv.dvc .dvc .gitignore && git commit -m "data: add raw dataset v1"

# 5. Pipeline complet (prepare -> train), ré-exécuté seulement si quelque chose change
make repro
dvc metrics show
```

Un coéquipier qui clone le repo récupère exactement les mêmes données avec `dvc pull`.

## Séance 1 — Reproductibilité

| Problème | Solution dans le projet |
|---|---|
| « Ça marche sur ma machine » | Dépendances **épinglées** (`requirements.txt`), services en **Docker Compose** |
| Hyperparamètres éparpillés | Un seul fichier `params.yaml`, loggé dans MLflow à chaque run |
| Résultats différents à chaque run | `set_seed()` (Python, NumPy, PyTorch) + split stratifié avec `random_state` |
| « Sur quelles données ce modèle a-t-il été entraîné ? » | **DVC** : le hash des données est dans Git, les fichiers dans MinIO |
| Étapes lancées à la main dans le désordre | Pipeline **`dvc.yaml`** : `dvc repro` relance uniquement les étapes impactées |
| Données de mauvaise qualité | `validate()` : schéma, valeurs nulles, cible binaire, montants ≥ 0 |
| Fuite de données | Normalisation calculée **sur le train uniquement**, seuil choisi sur la validation, test jamais vu |

## Séance 2 — MLflow Tracking & Model Registry

Chaque `python -m fraud_detection.train` crée un run dans l'expérience `fraud-detection` avec :

- **Paramètres** : tout `params.yaml` (aplati : `train.learning_rate`, `model.hidden_dims`…), taille des données, `pos_weight`.
- **Métriques** :
  - par epoch : `train_loss`, `val_pr_auc_epoch`, `val_roc_auc_epoch` (courbes dans l'UI) ;
  - finales : `val_*` et `test_*` (PR-AUC, ROC-AUC, précision, rappel, F1, TP/FP/FN/TN), `threshold`.
- **Artefacts** : courbe précision-rappel, matrice de confusion, statistiques des données, le modèle (avec signature + exemple d'entrée).
- **Tags** : type de modèle, commit Git (ajouté automatiquement par MLflow).

**Model Registry** : chaque modèle est enregistré sous `fraud-detector` (nouvelle version à chaque run).
La version est promue avec l'**alias `@champion`** uniquement si son PR-AUC de test bat celui du champion actuel ;
sinon elle reçoit le tag `promotion=rejected`. Le seuil de décision est stocké en tag de la version.
L'API chargera toujours `models:/fraud-detector@champion` → un rollback = déplacer l'alias.

> On utilise les **alias** plutôt que les *stages* (Staging/Production), dépréciés depuis MLflow 2.9.

## Choix techniques

- **Modèle** : MLP PyTorch (couches 64 → 32, BatchNorm, Dropout). Les données sont tabulaires
  (V1–V28 issues d'une PCA, `Time`, `Amount`), un MLP simple suffit ; l'enjeu du projet est le pipeline autour.
- **Déséquilibre (~0,17 % de fraudes)** :
  - perte `BCEWithLogitsLoss` avec `pos_weight = n_négatifs / n_positifs` ;
  - early stopping et sélection du modèle sur le **PR-AUC** (l'accuracy serait de 99,8 % en prédisant « jamais fraude ») ;
  - seuil de décision optimisé pour le F1 sur la validation (pas 0,5 par défaut).
- **Modèle autonome** : la moyenne/écart-type sont des *buffers* du modèle et la sortie est une probabilité
  → l'API enverra les features brutes, sans preprocessing à dupliquer.
- **Split** : 70/15/15 stratifié, doublons supprimés avant le split.

## Commandes utiles

```bash
make help        # liste des commandes
make test        # tests + couverture
make lint        # ruff
make train       # un entraînement (hors DVC)
make down        # arrêter la stack
```

Changer un hyperparamètre → modifier `params.yaml` → `make repro` → comparer les runs dans l'UI MLflow
(sélectionner plusieurs runs → *Compare*) ou avec `dvc params diff` / `dvc metrics diff`.

## Tests

```bash
make test
```

- `test_data.py` : validation, split stratifié et reproductible, écriture des fichiers.
- `test_model.py` : forme des sorties, probabilités dans [0,1], normalisation, métriques.
- `test_train.py` : **intégration** — entraînement complet sur un MLflow local jetable,
  vérification de l'enregistrement, de l'alias `@champion`, et chargement du modèle via `mlflow.pyfunc`
  (exactement ce que fera l'API).

Les tests utilisent un dataset synthétique avec le même schéma que Kaggle (pas besoin de télécharger les données).

## Résultats

_À compléter après l'entraînement sur le vrai dataset (captures de l'UI MLflow, PR-AUC, matrice de confusion)._

## Équipe

_À compléter._
