# Détection de fraude bancaire — Projet MLOps

Pipeline MLOps complet pour la détection de fraude sur cartes bancaires
(classification binaire, dataset [Kaggle creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).

> **État actuel : séances 1 à 3** — reproductibilité (uv, DVC), entraînement PyTorch, tracking MLflow,
> Model Registry, stockage des modèles sur MinIO (S3), CI/CD GitHub Actions + image Docker Hub.
> Prochaines étapes : API FastAPI (S4), Kubernetes (S5), monitoring (S6), drift & réentraînement (S7).

[![ci](https://github.com/carlachm/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/carlachm/fraud-detection-mlops/actions/workflows/ci.yml)

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
├── .github/
│   ├── workflows/ci.yml     # lint + tests + build/smoke test/push Docker
│   ├── workflows/train.yml  # entraînement + quality gate (runner self-hosted)
│   └── pull_request_template.md
├── docker/mlflow/     # image du serveur MLflow
├── scripts/           # setup_dvc.sh, smoke_test.sh
├── Dockerfile         # image d'entraînement (uv, multi-arch)
├── docker-compose.yml # environnement de dev
├── dvc.yaml           # pipeline DVC : prepare -> train
├── params.yaml        # TOUS les hyperparamètres (source unique)
├── pyproject.toml     # dépendances (ce qu'on demande)
├── uv.lock            # versions exactes (ce qu'on obtient) — commité
├── CONTRIBUTING.md    # workflow Git de l'équipe
└── Makefile           # raccourcis (make help)
```

## Démarrage rapide

Prérequis : Docker Desktop, Git, [uv](https://docs.astral.sh/uv/) (`brew install uv`), un compte Kaggle (clé API).

```bash
# 1. Environnement Python (uv installe Python 3.12 et crée .venv tout seul)
make install          # = uv sync --locked

# 2. Variables d'environnement (remplir KAGGLE_USERNAME / KAGGLE_KEY)
cp .env.example .env

# 3. Stack de dev : MinIO + PostgreSQL + MLflow
make up
#   MLflow UI     -> http://localhost:5000
#   MinIO console -> http://localhost:9001  (minioadmin / minioadmin123)

# 4. Données : init DVC, téléchargement Kaggle, versioning sur MinIO
make dvc-setup
make data           # download + dvc add + dvc push
git add data/raw/creditcard.csv.dvc .dvc .gitignore && git commit -m "data: add raw dataset v1"

# 5. Pipeline complet (prepare -> train), ré-exécuté seulement si quelque chose change
make repro
dvc metrics show
```

Pas de compte Kaggle sous la main ? `make sample` génère un dataset synthétique au même format.

> Le remote DVC est le MinIO **local** de chacun : un coéquipier refait `make data`
> (même fichier Kaggle → même hash que le `.dvc` commité, `dvc status` est propre).

## Séance 1 — Reproductibilité

| Problème | Solution dans le projet |
|---|---|
| « Ça marche sur ma machine » | **uv** : `pyproject.toml` + `uv.lock` (versions exactes + hashes), `uv sync --locked` partout ; services en **Docker Compose** |
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

**Quality gate** : un modèle n'est promu que si son PR-AUC de test dépasse `quality_gate.min_test_pr_auc`
(`params.yaml`) **et** la baseline aléatoire (PR-AUC d'un classifieur au hasard = taux de fraude, loggé en
`baseline_random_pr_auc`). Sinon la version reçoit `promotion=gate_failed` et la commande sort en erreur
→ le job CI / le stage DVC passe au rouge.

## Séance 3 — CI/CD avec GitHub Actions

```
pull request ──▶ lint ──┐
                        ├──▶ docker : build ─▶ smoke test ─▶ (main uniquement) push Docker Hub
                 test ──┘                                      multi-arch amd64 + arm64
```

| Workflow | Déclencheur | Ce qu'il fait |
|---|---|---|
| `ci.yml` › `lint` | PR + push `main` | `uv lock --check` (lock à jour), `ruff check`, `ruff format --check` |
| `ci.yml` › `test` | PR + push `main` | pytest (validation des données, modèle, registre MLflow), rapports couverture/JUnit en artefacts |
| `ci.yml` › `docker` | après lint + test | build de l'image, **smoke test** (pipeline complet dans le conteneur sur données synthétiques), puis sur `main` push sur Docker Hub taggé avec le commit |
| `train.yml` | bouton / appelé par un autre workflow | `dvc pull` + `dvc repro` sur le runner self-hosted, quality gate, promotion MLflow |

Bonnes pratiques appliquées (cf. cours) :

- actions **épinglées par SHA** (version en commentaire) — une étiquette `@v4` peut être déplacée, un SHA non ;
- `permissions: contents: read` (moindre privilège), `timeout-minutes` sur chaque job, `concurrency` pour annuler les runs obsolètes ;
- installation depuis le lock (`uv sync --locked`) avec cache ; un groupe de dépendances par job (`--only-group lint`) ;
- image **taguée avec le commit**, jamais `latest` ; cache des couches Docker (`type=gha`) ;
- on teste le **conteneur**, pas seulement le build ;
- pas d'entraînement lourd à chaque push : `train.yml` est manuel ;
- secrets jamais dans le code : `DOCKERHUB_TOKEN` dans les *secrets* GitHub.

Image : `docker pull <DOCKERHUB_USERNAME>/fraud-detection-train:<sha>` — _URL Docker Hub à compléter_.

### Configuration GitHub (une fois)

1. **Docker Hub** → *Account settings → Personal access tokens* → token *Read & Write*.
2. **GitHub** → *Settings → Secrets and variables → Actions* :
   - onglet *Variables* : `DOCKERHUB_USERNAME` = ton identifiant Docker Hub (en minuscules) ;
   - onglet *Secrets* : `DOCKERHUB_TOKEN` = le token ; `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` (pour `train.yml`).
3. **Settings → Branches → Add rule** sur `main` : *Require a pull request* (1 approbation) +
   *Require status checks to pass* → cocher `lint`, `test`, `docker`.

Le workflow de l'équipe (branches, commits, PR) est décrit dans [CONTRIBUTING.md](CONTRIBUTING.md).

## Serving — API, application web et Kubernetes

```
navigateur ──▶ ride-webapp (Streamlit, :8501) ──▶ ride-api (FastAPI, :8000) ──▶ MLflow registry (models:/ride-classifier@champion)
                                                      │
                                                      ├─▶ /metrics (Prometheus)
                                                      └─▶ logs/predictions.jsonl (monitoring, dérive)
```

| Élément | Fichier | Rôle |
|---|---|---|
| API | `src/fraud_detection/api/` | `/v1/predict`, `/v1/feedback`, `/health`, `/ready`, `/metrics`, Swagger sur `/docs` |
| Application web | `webapp/` | trajet de démo ou fichier CSV/Parquet (un ou plusieurs trajets), probabilités des 5 classes, retour de label |
| Images | `docker/api/`, `docker/webapp/` | utilisateur non-root, healthcheck, l'image web n'embarque ni torch ni mlflow |
| Kubernetes | `k8s/` | 2 réplicas d'API (RollingUpdate sans indisponibilité), sondes startup/readiness/liveness, requests/limits |
| CI | `ci.yml` › `serving` | build des 2 images, **smoke test des conteneurs**, push Docker Hub sur `main` |
| CD | `deploy.yml` | runner **self-hosted** : déploie sur le cluster local les images de CI et vérifie une prédiction via le Service ; rollback automatique si le rollout échoue |
| Charge | `loadtest/locustfile.py` | `make load` |

Contrat d'interface complet (format du modèle, de l'API, des logs et des métriques) : [docs/api-contract.md](docs/api-contract.md).

```bash
make up && make placeholder   # MLflow + modèle factice (jusqu'à l'entraînement du vrai modèle)
make api                      # http://localhost:8000/docs
make webapp                   # http://localhost:8501
make k8s-up                   # cluster kind local + déploiement (API sur :8080, web sur :8501)
# ou, avec Kubernetes de Docker Desktop (testé) :
make serving-build && MLFLOW_URI_FOR_PODS=http://host.docker.internal:5000 bash scripts/deploy_k8s.sh ride-api:dev ride-webapp:dev
make k8s-forward              # API sur :8080, web sur :8501
```

Sur macOS, le port 5000 est pris par AirPlay Receiver : mettre `MLFLOW_PORT=5050` dans `.env` (MLflow sera sur
http://localhost:5050) et lancer le déploiement avec `MLFLOW_URI_FOR_PODS=http://host.docker.internal:5050`.

_À compléter avec les captures (Swagger, application web, `kubectl get pods`) et les mesures de charge._

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
make help          # liste des commandes
make test          # tests + couverture
make lint          # mêmes vérifications que la CI
make docker-smoke  # build + smoke test de l'image, comme la CI
make train         # un entraînement (hors DVC)
make down          # arrêter la stack
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
  vérification de l'enregistrement, de l'alias `@champion`, du **quality gate**, et chargement du modèle
  via `mlflow.pyfunc` (exactement ce que fera l'API).
- `scripts/smoke_test.sh` : test **end-to-end** de l'image Docker (lancé par la CI).

Les tests utilisent un dataset synthétique avec le même schéma que Kaggle (pas besoin de télécharger les données).

## Résultats

_À compléter après l'entraînement sur le vrai dataset (captures de l'UI MLflow, PR-AUC, matrice de confusion)._

## Équipe

_À compléter._
