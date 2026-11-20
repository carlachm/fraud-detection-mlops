# Contribuer

## Workflow Git

1. Une **issue** par tâche, assignée à quelqu'un.
2. Une **branche** par issue, créée depuis `main` à jour :
   `feat/api`, `fix/threshold`, `ci/docker-push`, `docs/readme`…
3. Des **commits** courts au format [Conventional Commits](https://www.conventionalcommits.org/fr/) :
   `feat: add /v1/predict route`, `fix: ...`, `test: ...`, `ci: ...`, `docs: ...`, `chore: ...`
4. Une **pull request** vers `main` (le template se remplit tout seul), avec `Closes #<issue>`.
5. La CI doit être verte (`lint`, `test`, `docker`) et **1 review** est obligatoire avant le merge.
6. Merge en *squash*, puis suppression de la branche.

Personne ne pousse directement sur `main` (branche protégée).

## Avant d'ouvrir une PR

```bash
make lint      # mêmes vérifications que la CI
make test
make docker-smoke   # si le Dockerfile ou les dépendances ont changé
```

## Dépendances

On utilise **uv**, jamais `pip install` dans le projet :

```bash
uv add <package>          # dépendance d'exécution
uv add --group test <pkg> # dépendance de test
git add pyproject.toml uv.lock
```

`uv.lock` est toujours commité ; `.venv/` jamais.
