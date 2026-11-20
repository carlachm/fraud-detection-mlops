## Quoi ?
<!-- Ce que fait cette PR, en 1-3 phrases -->

## Pourquoi ?
<!-- Lien vers l'issue : Closes #... -->

## Comment tester ?
<!-- Commandes pour vérifier, ex : make test, make docker-smoke, curl ... -->

## Checklist
- [ ] `make lint` et `make test` passent en local
- [ ] Pas de secret, de `.env` ni de gros fichier de données dans le diff
- [ ] Si `pyproject.toml` a changé : `uv.lock` est mis à jour (`make lock`)
- [ ] README / docs mis à jour si besoin
