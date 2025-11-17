# LLM Code Rules (priorisées et synthétiques)

---

## 1. Principes généraux — Obligatoires

* **TRÈS IMPORTANT : ne jamais coder de fallback, valeur par défaut implicite, ou comportement silencieux.**
  Toujours lever une erreur explicite s’il manque un paramètre, fichier ou dépendance.
* **KISS** (Keep It Simple, Stupid) et **DRY** (Don’t Repeat Yourself).
* Respect strict de **PEP 8**.
* Code lisible avant tout et perfomance pour les fonctions critiques.
* Commenter le **pourquoi**, jamais le quoi.

---

## 2. Python (`src/*.py`)

### 2.1 Fonctions et types

* Type hints obligatoires pour tous les paramètres et retours.
* Maximum 40 lignes par fonction, une seule responsabilité.
* Docstrings en anglais (Google ou NumPy style).
* Pas de création de fonction non demandée.

### 2.2 Architecture

* Fonctions réutilisables → `src/utils.py`.
* Constantes, chemins, configs, magic numbers → `src/constants.py` et 'src/path.py' **uniquement**.
* **TRÈS IMPORTANT : aucune constante hardcodée.**
* Structure type : `__init__.py`, fichier principal, `main.py` (CLI), `test_*.py`.
* Import order : stdlib → third-party → local.
* `from __future__ import annotations` toujours en première ligne.
* Pas d’imports globaux `*`.
* Vérifier `requirements.txt` avant toute nouvelle dépendance.

### 2.3 Logging, erreurs et qualité

* Logger standard : `get_logger(__name__)`.
* **INFO** pour les étapes clés.
* Validation stricte des entrées.
* Exceptions claires et explicites, jamais de try/except silencieux.
* Supprimer tout code ou import inutile.
* Vérifier les erreurs avec `get_errors`.
* Formatter avec `black` et `ruff` avant toute validation.

### 2.4 Tests

* Créer un test unitaire pour chaque fonction.
* Données mockées uniquement (pas de vraies données).
* Utiliser pytest + fixtures + monkeypatch si nécessaire.
* Tester les cas normaux et extrêmes (vide, None, invalide).
* Nommage standard : `test_<module_name>.py`.

---

## 3. Modules Machine Learning

* **Reproductibilité obligatoire** : `DEFAULT_RANDOM_STATE` + seed fixée (numpy, pandas, sklearn, LightGBM, etc.).
* **Aucune fuite temporelle** : jamais de fit sur des données futures.

---

## 4. Interaction LLM

* Ne jamais supposer de méthode implicite.
* Demander clarification si le but ou le type de fichier n’est pas clair.
* Appliquer uniquement les règles correspondantes.
