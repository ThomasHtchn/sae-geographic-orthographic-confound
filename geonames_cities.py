"""
Récupère des villes depuis la base Geonames (Italie, Grande-Bretagne, Australie,
et 4 états américains), puis entraîne une régression logistique (pénalité L1)
à prédire le pays à partir des n-grammes de caractères du nom de la ville, en
sélectionnant le plus petit sous-ensemble de tokens qui préserve une bonne accuracy.
"""

import io
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

SMOLLM3_MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
GEONAMES_DUMP_URL = "https://download.geonames.org/export/dump/{code}.zip"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geonames_cache")

# Grille de forces de régularisation L1 balayée pour trouver le modèle le plus parcimonieux
L1_C_VALUES = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
# Écart d'accuracy toléré par rapport au meilleur modèle du balayage, pour privilégier la parcimonie
L1_ACCURACY_TOLERANCE = 0.01

# Colonnes du format "dump" de Geonames (fichiers <CODE_PAYS>.zip)
GEONAMES_COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
    "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]

# country_code Geonames -> (label pays à prédire, seuil de population minimal, admin1_codes à garder ou None)
COUNTRIES = {
    "IT": {"label": "Italy", "min_population": 500, "admin1_codes": None},
    "GB": {"label": "UK", "min_population": 1, "admin1_codes": None},
    "AU": {"label": "Australia", "min_population": 500, "admin1_codes": None},
    "PL": {"label": "Poland", "min_population": 500, "admin1_codes": None},
    "US": {
        "label": "USA",
        "min_population": 1,
        # Geonames utilise l'abréviation postale de l'état comme admin1_code pour les US
        "admin1_codes": {"NY", "NJ", "CT", "PA"},
    },
}


def download_country_dump(country_code: str) -> pd.DataFrame:
    """Télécharge (avec cache local) et charge le fichier dump Geonames d'un pays."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{country_code}.txt")

    if not os.path.exists(cache_path):
        url = GEONAMES_DUMP_URL.format(code=country_code)
        print(f"Téléchargement de {url} ...")
        with urllib.request.urlopen(url) as response:
            zip_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            with zf.open(f"{country_code}.txt") as f, open(cache_path, "wb") as out:
                out.write(f.read())

    return pd.read_csv(
        cache_path,
        sep="\t",
        header=None,
        names=GEONAMES_COLUMNS,
        quoting=3,  # QUOTE_NONE : certains noms contiennent des guillemets
        na_values="",
        keep_default_na=False,
        dtype={"admin1_code": str},
        low_memory=False,
    )


def build_cities_dataframe() -> pd.DataFrame:
    """Construit le dataframe final (colonnes: city, country) pour tous les pays configurés."""
    frames = []

    for country_code, conf in COUNTRIES.items():
        df = download_country_dump(country_code)

        # feature_class == "P" : "populated place" (ville, village, etc.)
        df = df[df["feature_class"] == "P"]
        df = df[df["population"] >= conf["min_population"]]

        if conf["admin1_codes"] is not None:
            df = df[df["admin1_code"].isin(conf["admin1_codes"])]

        cities = pd.DataFrame({
            "city": df["name"],
            "country": conf["label"],
        })
        frames.append(cities)

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates().reset_index(drop=True)
    return result


def detect_word_start_marker(tokenizer) -> str:
    """Détecte le marqueur qu'un tokenizer BPE byte-level utilise pour représenter un
    espace initial (ex: "Ġ"), en comparant la tokenisation d'un mot seul et précédé d'un espace."""
    for probe in ("the", "and", "test", "paris"):
        plain_tokens = tokenizer.tokenize(probe)
        spaced_tokens = tokenizer.tokenize(" " + probe)
        if len(plain_tokens) == 1 and len(spaced_tokens) == 1:
            plain, spaced = plain_tokens[0], spaced_tokens[0]
            if spaced != plain and spaced.endswith(plain):
                return spaced[: len(spaced) - len(plain)]
    return " "


def load_smollm3_vocabulary() -> list[str]:
    """Récupère le vocabulaire du tokenizer de SmolLM3-3B, ordonné par id de token.

    Les tokens de début de mot sont préfixés par le marqueur d'espace du tokenizer
    (ex: "Ġparis") plutôt que par un espace normal. CountVectorizer(analyzer="char_wb")
    délimite au contraire chaque mot par un espace normal (" paris "), ce qui empêcherait
    tout token de début de mot de matcher un n-gramme. On réécrit donc ce marqueur en
    espace normal pour que ces tokens s'alignent avec les n-grammes de char_wb.
    """
    tokenizer = AutoTokenizer.from_pretrained(SMOLLM3_MODEL_NAME)
    marker = detect_word_start_marker(tokenizer)

    vocab = tokenizer.get_vocab()  # token (str) -> id
    tokens = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]

    if marker != " ":
        tokens = [
            " " + token[len(marker):] if token.startswith(marker) else token
            for token in tokens
        ]
    return tokens


def vectorize_city_names(
    city_names: pd.Series, vocabulary: list[str]
) -> tuple[CountVectorizer, "scipy.sparse.spmatrix"]:
    """Vectorise les noms de ville en n-grammes de caractères (2 à 6 lettres), restreints
    au vocabulaire du tokenizer de SmolLM3-3B."""
    vectorizer = CountVectorizer(
        analyzer="char_wb", ngram_range=(2, 6), lowercase=True, vocabulary=vocabulary
    )
    X = vectorizer.fit_transform(city_names)
    return vectorizer, X


def make_l1_model(C: float) -> LogisticRegression:
    # solver="saga" est le seul solveur de LogisticRegression qui supporte la pénalité L1
    # avec plusieurs classes (multinomial). l1_ratio=1 (plutôt que penalty="l1", déprécié
    # depuis sklearn 1.8) donne une pénalité L1 pure.
    return LogisticRegression(solver="saga", C=C, l1_ratio=1.0, max_iter=3000, tol=1e-3)


def count_selected_tokens(model: LogisticRegression) -> int:
    """Nombre de tokens (colonnes) avec un coefficient non nul pour au moins une classe."""
    return int(np.count_nonzero(np.any(model.coef_ != 0, axis=0)))


def select_sparsest_model(X_train, y_train, X_test, y_test) -> tuple[LogisticRegression, float, int]:
    """Balaie plusieurs forces de régularisation L1 (L1_C_VALUES) et retient le modèle
    utilisant le moins de tokens dont l'accuracy reste à moins de L1_ACCURACY_TOLERANCE
    de la meilleure accuracy observée sur le balayage."""
    results = []
    for C in L1_C_VALUES:
        model = make_l1_model(C)
        model.fit(X_train, y_train)
        results.append({
            "C": C,
            "model": model,
            "accuracy": model.score(X_test, y_test),
            "n_tokens": count_selected_tokens(model),
        })

    best_accuracy = max(r["accuracy"] for r in results)
    candidates = [r for r in results if r["accuracy"] >= best_accuracy - L1_ACCURACY_TOLERANCE]
    chosen = min(candidates, key=lambda r: r["n_tokens"])

    print("\nBalayage de la régularisation L1 (C, tokens sélectionnés, accuracy test) :")
    for r in sorted(results, key=lambda r: r["C"]):
        flag = ""
        if r is chosen:
            flag = f"  <- retenu (le plus parcimonieux à moins de {L1_ACCURACY_TOLERANCE:.0%} du meilleur)"
        print(f"  C={r['C']:<8g} tokens={r['n_tokens']:<7d} accuracy={r['accuracy']:.3f}{flag}")

    return chosen["model"], chosen["C"], chosen["n_tokens"]


def top_tokens_per_class(
    model: LogisticRegression, feature_names: np.ndarray, top_n: int = 10
) -> dict[str, list[tuple[str, float]]]:
    """Pour chaque classe (pays), renvoie les top_n tokens dont le coefficient est le plus
    élevé, c'est-à-dire ceux qui contribuent le plus à prédire ce pays."""
    top_by_class = {}
    for class_index, country in enumerate(model.classes_):
        coefs = model.coef_[class_index]
        top_indices = np.argsort(coefs)[::-1][:top_n]
        top_by_class[country] = [(feature_names[i], coefs[i]) for i in top_indices]
    return top_by_class


def train_and_evaluate(df: pd.DataFrame, vocabulary: list[str], experiment_name: str) -> None:
    """Vectorise, sélectionne le modèle L1 le plus parcimonieux et affiche les métriques."""
    print(f"\n===== {experiment_name} =====")
    print(df["country"].value_counts())

    vectorizer, X = vectorize_city_names(df["city"], vocabulary)
    y = df["country"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, chosen_C, n_tokens = select_sparsest_model(X_train, y_train, X_test, y_test)
    print(f"\nModèle retenu : C={chosen_C:g} -- {n_tokens} tokens sélectionnés sur {X.shape[1]}")

    cv_scores = cross_val_score(make_l1_model(chosen_C), X_train, y_train, cv=5)
    print(f"Accuracy en validation croisée (5 folds) : {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    y_pred = model.predict(X_test)

    print("\nRapport de classification (jeu de test) :")
    print(classification_report(y_test, y_pred))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))
    print("Labels :", list(model.classes_))

    feature_names = vectorizer.get_feature_names_out()
    selected_tokens = sorted(feature_names[np.any(model.coef_ != 0, axis=0)])
    print(f"\nTokens sélectionnés ({len(selected_tokens)}) : {selected_tokens}")

    print("\nTop 10 tokens les plus importants par pays :")
    for country, tokens in top_tokens_per_class(model, feature_names).items():
        formatted = ", ".join(f"{token!r} ({coef:.2f})" for token, coef in tokens)
        print(f"  {country}: {formatted}")


def main():
    cities_df = build_cities_dataframe()
    print(f"Nombre total de villes récupérées : {len(cities_df)}")
    print(cities_df["country"].value_counts())

    print(f"Chargement du vocabulaire du tokenizer de {SMOLLM3_MODEL_NAME} ...")
    vocabulary = load_smollm3_vocabulary()
    print(f"Taille du vocabulaire : {len(vocabulary)}")

    # 1) Sans les villes d'Australie
    df_without_australia = cities_df[cities_df["country"] != "Australia"]
    train_and_evaluate(df_without_australia, vocabulary, "Sans l'Australie")

    # 2) Avec les villes d'Australie
    train_and_evaluate(cities_df, vocabulary, "Avec l'Australie")


if __name__ == "__main__":
    main()
