import re
from pathlib import Path
import base64

import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import KeyedVectors, Word2Vec

st.set_page_config(page_title="Word2Vec App By Safwen Gharbi", page_icon="C", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg: #0b1016;
  --panel: rgba(14, 20, 30, 0.76);
  --panel-strong: rgba(20, 28, 40, 0.92);
  --text: #e6edf3;
  --muted: #98a7b8;
  --accent: #2ce3b5;
  --accent-2: #fca311;
  --outline: rgba(90, 115, 140, 0.35);
}

html, body, [data-testid="stAppViewContainer"] {
  height: 100%;
  background: radial-gradient(800px 320px at 15% -10%, rgba(44, 227, 181, 0.15), transparent 60%),
              radial-gradient(640px 360px at 90% 10%, rgba(252, 163, 17, 0.15), transparent 60%),
              linear-gradient(180deg, #0b1016 0%, #0f1622 100%);
  color: var(--text);
  font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stHeader"] {
  background: rgba(0, 0, 0, 0);
}

.main .block-container {
  padding-top: 2.2rem;
  padding-bottom: 4rem;
  position: relative;
  z-index: 1;
}

.ambient {
  position: fixed;
  width: 460px;
  height: 460px;
  border-radius: 50%;
  filter: blur(90px);
  opacity: 0.35;
  z-index: 0;
  pointer-events: none;
}
.ambient.one { background: rgba(44, 227, 181, 0.35); top: -160px; left: -140px; }
.ambient.two { background: rgba(252, 163, 17, 0.28); bottom: -180px; right: -120px; }

.hero {
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  gap: 24px;
  align-items: center;
  padding: 24px;
  background: var(--panel);
  border: 1px solid var(--outline);
  border-radius: 20px;
  box-shadow: 0 12px 40px rgba(4, 7, 15, 0.45);
  backdrop-filter: blur(10px);
  animation: fadeUp 0.6s ease both;
}

.kicker {
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.18em;
  color: var(--accent);
  font-weight: 600;
}

.hero-title {
  font-size: 2.6rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 0.4rem 0 0.6rem;
}

.hero-sub {
  color: var(--muted);
  font-size: 1.05rem;
  line-height: 1.6;
}

.badges {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 14px;
}

.badge {
  background: rgba(44, 227, 181, 0.12);
  border: 1px solid rgba(44, 227, 181, 0.45);
  color: var(--accent);
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.stat-card {
  background: var(--panel-strong);
  border: 1px solid var(--outline);
  border-radius: 16px;
  padding: 14px 16px;
}

.stat-label {
  color: var(--muted);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.stat-value {
  font-size: 1.3rem;
  font-weight: 600;
  margin-top: 6px;
}

.case-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin-top: 18px;
}

.case-card {
  background: rgba(18, 26, 38, 0.85);
  border: 1px solid var(--outline);
  border-radius: 14px;
  padding: 12px 14px;
}

.case-title {
  font-weight: 600;
  margin-bottom: 4px;
}

.case-text {
  color: var(--muted);
  font-size: 0.9rem;
}

.notice {
  margin-top: 16px;
  padding: 10px 14px;
  border-radius: 12px;
  border: 1px solid rgba(44, 227, 181, 0.45);
  background: rgba(44, 227, 181, 0.1);
  color: var(--text);
  font-size: 0.95rem;
}

.credit-card {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  margin-top: 18px;
  padding: 14px 18px;
  background: var(--panel-strong);
  border: 1px solid var(--outline);
  border-radius: 16px;
}

.credit-logo {
  width: 72px;
  height: 72px;
  object-fit: contain;
  border-radius: 12px;
  background: rgba(10, 14, 22, 0.4);
  padding: 6px;
}

.credit-name {
  font-size: 1.05rem;
  font-weight: 600;
}

.credit-faculty {
  color: var(--muted);
  font-size: 0.95rem;
}

.flag {
  width: 22px;
  height: 16px;
  display: inline-block;
  vertical-align: -2px;
  margin-left: 6px;
}

[data-testid="stSidebar"] {
  background: rgba(10, 14, 22, 0.92);
  border-right: 1px solid var(--outline);
}

.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
.stSelectbox select {
  background: rgba(9, 13, 20, 0.7);
  border: 1px solid var(--outline);
  color: var(--text);
  border-radius: 10px;
}

.stButton button {
  background: linear-gradient(135deg, #2ce3b5, #1fa2ff);
  color: #081017;
  border: none;
  border-radius: 12px;
  padding: 0.6rem 1.2rem;
  font-weight: 600;
}

.stButton button:hover {
  filter: brightness(1.06);
}

.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  margin-top: 12px;
}

.stTabs [data-baseweb="tab"] {
  background: rgba(16, 22, 32, 0.7);
  border: 1px solid var(--outline);
  border-radius: 999px;
  padding: 6px 14px;
  color: var(--muted);
}

.stTabs [aria-selected="true"] {
  background: rgba(44, 227, 181, 0.2);
  color: var(--text);
  border-color: rgba(44, 227, 181, 0.6);
}

code, pre {
  font-family: 'JetBrains Mono', monospace;
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 900px) {
  .hero { grid-template-columns: 1fr; }
  .hero-title { font-size: 2.1rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="ambient one"></div><div class="ambient two"></div>', unsafe_allow_html=True)

MODEL_PATH = Path(__file__).with_name("word2vec.model")
LOGO_CANDIDATES = [
    Path(__file__).with_name("fsb.png"),
    Path(__file__).with_name("fsb logo.png"),
]
CREATOR_NAME = "Safwen Gharbi"
FACULTY_NAME = "Faculté des Sciences de Bizerte"
TUNISIA_FLAG_SVG = """
<svg class="flag" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 20" aria-label="Drapeau de la Tunisie" role="img">
  <rect width="28" height="20" fill="#E70013"/>
  <circle cx="14" cy="10" r="6" fill="#FFFFFF"/>
  <circle cx="12.8" cy="10" r="4.2" fill="#E70013"/>
  <circle cx="14.2" cy="10" r="3.3" fill="#FFFFFF"/>
  <polygon points="16.00,7.80 16.53,9.27 18.09,9.32 16.86,10.28 17.29,11.78 16.00,10.90 14.71,11.78 15.14,10.28 13.91,9.32 15.47,9.27" fill="#E70013"/>
</svg>
"""
SAMPLE_TERMS = [
    "malware",
    "phishing",
    "ransomware",
    "botnet",
    "exploit",
    "vulnerability",
    "patch",
    "firewall",
    "zero-day",
    "forensics",
    "intrusion",
    "payload",
    "endpoint",
    "credential",
    "encryption",
    "dns",
    "c2",
    "backdoor",
    "trojan",
    "sandbox",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+\-./]{1,}")


@st.cache_resource(show_spinner=False)
def load_vectors(path: Path) -> KeyedVectors:
    last_error = None
    try:
        model = Word2Vec.load(str(path))
        return model.wv
    except Exception as exc:
        last_error = exc
    try:
        return KeyedVectors.load(str(path), mmap="r")
    except Exception:
        pass
    try:
        return KeyedVectors.load_word2vec_format(str(path), binary=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load model from {path}. Error: {last_error or exc}")


def normalize_token(token: str, lower: bool) -> str:
    token = token.strip()
    return token.lower() if lower else token


def tokenize(text: str, lower: bool = True) -> list[str]:
    if not text:
        return []
    tokens = TOKEN_RE.findall(text)
    return [t.lower() for t in tokens] if lower else tokens


def in_vocab(kv: KeyedVectors, token: str) -> bool:
    return token in kv.key_to_index


def filter_known(kv: KeyedVectors, tokens: list[str]) -> tuple[list[str], list[str]]:
    known = []
    missing = []
    for token in tokens:
        if in_vocab(kv, token):
            known.append(token)
        else:
            missing.append(token)
    return known, missing


def mean_vector(kv: KeyedVectors, tokens: list[str]) -> tuple[np.ndarray | None, list[str]]:
    known, _ = filter_known(kv, tokens)
    if not known:
        return None, []
    vectors = [kv.get_vector(token) for token in known]
    return np.mean(vectors, axis=0), known


def cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def parse_list(text: str, lower: bool = True) -> list[str]:
    raw = re.split(r"[,\n]+", text or "")
    return [normalize_token(item, lower) for item in raw if item.strip()]


def odd_one_out(kv: KeyedVectors, terms: list[str]) -> tuple[str | None, list[str]]:
    known, _ = filter_known(kv, terms)
    if len(known) < 2:
        return None, known
    vectors = np.vstack([kv.get_vector(term) for term in known])
    centroid = np.mean(vectors, axis=0)
    scored = [(term, cosine(kv.get_vector(term), centroid)) for term in known]
    scored.sort(key=lambda item: item[1])
    return scored[0][0], known


def prefix_search(kv: KeyedVectors, prefix: str, limit: int = 50, lower: bool = True) -> list[str]:
    prefix = normalize_token(prefix, lower)
    if not prefix:
        return []
    matches = []
    for word in kv.key_to_index.keys():
        test_word = word.lower() if lower else word
        if test_word.startswith(prefix):
            matches.append(word)
            if len(matches) >= limit:
                break
    return matches


if not MODEL_PATH.exists():
    st.error(f"Fichier modèle introuvable : {MODEL_PATH}")
    st.stop()

with st.spinner("Chargement du modèle Word2Vec..."):
    vectors = load_vectors(MODEL_PATH)

with st.sidebar:
    st.markdown("## Contrôles")
    st.caption("Entrées en anglais uniquement (modèle entraîné sur des articles anglais).")
    top_n = st.slider("Top N résultats", 3, 50, 12)
    lowercase = st.checkbox("Passer en minuscules", value=True)
    st.markdown("### Recherche par préfixe du vocabulaire")
    prefix = st.text_input("Préfixe (anglais)", "mal")
    limit = st.slider("Limite de résultats", 10, 100, 30)
    if prefix:
        matches = prefix_search(vectors, prefix, limit, lowercase)
        if matches:
            st.write(", ".join(matches))
        else:
            st.caption("Aucun résultat.")
    st.markdown("### Termes d'exemple")
    st.caption(", ".join(SAMPLE_TERMS))

vocab_size = len(vectors)
vector_size = vectors.vector_size
vector_type = vectors.__class__.__name__

logo_path = next((path for path in LOGO_CANDIDATES if path.exists()), None)
logo_html = ""
if logo_path:
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    logo_html = (
        f'<img class="credit-logo" src="data:image/png;base64,{logo_b64}" '
        f'alt="Logo FSB" />'
    )

hero_html = f"""
<div class="hero">
  <div>
    <div class="kicker">CYBERSÉCURITÉ WORD2VEC</div>
    <div class="hero-title">Laboratoire CyberWord2Vec</div>
    <div class="hero-sub">Laboratoire mono-page pour explorer la sémantique du vocabulaire cybersécurité. Le modèle est entraîné sur des articles en anglais : utilisez des entrées en anglais pour de meilleurs résultats.</div>
    <div class="badges">
      <span class="badge">voisins</span>
      <span class="badge">analogie</span>
    </div>
  </div>
  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-label">Taille du vocabulaire</div>
      <div class="stat-value">{vocab_size:,}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Dimension des vecteurs</div>
      <div class="stat-value">{vector_size}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Type de modèle</div>
      <div class="stat-value">{vector_type}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Fichier modèle</div>
      <div class="stat-value">word2vec.model</div>
    </div>
  </div>
</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)

st.markdown(
    '<div class="notice">Entrées en anglais uniquement - le modèle est entraîné sur des articles anglais.</div>',
    unsafe_allow_html=True,
)

credit_html = f"""
<div class="credit-card">
  {logo_html}
  <div class="credit-info">
    <div class="credit-name">Créé par {CREATOR_NAME} {TUNISIA_FLAG_SVG}</div>
    <div class="credit-faculty">{FACULTY_NAME}</div>
  </div>
</div>
"""

st.markdown(credit_html, unsafe_allow_html=True)

cases_html = """
<div class="case-grid">
  <div class="case-card">
    <div class="case-title">Voisins proches</div>
    <div class="case-text">Trouver les termes qui vivent dans le même voisinage sémantique.</div>
  </div>
  <div class="case-card">
    <div class="case-title">Solveur d'analogies</div>
    <div class="case-text">Résoudre les requêtes de type A est à B comme C est à D.</div>
  </div>
</div>
"""

st.markdown(cases_html, unsafe_allow_html=True)


def show_results(results: list[tuple[str, float]]) -> None:
    if not results:
        st.info("Aucun résultat.")
        return
    df = pd.DataFrame(results, columns=["term", "score"])
    st.dataframe(df, use_container_width=True)


tabs = st.tabs(
    [
        "Voisins",
        "Similarité",
        "Analogie",
        "Intrus",
        "Similarité de textes",
    ]
)

with tabs[0]:
    st.subheader("Voisins les plus proches")
    st.write("Obtenir les termes les plus proches pour un mot de départ.")
    with st.form("neighbors_form"):
        word = st.text_input("Terme de départ (anglais)", value="malware")
        submitted = st.form_submit_button("Trouver les voisins")
    if submitted:
        token = normalize_token(word, lowercase)
        if not token:
            st.warning("Saisissez un terme.")
        elif not in_vocab(vectors, token):
            st.error(f"'{token}' n'est pas dans le vocabulaire.")
        else:
            if token != word.strip():
                st.caption(f"Terme normalisé : {token}")
            show_results(vectors.most_similar(token, topn=top_n))

with tabs[1]:
    st.subheader("Similarité de mots")
    st.write("Comparer deux termes avec la similarité cosinus.")
    with st.form("similarity_form"):
        col1, col2 = st.columns(2)
        with col1:
            term_a = st.text_input("Terme A (anglais)", value="phishing")
        with col2:
            term_b = st.text_input("Terme B (anglais)", value="ransomware")
        submitted = st.form_submit_button("Comparer")
    if submitted:
        term_a = normalize_token(term_a, lowercase)
        term_b = normalize_token(term_b, lowercase)
        missing = [t for t in (term_a, term_b) if not in_vocab(vectors, t)]
        if missing:
            st.error(f"Manquants du vocabulaire : {', '.join(missing)}")
        else:
            score = float(vectors.similarity(term_a, term_b))
            st.metric("Similarité cosinus", f"{score:.3f}")

with tabs[2]:
    st.subheader("Solveur d'analogies")
    st.write("Résoudre A est à B comme C est à ?")
    with st.form("analogy_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            term_a = st.text_input("A (anglais)", value="malware")
        with col2:
            term_b = st.text_input("B (anglais)", value="ransomware")
        with col3:
            term_c = st.text_input("C (anglais)", value="phishing")
        submitted = st.form_submit_button("Résoudre l'analogie")
    if submitted:
        term_a = normalize_token(term_a, lowercase)
        term_b = normalize_token(term_b, lowercase)
        term_c = normalize_token(term_c, lowercase)
        missing = [t for t in (term_a, term_b, term_c) if not in_vocab(vectors, t)]
        if missing:
            st.error(f"Manquants du vocabulaire : {', '.join(missing)}")
        else:
            results = vectors.most_similar(
                positive=[term_b, term_c], negative=[term_a], topn=top_n
            )
            show_results(results)

with tabs[3]:
    st.subheader("Intrus")
    st.write("Détecter le terme qui ne correspond pas au groupe.")
    with st.form("outlier_form"):
        terms_text = st.text_area(
            "Termes (anglais, séparés par virgules ou retours à la ligne)",
            value="phishing, ransomware, malware, firewall",
        )
        submitted = st.form_submit_button("Trouver l'intrus")
    if submitted:
        terms = parse_list(terms_text, lowercase)
        if len(terms) < 3:
            st.warning("Saisissez au moins trois termes.")
        else:
            outlier, known = odd_one_out(vectors, terms)
            missing = [term for term in terms if term not in known]
            if missing:
                st.info(f"Termes manquants ignorés : {', '.join(missing)}")
            if outlier:
                st.success(f"Intrus : {outlier}")
                centroid = np.mean([vectors.get_vector(t) for t in known], axis=0)
                scored = [(t, cosine(vectors.get_vector(t), centroid)) for t in known]
                scored.sort(key=lambda item: item[1])
                st.dataframe(
                    pd.DataFrame(scored, columns=["term", "similarity_to_centroid"]),
                    use_container_width=True,
                )
            else:
                st.error("Pas assez de termes connus pour calculer un intrus.")

with tabs[4]:
    st.subheader("Similarité de textes")
    st.write("Comparer deux textes courts via la moyenne des embeddings.")
    with st.form("textsim_form"):
        col1, col2 = st.columns(2)
        with col1:
            text_a = st.text_area(
                "Texte A (anglais)",
                value="phishing email with a malicious link",
            )
        with col2:
            text_b = st.text_area(
                "Texte B (anglais)",
                value="ransomware delivered via email attachment",
            )
        submitted = st.form_submit_button("Comparer les textes")
    if submitted:
        tokens_a = tokenize(text_a, lowercase)
        tokens_b = tokenize(text_b, lowercase)
        vec_a, used_a = mean_vector(vectors, tokens_a)
        vec_b, used_b = mean_vector(vectors, tokens_b)
        if vec_a is None or vec_b is None:
            st.error("Chaque texte doit contenir au moins un token connu.")
        else:
            score = cosine(vec_a, vec_b)
            st.metric("Similarité cosinus", f"{score:.3f}")
            st.caption(f"Tokens utilisés A : {', '.join(used_a)}")
            st.caption(f"Tokens utilisés B : {', '.join(used_b)}")

st.caption(
    "Astuce : si un terme manque, essayez une variante proche ou vérifiez la recherche par préfixe dans la barre latérale."
)
