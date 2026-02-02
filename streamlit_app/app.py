"""
streamlit_app/app.py — Interface Streamlit multi-collections pour le chatbot RAG.

Lance avec : streamlit run streamlit_app/app.py
"""

import sys
import tempfile
from pathlib import Path

# Ajouter la racine du projet au path pour les imports core.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from core.embeddings import verifier_ollama, OLLAMA_MODEL
from core.collection_manager import CollectionManager
from core.document_manager import DocumentManager
from core.search import RAGEngine
from core.parsers import extensions_supportees

# --- Configuration page ---
st.set_page_config(
    page_title="Assistant RAG Multi-Collections",
    page_icon="🤖",
    layout="centered",
)

st.markdown("""
<style>
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# --- Instances partagées ---
cm = CollectionManager()
dm = DocumentManager(cm)

# --- Vérification Ollama ---
if not verifier_ollama():
    st.error(
        "**Ollama n'est pas accessible.**\n\n"
        "Lancez-le dans un terminal avec :\n```bash\nollama serve\n```\n"
        "Puis rechargez cette page."
    )
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("Collections")

    # Lister les collections
    collections = cm.lister_collections()

    if collections:
        collection_active = st.selectbox(
            "Collection active",
            collections,
            key="collection_selectbox",
        )
    else:
        collection_active = None
        st.info("Aucune collection. Créez-en une ci-dessous.")

    # Créer une nouvelle collection
    st.divider()
    st.subheader("Nouvelle collection")
    nouveau_nom = st.text_input("Nom", key="new_collection_name", placeholder="ma_collection")
    if st.button("Créer", use_container_width=True, key="btn_create_collection"):
        if nouveau_nom and nouveau_nom.strip():
            nom = nouveau_nom.strip().replace(" ", "_").lower()
            if cm.collection_existe(nom):
                st.warning(f"La collection '{nom}' existe déjà.")
            else:
                cm.creer_collection(nom)
                st.success(f"Collection '{nom}' créée.")
                st.rerun()
        else:
            st.warning("Entrez un nom de collection.")

    # Upload de fichiers
    if collection_active:
        st.divider()
        st.subheader("Ajouter des documents")
        ext_list = extensions_supportees()
        ext_display = ", ".join(ext_list)
        fichiers_up = st.file_uploader(
            f"Formats : {ext_display}",
            accept_multiple_files=True,
            type=[e.lstrip(".") for e in ext_list],
            key="file_uploader",
        )
        if fichiers_up and st.button("Indexer", use_container_width=True, key="btn_index"):
            for fichier_up in fichiers_up:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(fichier_up.name).suffix,
                ) as tmp:
                    tmp.write(fichier_up.getbuffer())
                    tmp_path = Path(tmp.name)

                with st.spinner(f"Indexation de {fichier_up.name}..."):
                    resultat = dm.ajouter_document(collection_active, tmp_path)

                tmp_path.unlink(missing_ok=True)

                if resultat["status"] == "indexed":
                    st.success(resultat["message"])
                else:
                    st.info(resultat["message"])
            st.rerun()

        # Liste des documents indexés
        st.divider()
        st.subheader("Documents indexés")
        docs = dm.lister_documents(collection_active)
        if docs:
            for doc in docs:
                st.markdown(
                    f"- **{doc['nom']}** — {doc['nb_chunks']} chunks, "
                    f"{doc['nb_pages']} page(s)"
                )
        else:
            st.caption("Aucun document indexé.")

    # Effacer l'historique
    if collection_active:
        st.divider()
        cle_messages = f"messages_{collection_active}"
        if st.button("Effacer l'historique", use_container_width=True, key="btn_clear"):
            st.session_state[cle_messages] = []
            st.rerun()

    st.divider()
    st.caption(f"Modèle : `{OLLAMA_MODEL}`")
    st.caption("100% local — Aucune donnée envoyée à l'extérieur")

# --- Zone principale ---

st.title("🤖 Assistant RAG Multi-Collections")

if not collection_active:
    st.info("Créez ou sélectionnez une collection dans la barre latérale pour commencer.")
    st.stop()

st.caption(f"Collection active : **{collection_active}**")

# Vérifier que la collection contient des documents
if not dm.lister_documents(collection_active):
    st.warning(
        "Cette collection est vide. "
        "Ajoutez des documents via la barre latérale ou avec :\n\n"
        f"```bash\npython ingest.py {collection_active} ./documents/\n```"
    )

# --- Historique de conversation par collection ---
cle_messages = f"messages_{collection_active}"
if cle_messages not in st.session_state:
    st.session_state[cle_messages] = []

# Afficher les messages précédents
for msg in st.session_state[cle_messages]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources utilisées"):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **{src['fichier']}** — page {src['page']} "
                        f"*(score : {src['score']})*"
                    )

# --- Zone de saisie ---
question = st.chat_input("Posez votre question...")

if question:
    # Afficher la question
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state[cle_messages].append({"role": "user", "content": question})

    # Déterminer le prompt selon la collection
    prompt_name = "vlm_robotics" if collection_active == "vlm_robotics" else "defaut"

    try:
        engine = RAGEngine(collection_active, prompt_name=prompt_name, collection_manager=cm)
    except ValueError:
        st.error("Collection introuvable ou vide. Indexez des documents d'abord.")
        st.stop()

    # Recherche + génération streaming
    with st.spinner("Recherche dans la base de connaissances..."):
        try:
            resultat = engine.generer_avec_sources(question, stream=True)
        except Exception as e:
            st.error(f"Erreur lors de la recherche : {e}")
            st.stop()

    sources = resultat["sources"]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        texte_complet = ""

        for token in resultat["reponse"]:
            texte_complet += token
            placeholder.markdown(texte_complet + "▌")

        placeholder.markdown(texte_complet)

        if sources:
            with st.expander("Sources utilisées"):
                for src in sources:
                    st.markdown(
                        f"- **{src['fichier']}** — page {src['page']} "
                        f"*(score : {src['score']})*"
                    )

    # Sauvegarder dans l'historique
    st.session_state[cle_messages].append({
        "role": "assistant",
        "content": texte_complet,
        "sources": sources,
    })
