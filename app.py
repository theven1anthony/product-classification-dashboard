"""
Dashboard Streamlit - Classification de Produits E-commerce
Comparaison VGG16 vs ConvNeXt-Tiny
"""

import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Classification Produits E-commerce",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour accessibilité WCAG
st.markdown("""
<style>
    /* Contraste amélioré WCAG AA */
    .stMarkdown p, .stMarkdown li {
        color: #1a1a1a;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Focus visible pour navigation clavier */
    *:focus {
        outline: 3px solid #0066cc !important;
        outline-offset: 2px;
    }

    /* Titres accessibles */
    h1, h2, h3 {
        color: #0d1117;
    }

    /* Liens avec contraste suffisant */
    a {
        color: #0066cc;
        text-decoration: underline;
    }

    /* Boutons avec contraste */
    .stButton > button {
        background-color: #0066cc;
        color: white;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #004d99;
    }

    /* Cards avec bordures visibles */
    .metric-card {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Classe pour lecteurs d'écran uniquement (WCAG 1.1.1) */
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    }

    /* Style pour les détails/descriptions accessibles */
    details {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    details summary {
        cursor: pointer;
        color: #0066cc;
    }

    details p, details li {
        color: #333;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-simple-bot.svg/128px-24701-simple-bot.svg.png", width=80)
    st.title("Navigation")
    st.markdown("---")
    st.markdown("""
    ### Pages disponibles
    - **Accueil** : Présentation du projet
    - **EDA** : Exploration des données
    - **Prédiction** : Classifier une image
    - **Résultats** : Comparaison des modèles
    """)
    st.markdown("---")
    st.caption("Projet OpenClassrooms - Ingénieur IA")
    st.caption("Classification automatique de produits")

# Page d'accueil
st.title("🛒 Classification de Produits E-commerce")
st.markdown("### Comparaison VGG16 (2014) vs ConvNeXt-Tiny (2022)")

st.markdown("---")

# Introduction
st.markdown("""
## Objectif du Projet

Ce dashboard présente les résultats d'une **veille technologique** comparant deux architectures
de deep learning pour la classification automatique de produits e-commerce :

| Modèle | Année | Rôle |
|--------|-------|------|
| **VGG16** | 2014 | Baseline (référence) |
| **ConvNeXt-Tiny** | 2022 | Nouveau modèle (< 5 ans) |
""")

# Métriques principales
st.markdown("## Résultats Clés")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Accuracy ConvNeXt",
        value="85.71%",
        delta="+2.38 pts vs VGG16",
        help="Précision sur le jeu de test (168 images)"
    )

with col2:
    st.metric(
        label="Accuracy VGG16",
        value="83.33%",
        delta=None,
        help="Baseline - Précision sur le jeu de test"
    )

with col3:
    st.metric(
        label="Catégories",
        value="7",
        help="Nombre de classes de produits"
    )

st.markdown("---")

# Dataset
st.markdown("## Dataset Flipkart")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Caractéristiques
    - **1050 images** de produits
    - **7 catégories** équilibrées (150 images chacune)
    - Images RGB redimensionnées en 224×224
    - Split : 64% train / 20% val / 16% test
    """)

with col2:
    st.markdown("""
    ### Catégories
    1. Baby Care
    2. Beauty and Personal Care
    3. Computers
    4. Home Decor & Festive Needs
    5. Home Furnishing
    6. Kitchen & Dining
    7. Watches
    """)

st.markdown("---")

# Navigation
st.markdown("## Explorer le Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 EDA
    Explorez la distribution des données,
    visualisez des exemples d'images par catégorie
    et les transformations d'augmentation.
    """)
    st.page_link("pages/1_📊_EDA.py", label="Aller à l'EDA", icon="📊")

with col2:
    st.markdown("""
    ### 🔮 Prédiction
    Uploadez une image de produit et obtenez
    une prédiction avec visualisation Grad-CAM
    des zones d'attention du modèle.
    """)
    st.page_link("pages/2_🔮_Prediction.py", label="Faire une prédiction", icon="🔮")

with col3:
    st.markdown("""
    ### 📈 Résultats
    Comparez les performances de VGG16 et
    ConvNeXt-Tiny : accuracy, loss, matrices
    de confusion et analyse détaillée.
    """)
    st.page_link("pages/3_📈_Resultats.py", label="Voir les résultats", icon="📈")

# Section Accessibilité
st.markdown("---")
st.markdown("## Accessibilité")

with st.expander("Conformité WCAG - Critères d'accessibilité respectés"):
    st.markdown("""
    Ce dashboard respecte les critères d'accessibilité WCAG suivants :

    | Critère | Description | Implémentation |
    |---------|-------------|----------------|
    | **1.1.1** | Contenu non textuel | Descriptions alternatives pour toutes les images et graphiques |
    | **1.4.1** | Utilisation de la couleur | Informations transmises par texte en plus des couleurs |
    | **1.4.3** | Contraste minimum | Ratio de contraste ≥ 4.5:1 pour le texte |
    | **1.4.4** | Redimensionnement texte | Layout responsive, tailles en rem |
    | **2.4.2** | Titre de page | Chaque page a un titre descriptif unique |

    **Fonctionnalités d'accessibilité :**
    - Navigation au clavier avec focus visible
    - Descriptions textuelles des graphiques (cliquez sur "Description du graphique")
    - Tableaux de données en complément des visualisations
    - Captions sur toutes les images
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Développé dans le cadre de la formation Ingénieur IA - OpenClassrooms</p>
    <p>Modèle ConvNeXt-Tiny basé sur l'article :
    <a href="https://arxiv.org/abs/2201.03545" target="_blank" rel="noopener"
       aria-label="Lien vers l'article ConvNeXt sur ArXiv (s'ouvre dans un nouvel onglet)">
       A ConvNet for the 2020s (Liu et al., 2022)
    </a></p>
</div>
""", unsafe_allow_html=True)