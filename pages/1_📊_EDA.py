"""
Page EDA - Exploration des Données
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import random

st.set_page_config(
    page_title="EDA - Classification Produits",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploration des Données (EDA)")
st.markdown("Analyse du dataset Flipkart - 1050 images de produits e-commerce")

# Chemins
BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "inputs" / "Flipkart" / "images"
CSV_PATH = BASE_DIR / "inputs" / "Flipkart" / "flipkart_com-ecommerce_sample_1050.csv"

# Chargement des données
@st.cache_data
def load_data():
    """Charge et prépare le dataset."""
    df = pd.read_csv(CSV_PATH, index_col=0, encoding="ISO-8859-1")
    df["category"] = df["product_category_tree"].apply(
        lambda x: x.split('>>')[0].replace('["', "").strip()
    )
    df["image_path"] = df["image"].apply(lambda x: str(IMAGES_DIR / x))
    return df

try:
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    data_loaded = False

if data_loaded:
    st.markdown("---")

    # Section 1 : Distribution des catégories
    st.markdown("## 1. Distribution des Catégories")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Graphique interactif avec Plotly
        category_counts = df["category"].value_counts().reset_index()
        category_counts.columns = ["Catégorie", "Nombre d'images"]

        fig = px.bar(
            category_counts,
            x="Catégorie",
            y="Nombre d'images",
            color="Catégorie",
            title="Distribution des images par catégorie",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            height=400,
            font=dict(size=12)
        )
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Images: %{y}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Description textuelle du graphique pour WCAG 1.1.1
        st.markdown("""
        <details>
        <summary><strong>Description du graphique</strong> (accessibilité)</summary>
        <p>Ce graphique en barres montre la distribution des 1050 images réparties équitablement
        entre 7 catégories. Chaque catégorie contient exactement 150 images, illustrant
        un dataset parfaitement équilibré.</p>
        </details>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Statistiques")
        st.markdown(f"""
        - **Total images** : {len(df)}
        - **Catégories** : {df['category'].nunique()}
        - **Images/catégorie** : {len(df) // df['category'].nunique()}

        Le dataset est **parfaitement équilibré** avec 150 images par catégorie.
        """)

        # Tableau accessible
        st.dataframe(
            category_counts,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Section 2 : Exemples d'images par catégorie
    st.markdown("## 2. Exemples d'Images par Catégorie")

    selected_category = st.selectbox(
        "Sélectionnez une catégorie :",
        options=sorted(df["category"].unique()),
        index=0,
        help="Choisissez une catégorie pour voir des exemples d'images"
    )

    # Nombre d'images à afficher
    n_images = st.slider(
        "Nombre d'images à afficher :",
        min_value=3,
        max_value=9,
        value=6,
        step=3,
        help="Ajustez le nombre d'exemples affichés"
    )

    # Filtrer et afficher
    category_df = df[df["category"] == selected_category]
    sample_images = category_df.sample(min(n_images, len(category_df)), random_state=42)

    cols = st.columns(3)
    for idx, (_, row) in enumerate(sample_images.iterrows()):
        with cols[idx % 3]:
            try:
                img_path = row["image_path"]
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    # Alt text descriptif pour accessibilité WCAG 1.1.1
                    alt_text = f"Image de produit de la catégorie {selected_category}"
                    st.image(
                        img,
                        caption=f"Exemple de produit : {selected_category}",
                        use_container_width=True
                    )
                    # Description textuelle alternative
                    st.markdown(f"<span class='sr-only'>{alt_text}</span>", unsafe_allow_html=True)
                else:
                    st.warning(f"Image non trouvée")
            except Exception as e:
                st.error(f"Erreur : {e}")

    st.markdown("---")

    # Section 3 : Dimensions des images
    st.markdown("## 3. Analyse des Dimensions")

    @st.cache_data
    def analyze_dimensions(sample_size=100):
        """Analyse les dimensions d'un échantillon d'images."""
        dims = []
        sample = df.sample(min(sample_size, len(df)), random_state=42)
        for _, row in sample.iterrows():
            try:
                if os.path.exists(row["image_path"]):
                    img = Image.open(row["image_path"])
                    dims.append({
                        "width": img.size[0],
                        "height": img.size[1],
                        "ratio": img.size[0] / img.size[1]
                    })
            except:
                pass
        return pd.DataFrame(dims)

    dims_df = analyze_dimensions()

    if not dims_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(
                dims_df,
                x="width",
                y="height",
                title="Distribution des dimensions (échantillon)",
                labels={"width": "Largeur (px)", "height": "Hauteur (px)"},
                opacity=0.6
            )
            fig.add_shape(
                type="rect",
                x0=224, y0=224, x1=224, y1=224,
                line=dict(color="red", width=2, dash="dash")
            )
            fig.add_annotation(
                x=224, y=224,
                text="Taille cible (224×224)",
                showarrow=True,
                arrowhead=2
            )
            st.plotly_chart(fig, use_container_width=True)

            # Description textuelle pour accessibilité
            st.markdown("""
            <details>
            <summary><strong>Description du graphique</strong> (accessibilité)</summary>
            <p>Ce nuage de points montre la variété des dimensions originales des images
            (largeur vs hauteur en pixels). Un marqueur rouge indique la taille cible de 224×224
            pixels vers laquelle toutes les images sont redimensionnées pour l'entraînement.</p>
            </details>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Statistiques des dimensions")
            st.markdown(f"""
            | Métrique | Largeur | Hauteur |
            |----------|---------|---------|
            | **Min** | {dims_df['width'].min()} px | {dims_df['height'].min()} px |
            | **Max** | {dims_df['width'].max()} px | {dims_df['height'].max()} px |
            | **Moyenne** | {dims_df['width'].mean():.0f} px | {dims_df['height'].mean():.0f} px |

            **Note** : Toutes les images sont redimensionnées en **224×224** pour
            l'entraînement des modèles VGG16 et ConvNeXt-Tiny.
            """)

    st.markdown("---")

    # Section 4 : Augmentation de données
    st.markdown("## 4. Pipeline d'Augmentation")

    st.markdown("""
    L'augmentation de données est utilisée pour améliorer la généralisation du modèle.
    Voici les transformations appliquées pendant l'entraînement :
    """)

    aug_data = {
        "Transformation": [
            "GaussNoise",
            "RandomBrightnessContrast",
            "Blur",
            "Affine (scale, rotate, translate)",
            "HorizontalFlip",
            "VerticalFlip"
        ],
        "Probabilité": ["10%", "10%", "20%", "30%", "30%", "10%"],
        "Description": [
            "Ajout de bruit gaussien léger",
            "Variation de luminosité/contraste (±10%)",
            "Flou léger (kernel 3)",
            "Échelle ±5%, rotation ±10°, translation ±5%",
            "Retournement horizontal",
            "Retournement vertical"
        ]
    }

    st.dataframe(
        pd.DataFrame(aug_data),
        use_container_width=True,
        hide_index=True
    )

    # Démonstration visuelle
    st.markdown("### Démonstration")

    demo_category = st.selectbox(
        "Catégorie pour la démo :",
        options=sorted(df["category"].unique()),
        key="demo_category"
    )

    demo_df = df[df["category"] == demo_category].sample(1, random_state=42)
    demo_path = demo_df.iloc[0]["image_path"]

    if os.path.exists(demo_path):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original**")
            img = Image.open(demo_path)
            st.image(img, caption=f"Image originale - {demo_category}", use_container_width=True)

        with col2:
            st.markdown("**Flip Horizontal**")
            img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            st.image(img_flip, caption="Transformation : retournement horizontal", use_container_width=True)

        with col3:
            st.markdown("**Rotation + Flip**")
            img_rot = img.rotate(10).transpose(Image.FLIP_LEFT_RIGHT)
            st.image(img_rot, caption="Transformation : rotation 10° + flip", use_container_width=True)

        # Description textuelle pour WCAG 1.4.1 (ne pas utiliser que la couleur)
        st.markdown("""
        **Description des transformations** : Les trois images ci-dessus montrent l'image originale
        (gauche), puis la même image retournée horizontalement (centre), et enfin pivotée de 10 degrés
        avec retournement (droite). Ces transformations permettent d'augmenter artificiellement le dataset.
        """)

    st.markdown("---")

    # Section 5 : Résumé
    st.markdown("## 5. Résumé du Dataset")

    st.success("""
    **Points clés :**
    - Dataset équilibré (150 images × 7 catégories = 1050 images)
    - Images de tailles variées, normalisées en 224×224
    - Augmentation modérée pour éviter l'overfitting
    - Split stratifié : 64% train / 20% val / 16% test
    """)

else:
    st.info("Veuillez vérifier que le dataset est présent dans le dossier `inputs/Flipkart/`")