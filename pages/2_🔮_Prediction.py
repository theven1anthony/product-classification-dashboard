"""
Page Prédiction - Classification d'images avec Grad-CAM
"""

import streamlit as st
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image
import pickle

st.set_page_config(
    page_title="Prédiction - Classification Produits",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Prédiction de Catégorie")
st.markdown("Uploadez une image de produit pour obtenir une prédiction avec visualisation Grad-CAM")

# Chemins
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "convnext_model_dashboard.keras"
ENCODER_PATH = BASE_DIR / "outputs" / "label_encoder.pkl"
CONFIG_PATH = BASE_DIR / "outputs" / "model_config.json"
IMAGES_DIR = BASE_DIR / "inputs" / "Flipkart" / "images"

# Chargement du modèle et configuration
@st.cache_resource
def load_model():
    """Charge le modèle ConvNeXt."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(MODEL_PATH))
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return None

@st.cache_data
def load_config():
    """Charge la configuration."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
        return config, encoder
    except Exception as e:
        st.error(f"Erreur chargement config : {e}")
        return None, None

def preprocess_image(img, target_size=(224, 224)):
    """Prétraite l'image pour le modèle."""
    from tensorflow.keras.applications.convnext import preprocess_input

    # Redimensionner
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    # Convertir en array
    img_array = np.array(img)

    # S'assurer que c'est RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Ajouter dimension batch et prétraiter
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    img_array = preprocess_input(img_array)

    return img_array

def get_gradcam(model, img_array, pred_index):
    """Génère la heatmap Grad-CAM."""
    import tensorflow as tf

    # Trouver la dernière couche conv
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower() or 'block' in layer.name.lower():
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        return None

    # Créer modèle Grad-CAM
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    # Calculer gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Créer heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def apply_heatmap(img, heatmap):
    """Applique la heatmap sur l'image."""
    import cv2

    # Redimensionner heatmap
    heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))

    # Convertir en colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superposer
    img_array = np.array(img)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    superimposed = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)

    return Image.fromarray(heatmap_colored), Image.fromarray(superimposed)

# Interface principale
st.markdown("---")

# Charger modèle et config
model = load_model()
config, encoder = load_config()

if model is not None and config is not None:
    st.success(f"Modèle **{config['model_name']}** chargé (Accuracy: {config['accuracy']*100:.2f}%)")

    # Tabs pour les deux modes
    tab1, tab2 = st.tabs(["📤 Upload d'image", "📁 Sélection depuis le dataset"])

    with tab1:
        st.markdown("### Uploadez votre image")
        uploaded_file = st.file_uploader(
            "Choisissez une image de produit",
            type=["jpg", "jpeg", "png", "webp"],
            help="Formats acceptés : JPG, PNG, WebP"
        )

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('RGB')
            source_img = img
            run_prediction = True
        else:
            run_prediction = False

    with tab2:
        st.markdown("### Sélectionnez une image du dataset")

        # Liste des catégories
        categories = config.get('class_names', [])
        selected_cat = st.selectbox(
            "Catégorie :",
            options=categories,
            help="Choisissez une catégorie pour voir les images disponibles"
        )

        # Lister les images de cette catégorie
        import pandas as pd
        CSV_PATH = BASE_DIR / "inputs" / "Flipkart" / "flipkart_com-ecommerce_sample_1050.csv"

        try:
            df = pd.read_csv(CSV_PATH, index_col=0, encoding="ISO-8859-1")
            df["category"] = df["product_category_tree"].apply(
                lambda x: x.split('>>')[0].replace('["', "").strip()
            )

            # Filtrer pour ne garder que les images présentes dans le dossier
            cat_images = df[df["category"] == selected_cat]["image"].tolist()
            cat_images = [img for img in cat_images if (IMAGES_DIR / img).exists()]

            selected_img = st.selectbox(
                "Image :",
                options=cat_images,
                help="Sélectionnez une image"
            )

            if st.button("Prédire cette image", type="primary"):
                img_path = IMAGES_DIR / selected_img
                if img_path.exists():
                    img = Image.open(img_path).convert('RGB')
                    source_img = img
                    run_prediction = True
                else:
                    st.error("Image non trouvée")
                    run_prediction = False
            else:
                run_prediction = False

        except Exception as e:
            st.error(f"Erreur : {e}")
            run_prediction = False

    # Exécuter la prédiction
    if 'run_prediction' in dir() and run_prediction and 'source_img' in dir():
        st.markdown("---")
        st.markdown("## Résultats de la Prédiction")

        with st.spinner("Analyse en cours..."):
            # Prétraitement
            img_processed = preprocess_image(source_img)

            # Prédiction
            predictions = model.predict(img_processed, verbose=0)
            pred_class_idx = np.argmax(predictions[0])
            pred_confidence = predictions[0][pred_class_idx]
            pred_class_name = config['class_names'][pred_class_idx]

            # Grad-CAM
            try:
                heatmap = get_gradcam(model, img_processed, pred_class_idx)
                heatmap_img, superimposed_img = apply_heatmap(source_img, heatmap)
                gradcam_success = True
            except Exception as e:
                st.warning(f"Grad-CAM non disponible : {e}")
                gradcam_success = False

        # Affichage des résultats
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Image Originale")
            st.image(source_img, caption=f"Image uploadée pour classification", use_container_width=True)

        with col2:
            if gradcam_success:
                st.markdown("### Heatmap Grad-CAM")
                st.image(heatmap_img, caption="Carte de chaleur montrant les zones d'attention du modèle (rouge=haute importance, bleu=faible)", use_container_width=True)

        with col3:
            if gradcam_success:
                st.markdown("### Superposition")
                st.image(superimposed_img, caption="Image originale avec zones d'attention superposées", use_container_width=True)

        # Description textuelle des images pour accessibilité WCAG 1.1.1
        if gradcam_success:
            st.markdown(f"""
            <details>
            <summary><strong>Description des visualisations</strong> (accessibilité)</summary>
            <p><strong>Image originale</strong> : L'image du produit soumise pour classification.</p>
            <p><strong>Heatmap Grad-CAM</strong> : Carte de chaleur où les couleurs chaudes (rouge, jaune)
            indiquent les zones que le modèle considère importantes pour sa décision, et les couleurs froides
            (bleu, vert) les zones moins pertinentes.</p>
            <p><strong>Superposition</strong> : Combinaison de l'image originale et de la heatmap permettant
            de voir directement quelles parties du produit ont influencé la prédiction "{pred_class_name}".</p>
            </details>
            """, unsafe_allow_html=True)

        # Prédiction principale
        st.markdown("---")
        st.markdown("### Prédiction")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(
                label="Catégorie prédite",
                value=pred_class_name,
                help="Catégorie avec la plus haute probabilité"
            )
            st.metric(
                label="Confiance",
                value=f"{pred_confidence*100:.1f}%",
                help="Probabilité de la prédiction"
            )

        with col2:
            st.markdown("#### Distribution des probabilités")

            # Créer DataFrame pour affichage
            proba_df = pd.DataFrame({
                "Catégorie": config['class_names'],
                "Probabilité": predictions[0]
            }).sort_values("Probabilité", ascending=False)

            # Bar chart
            import plotly.express as px
            fig = px.bar(
                proba_df,
                x="Probabilité",
                y="Catégorie",
                orientation='h',
                color="Probabilité",
                color_continuous_scale="Blues"
            )
            fig.update_layout(
                height=300,
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Description textuelle du graphique pour accessibilité
            st.markdown(f"""
            <details>
            <summary><strong>Description du graphique</strong> (accessibilité)</summary>
            <p>Graphique en barres horizontales montrant la probabilité attribuée par le modèle
            à chaque catégorie. La catégorie "{pred_class_name}" a la probabilité la plus élevée
            ({pred_confidence*100:.1f}%), ce qui détermine la prédiction finale.</p>
            </details>
            """, unsafe_allow_html=True)

        # Interprétation Grad-CAM
        if gradcam_success:
            st.markdown("---")
            st.markdown("### Interprétation Grad-CAM")
            st.info("""
            **Grad-CAM** (Gradient-weighted Class Activation Mapping) visualise les zones de l'image
            qui ont le plus contribué à la prédiction du modèle.

            - **Rouge/Jaune** : Zones fortement activées (importantes pour la décision)
            - **Bleu/Vert** : Zones faiblement activées (peu d'influence)

            Cette technique permet d'expliquer les décisions du modèle et de vérifier
            qu'il se concentre sur les bonnes caractéristiques du produit.
            """)

else:
    st.warning("""
    **Modèle non disponible**

    Pour utiliser cette page, assurez-vous que les fichiers suivants sont présents :
    - `outputs/convnext_model_dashboard.keras`
    - `outputs/label_encoder.pkl`
    - `outputs/model_config.json`

    Ces fichiers sont générés par le notebook `notebook_comparatif_vgg16_convnext.ipynb`.
    """)

# Footer accessibilité
st.markdown("---")
st.caption("""
**Accessibilité** : Les images incluent des descriptions alternatives.
Les résultats sont présentés sous forme de texte et de graphiques pour une meilleure compréhension.
""")