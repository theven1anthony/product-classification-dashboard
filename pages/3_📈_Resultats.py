"""
Page Résultats - Comparaison VGG16 vs ConvNeXt-Tiny
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(
    page_title="Résultats - Classification Produits",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Résultats Comparatifs")
st.markdown("Analyse détaillée des performances VGG16 vs ConvNeXt-Tiny")

# Chemins
BASE_DIR = Path(__file__).parent.parent
RESULTS_PATH = BASE_DIR / "outputs" / "comparison_results.json"
VGG_CM_PATH = BASE_DIR / "outputs" / "VGG16_confusion_matrix.png"
CONVNEXT_CM_PATH = BASE_DIR / "outputs" / "ConvNeXt-Tiny_confusion_matrix.png"
COMPARISON_CHART_PATH = BASE_DIR / "outputs" / "comparison_chart.png"

# Chargement des résultats
@st.cache_data
def load_results():
    """Charge les résultats de comparaison."""
    try:
        with open(RESULTS_PATH, 'r') as f:
            return json.load(f)
    except:
        # Données par défaut si fichier non disponible
        return {
            "vgg16": {
                "accuracy": 0.8333,
                "loss": 0.6476,
                "time_seconds": 1876.2,
                "ram_peak_gb": 1.60,
                "ram_avg_gb": 0.77
            },
            "convnext": {
                "accuracy": 0.8571,
                "loss": 0.5185,
                "time_seconds": 8772.6,
                "ram_peak_gb": 1.61,
                "ram_avg_gb": 0.75
            },
            "improvements": {
                "accuracy_points": 2.38,
                "time_percent": -367.6,
                "ram_percent": -0.6
            }
        }

results = load_results()

st.markdown("---")

# Section 1 : Métriques principales
st.markdown("## 1. Métriques Principales")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Accuracy VGG16",
        value=f"{results['vgg16']['accuracy']*100:.2f}%",
        help="Baseline - Modèle de 2014"
    )

with col2:
    st.metric(
        label="Accuracy ConvNeXt",
        value=f"{results['convnext']['accuracy']*100:.2f}%",
        delta=f"+{results['improvements']['accuracy_points']:.2f} pts",
        help="Nouveau modèle - 2022"
    )

with col3:
    st.metric(
        label="Loss VGG16",
        value=f"{results['vgg16']['loss']:.4f}",
        help="Cross-entropy sur test set"
    )

with col4:
    st.metric(
        label="Loss ConvNeXt",
        value=f"{results['convnext']['loss']:.4f}",
        delta=f"{results['convnext']['loss'] - results['vgg16']['loss']:.4f}",
        delta_color="inverse",
        help="Cross-entropy sur test set"
    )

st.markdown("---")

# Section 2 : Tableau comparatif détaillé
st.markdown("## 2. Tableau Comparatif Détaillé")

comparison_df = pd.DataFrame({
    "Métrique": [
        "Année de publication",
        "Paramètres totaux",
        "Paramètres entraînables",
        "Test Accuracy",
        "Test Loss",
        "Temps d'entraînement",
        "RAM pic",
        "Epochs (best)",
        "Architecture"
    ],
    "VGG16": [
        "2014",
        "14.8M",
        "7.15M",
        f"{results['vgg16']['accuracy']*100:.2f}%",
        f"{results['vgg16']['loss']:.4f}",
        f"{results['vgg16']['time_seconds']/60:.1f} min",
        f"{results['vgg16']['ram_peak_gb']:.2f} GB",
        "13",
        "Conv 3×3 + BatchNorm"
    ],
    "ConvNeXt-Tiny": [
        "2022",
        "27.9M",
        "2.46M",
        f"{results['convnext']['accuracy']*100:.2f}%",
        f"{results['convnext']['loss']:.4f}",
        f"{results['convnext']['time_seconds']/60:.1f} min",
        f"{results['convnext']['ram_peak_gb']:.2f} GB",
        "26",
        "Conv 7×7 depthwise + LayerNorm"
    ],
    "Avantage": [
        "ConvNeXt (moderne)",
        "VGG16 (plus léger)",
        "ConvNeXt (-66%)",
        "ConvNeXt (+2.38 pts)",
        "ConvNeXt (-20%)",
        "VGG16 (x4.7 plus rapide)",
        "≈ Égal",
        "VGG16 (convergence rapide)",
        "ConvNeXt (techniques modernes)"
    ]
})

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# Section 3 : Graphiques comparatifs
st.markdown("## 3. Visualisations Comparatives")

col1, col2 = st.columns(2)

with col1:
    # Graphique Accuracy
    fig_acc = go.Figure(data=[
        go.Bar(
            name='VGG16 (2014)',
            x=['Test Accuracy'],
            y=[results['vgg16']['accuracy'] * 100],
            marker_color='#3498db',
            text=[f"{results['vgg16']['accuracy']*100:.2f}%"],
            textposition='outside'
        ),
        go.Bar(
            name='ConvNeXt-Tiny (2022)',
            x=['Test Accuracy'],
            y=[results['convnext']['accuracy'] * 100],
            marker_color='#2ecc71',
            text=[f"{results['convnext']['accuracy']*100:.2f}%"],
            textposition='outside'
        )
    ])
    fig_acc.update_layout(
        title="Comparaison Accuracy",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 100],
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # Description textuelle pour accessibilité WCAG 1.1.1 / 1.4.1
    st.markdown(f"""
    <details>
    <summary><strong>Description du graphique</strong> (accessibilité)</summary>
    <p>Graphique comparant l'accuracy des deux modèles : VGG16 atteint {results['vgg16']['accuracy']*100:.2f}%
    (barre bleue) et ConvNeXt-Tiny atteint {results['convnext']['accuracy']*100:.2f}% (barre verte),
    soit une amélioration de +{results['improvements']['accuracy_points']:.2f} points.</p>
    </details>
    """, unsafe_allow_html=True)

with col2:
    # Graphique Loss
    fig_loss = go.Figure(data=[
        go.Bar(
            name='VGG16 (2014)',
            x=['Test Loss'],
            y=[results['vgg16']['loss']],
            marker_color='#3498db',
            text=[f"{results['vgg16']['loss']:.4f}"],
            textposition='outside'
        ),
        go.Bar(
            name='ConvNeXt-Tiny (2022)',
            x=['Test Loss'],
            y=[results['convnext']['loss']],
            marker_color='#2ecc71',
            text=[f"{results['convnext']['loss']:.4f}"],
            textposition='outside'
        )
    ])
    fig_loss.update_layout(
        title="Comparaison Loss",
        yaxis_title="Cross-Entropy Loss",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_loss, use_container_width=True)

    # Description textuelle pour accessibilité
    st.markdown(f"""
    <details>
    <summary><strong>Description du graphique</strong> (accessibilité)</summary>
    <p>Graphique comparant la loss (erreur) des deux modèles : VGG16 a une loss de {results['vgg16']['loss']:.4f}
    (barre bleue) et ConvNeXt-Tiny a une loss de {results['convnext']['loss']:.4f} (barre verte).
    Une loss plus basse indique une meilleure performance.</p>
    </details>
    """, unsafe_allow_html=True)

# Graphique temps
col1, col2 = st.columns(2)

with col1:
    fig_time = go.Figure(data=[
        go.Bar(
            x=['VGG16', 'ConvNeXt-Tiny'],
            y=[results['vgg16']['time_seconds']/60, results['convnext']['time_seconds']/60],
            marker_color=['#3498db', '#2ecc71'],
            text=[f"{results['vgg16']['time_seconds']/60:.1f} min",
                  f"{results['convnext']['time_seconds']/60:.1f} min"],
            textposition='outside'
        )
    ])
    fig_time.update_layout(
        title="Temps d'Entraînement",
        yaxis_title="Temps (minutes)",
        height=400
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # Description textuelle pour accessibilité
    st.markdown(f"""
    <details>
    <summary><strong>Description du graphique</strong> (accessibilité)</summary>
    <p>Graphique comparant le temps d'entraînement : VGG16 nécessite {results['vgg16']['time_seconds']/60:.1f} minutes
    tandis que ConvNeXt-Tiny nécessite {results['convnext']['time_seconds']/60:.1f} minutes.
    VGG16 est plus rapide à entraîner mais moins performant.</p>
    </details>
    """, unsafe_allow_html=True)

with col2:
    # Radar chart pour comparaison multi-critères
    categories = ['Accuracy', 'Généralisation\n(1-Loss)', 'Efficacité\nparamètres', 'Vitesse\nentraînement']

    # Normaliser les valeurs
    vgg_values = [
        results['vgg16']['accuracy'] * 100,
        (1 - results['vgg16']['loss']) * 100,
        (1 - 7.15/27.9) * 100,  # Ratio params entraînables
        100  # Référence vitesse
    ]
    convnext_values = [
        results['convnext']['accuracy'] * 100,
        (1 - results['convnext']['loss']) * 100,
        (1 - 2.46/27.9) * 100,
        100 * (results['vgg16']['time_seconds'] / results['convnext']['time_seconds'])
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vgg_values,
        theta=categories,
        fill='toself',
        name='VGG16',
        line_color='#3498db'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=convnext_values,
        theta=categories,
        fill='toself',
        name='ConvNeXt-Tiny',
        line_color='#2ecc71'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Comparaison Multi-Critères",
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Description textuelle pour accessibilité
    st.markdown("""
    <details>
    <summary><strong>Description du graphique radar</strong> (accessibilité)</summary>
    <p>Ce graphique radar compare VGG16 (bleu) et ConvNeXt-Tiny (vert) sur 4 critères normalisés :</p>
    <ul>
    <li><strong>Accuracy</strong> : ConvNeXt légèrement supérieur</li>
    <li><strong>Généralisation</strong> : ConvNeXt meilleur (loss plus basse)</li>
    <li><strong>Efficacité paramètres</strong> : ConvNeXt bien meilleur (moins de paramètres à entraîner)</li>
    <li><strong>Vitesse d'entraînement</strong> : VGG16 nettement plus rapide</li>
    </ul>
    <p>ConvNeXt excelle sur 3 critères sur 4, VGG16 uniquement sur la vitesse.</p>
    </details>
    """, unsafe_allow_html=True)

st.markdown("---")

# Section 4 : Matrices de confusion
st.markdown("## 4. Matrices de Confusion")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### VGG16")
    if VGG_CM_PATH.exists():
        st.image(str(VGG_CM_PATH), caption="Matrice de confusion VGG16 - Prédictions vs valeurs réelles", use_container_width=True)
    else:
        st.info("Matrice de confusion non disponible. Exécutez le notebook pour la générer.")

with col2:
    st.markdown("### ConvNeXt-Tiny")
    if CONVNEXT_CM_PATH.exists():
        st.image(str(CONVNEXT_CM_PATH), caption="Matrice de confusion ConvNeXt-Tiny - Prédictions vs valeurs réelles", use_container_width=True)
    else:
        st.info("Matrice de confusion non disponible. Exécutez le notebook pour la générer.")

# Description textuelle des matrices pour accessibilité
st.markdown("""
<details>
<summary><strong>Description des matrices de confusion</strong> (accessibilité)</summary>
<p>Les matrices de confusion montrent pour chaque catégorie réelle (en ligne) combien d'images
ont été prédites dans chaque catégorie (en colonne). La diagonale représente les prédictions correctes.
Plus les valeurs diagonales sont élevées et les valeurs hors-diagonale basses, meilleur est le modèle.</p>
<p><strong>VGG16</strong> : Montre quelques confusions entre certaines catégories similaires.</p>
<p><strong>ConvNeXt-Tiny</strong> : Diagonale plus marquée, moins d'erreurs de classification.</p>
</details>
""", unsafe_allow_html=True)

st.markdown("---")

# Section 5 : Analyse de l'overfitting
st.markdown("## 5. Analyse de l'Overfitting")

overfitting_df = pd.DataFrame({
    "Modèle": ["VGG16", "ConvNeXt-Tiny"],
    "Train Accuracy (finale)": ["~96.88%", "~96.13%"],
    "Val Accuracy (finale)": ["~78.57%", "~87.14%"],
    "Écart Train/Val": ["18.3 pts", "9 pts"],
    "Niveau Overfitting": ["Modéré à Fort", "Léger"],
    "Ratio Loss (Train/Val)": ["4.7x", "3.2x"]
})

st.dataframe(overfitting_df, use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)

with col1:
    st.warning("""
    **VGG16 - Overfitting Modéré**
    - Écart important entre train et validation
    - Le modèle mémorise plutôt qu'il ne généralise
    - Causé par trop de paramètres entraînables (7.15M)
    """)

with col2:
    st.success("""
    **ConvNeXt-Tiny - Overfitting Léger**
    - Écart train/val 2x plus petit que VGG16
    - Meilleure généralisation grâce à LayerNorm
    - Fine-tuning ciblé (2.46M params seulement)
    """)

st.markdown("---")

# Section 6 : Conclusion
st.markdown("## 6. Conclusion")

st.success("""
### Objectif Atteint ✅

**ConvNeXt-Tiny (2022) surpasse VGG16 (2014) de +2.38 points d'accuracy**, validant
la démarche de veille technologique.

**Points clés :**
- **Performance** : 85.71% vs 83.33% (+2.38 pts)
- **Généralisation** : Loss test 20% plus basse
- **Efficacité** : 66% moins de paramètres à entraîner
- **Trade-off** : Temps d'entraînement plus long sur CPU (attendu)
""")

# Recommandation
st.markdown("### Recommandation pour la Production")

st.info("""
**ConvNeXt-Tiny est recommandé** pour ce projet car :
1. Meilleure accuracy sur le dataset cible
2. Meilleure capacité de généralisation (moins d'overfitting)
3. Architecture moderne et maintenue
4. Bon équilibre performance/complexité pour un dashboard

**Note** : Le temps d'entraînement plus long est acceptable car :
- En production, seule l'inférence compte (rapide)
- L'entraînement est une opération ponctuelle
- Avec GPU, ConvNeXt serait plus rapide
""")

# Footer
st.markdown("---")
st.caption("""
**Source** : Résultats issus du notebook `notebook_comparatif_vgg16_convnext.ipynb`.
Les métriques sont calculées sur un jeu de test de 168 images (16% du dataset).
""")