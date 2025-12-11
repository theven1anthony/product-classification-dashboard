# Product Classification Dashboard

Dashboard Streamlit pour la classification automatique de produits e-commerce avec comparaison VGG16 vs ConvNeXt-Tiny.

## Fonctionnalités

- **EDA** : Exploration du dataset Flipkart (1050 images, 7 catégories)
- **Prédiction** : Classification d'images avec visualisation Grad-CAM
- **Résultats** : Comparaison détaillée VGG16 vs ConvNeXt-Tiny

## Résultats

| Modèle | Accuracy | Loss |
|--------|----------|------|
| VGG16 (2014) | 83.33% | 0.6476 |
| **ConvNeXt-Tiny (2022)** | **85.71%** | **0.5185** |

**Amélioration : +2.38 points d'accuracy**

## Installation locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Accessibilité

Ce dashboard respecte les critères WCAG :
- 1.1.1 Contenu non textuel
- 1.4.1 Utilisation de la couleur
- 1.4.3 Contraste minimum
- 1.4.4 Redimensionnement du texte
- 2.4.2 Titre de page

## Projet

Développé dans le cadre de la formation Ingénieur IA - OpenClassrooms.

Modèle basé sur l'article : [A ConvNet for the 2020s (Liu et al., 2022)](https://arxiv.org/abs/2201.03545)