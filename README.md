# CLEFO — BirdCLEF+ 2026

Repo de compétition Kaggle : [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)

**Tâche** : identifier des espèces animales depuis des enregistrements audio passifs (Pantanal, Amérique du Sud)
**Métrique** : class mean Average Precision (cmAP)
**Deadline** : 3 juin 2026

---

## Structure

```
notebooks/
  birdclef2026_baseline.ipynb   ← Baseline complet (EfficientNetV2-S + Mel Spec)
```

## Pipeline baseline

```
Audio (OGG) → Chunk 5s → Mel Spectrogram (128 bins)
    → EfficientNetV2-S → Multilabel BCE Loss
    → Mixup + SpecAugment
    → Soumission (cmAP)
```

## Comment utiliser sur Kaggle

1. Créer un nouveau notebook sur Kaggle
2. Importer `notebooks/birdclef2026_baseline.ipynb`
3. Activer le GPU (Settings → Accelerator → GPU T4 x2)
4. Ajouter la compétition comme source de données
5. Lancer toutes les cellules → `submission.csv` généré automatiquement

## Roadmap

- [x] Baseline EfficientNetV2-S
- [ ] BEATs / AST (transformer audio pré-entraîné)
- [ ] Ensemble multi-modèles
- [ ] Optimisation de seuil par classe
- [ ] TTA (Test Time Augmentation)
