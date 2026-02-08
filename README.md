# Comparing Hierarchical Models: Statistical vs Tree-Based vs Neural Networks

Code for our paper in *Machine Learning with Applications* (Elsevier, 2025).

## What's this about?

Healthcare data is naturally hierarchical, patients are nested within hospitals, hospitals within regions. But most prediction models ignore this structure. We wanted to know: **which modeling approach handles hierarchical data best?**

We compared three fundamentally different approaches:
- **Statistical** — Hierarchical Mixed Models (the classic approach)
- **Tree-based** — Hierarchical Random Forest (our adaptation)  
- **Neural** — Hierarchical Neural Networks (with entity embeddings)

Spoiler: tree-based models won on most metrics, but each approach has its strengths.

## Data

We used two datasets:
1. **NIS 2019** — 7+ million hospital stays from ~4,500 hospitals across 4 US regions
2. **MIMIC-IV** — ICU data for external validation

The hierarchy: Patient → Hospital → Region

## Key findings

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **Tree-based (HRF)** | Best accuracy, fast, handles mixed data well | Less interpretable than statistical |
| **Neural (HNN)** | Captures group distinctions, flexible | Slow, prediction bias, needs tuning |
| **Statistical (HMM)** | Interpretable, principled uncertainty | Lower accuracy, convergence issues |

Tree-based models consistently outperformed the others across different sample sizes, simplified hierarchies, and the external MIMIC dataset.


## Quick start

```bash
pip install -r requirements.txt
```

```python
from src.models import HierarchicalRandomForest, HierarchicalNeuralNetwork, HierarchicalMixedModel

# Tree-based model
hrf = HierarchicalRandomForest()
hrf.fit(X_train, y_train, groups_train)
predictions = hrf.predict(X_test, groups_test)

# Neural model
hnn = HierarchicalNeuralNetwork(input_dim, num_hospitals, num_regions)
hnn.fit(X_train, y_train, groups_train)

# Statistical model  
hmm = HierarchicalMixedModel()
hmm.fit(X_train, y_train, groups_train)
```

## Data access

- **NIS**: Available from [HCUP](https://www.hcup-us.ahrq.gov/nisoverview.jsp) (requires DUA)
- **MIMIC-IV**: Available from [PhysioNet](https://physionet.org/content/mimiciv/) (requires credentialing)

## Citation

```
Shahbazi, M.A., & Azadeh-Fard, N. (2025). Hierarchical data modeling: A systematic 
comparison of statistical, tree-based, and neural network approaches. 
Machine Learning with Applications, 100688.
```

## Contact

Marzieh Amiri Shahbazi  
ma7684@g.rit.edu
