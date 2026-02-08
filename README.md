# Comparing Hierarchical Models: Statistical vs Tree-Based vs Neural Networks

Code for our paper in *Machine Learning with Applications* (Elsevier, 2025).

## What's this about?

Healthcare data is naturally hierarchical, patients are nested within hospitals, hospitals within regions. But most prediction models ignore this structure. We wanted to know: **which modeling approach handles hierarchical data best?**

We compared three fundamentally different approaches:
- **Statistical** — Hierarchical Mixed Models (the classic approach)
- **Tree-based** — Hierarchical Random Forest (our adaptation)  
- **Neural** — Hierarchical Neural Networks (with entity embeddings)

Spoiler: tree-based models won on most metrics, but each approach has its strengths.

## Quick start
```bash
pip install -r requirements.txt
python run_demo.py
```

## Citation
```
Shahbazi, M.A., & Azadeh-Fard, N. (2025). Hierarchical data modeling: A systematic 
comparison of statistical, tree-based, and neural network approaches. 
Machine Learning with Applications, 100688.
```

## Contact

Marzieh Amiri Shahbazi — ma7684@g.rit.edu
