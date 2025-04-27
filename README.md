# DcatQueryClassifier

Projekt, ktorý klasifikuje otázky v prirodzenom jazyku na predpripravené SPARQL dotazy.

## Ako spustiť projekt

1. Klonujte tento repozitár:
   git clone https://github.com/yourusername/DcatQueryClassifier.git

2. Nainštalujte potrebné knižnice:

pip install -r requirements.txt

3. Trénujte model:

python src/train.py

4. Po natrénovaní modelu môžete predpovedať dotazy:

python src/predict.py "chcem datasety po 14.6.2010"

