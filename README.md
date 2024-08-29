# Disorder-type-prediction

This repository contains the models to predict the disorder type of a compound (O-ordered, S-substitutional disorder, V-vacancies, M-mixed). This is classification done with the latest disorder script on ICSD (in which there is MC calculation of entropy and only six orbit disorder types: O, S, V, SV, VP, SVP. See corresponding paper draft for details)

Trained models are availible on the external hard drive. Models are trained on compounds with no hydrogen. However, there is data extracted for the compounds with hydrogens as well, so it is possible to eliminate this feature of the model by retraining the model on all compounds with "full structures" (which are defined as those having the same set of elements in structure and in composition).

Again there is a generic environment file, appologies

**List of models:**

**RF + Magpie**
Balanced accuracy = 0.75, f1-score (macro) = 0.77, mcc = 0.73

**CrabNet + Mat2Vec**
Balanced accuracy = 0.78, f1-score (macro) = 0.78, mcc = 0.75

**Ensemble-10CrabNet + Mat2Vec**
Balanced accuracy = 0.80, f1-score (macro) = 0.81, mcc = 0.79

Confusion matrixes are in images folder
