# Disorder-type-prediction

This repository contains the models to predict the disorder type of the compound (O-ordered, S-substitutional disorder, V-vacancies, M-mixed). This is classification done with the latest disorder script on ICSD (in which there is MC calculation of entropy and only six orbit disorder types: O, S, V, SV, VP, SVP. See corresponding paper draft for details)

Trained models are availible on the external hard drive

Again there is a generic environment file, appologies

**List of models:**

**RF + Magpie**
Balanced accuracy = 0.75, f1-score (macro) = 0.77, mcc = 0.73

**CrabNet + Mat2Vec**
Balanced accuracy = 0.78, f1-score (macro) = 0.78, mcc = 0.75

**Ensemble-10CrabNet + Mat2Vec**
Balanced accuracy = 0.80, f1-score (macro) = 0.81, mcc = 0.79
