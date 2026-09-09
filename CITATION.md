# Data source and citation

## Dataset

**Chronic Kidney Disease** data set, UCI Machine Learning Repository.

* Creators: L. Rubini, P. Soundarapandian, P. Eswaran.
* Donated to the UCI repository in 2015; assembled from hospital records.
* 400 patient records, 24 clinical attributes plus a binary class label.
* Repository page: <https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease>
* DOI: <https://doi.org/10.24432/C5G020>
* Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

Because the licence is CC BY 4.0, redistributing `data/kidney_disease.csv`
inside this repository is permitted provided the creators above are credited,
which is the purpose of this file.

The copy in `data/kidney_disease.csv` is the repository file, unmodified. All
cleaning happens in code (`load_and_clean` in `src/ckd_pipeline.py`) so that
the raw defects documented in REPORT.md &sect;1.2 remain inspectable.

## Data provenance and ethics

The records are de-identified: the file carries no name, address, admission
date or hospital identifier, and the only key column (`id`) is dropped before
any analysis. No attempt is made in this project to re-identify any patient,
and none of the fields would support it.

The cohort is small (400 patients), drawn from a single clinical source, and
its 62.5% CKD prevalence is a sampling artefact of how the records were
assembled -- not a population prevalence. Any statement in
this repository about model performance is a statement about *this* benchmark.

**This project is coursework. Nothing in it is a medical device, and no
output should be used to inform the care of any patient.** A model trained on
these features would in any case largely be re-deriving the diagnostic
criteria it was given; see the `ablation` experiment in REPORT.md.

## Citing this repository

If you refer to the analysis rather than the data:

```
CKD Classification -- Minor Project.
https://github.com/Junecoderr/ckd
```
