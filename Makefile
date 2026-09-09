# Reproduce every number in REPORT.md from the raw CSV.
#
#   make install    install pinned dependencies
#   make test       run the test suite
#   make all        run every experiment and regenerate results/
#   make clean      delete generated results and logs

PYTHON ?= python3
CSV    ?= data/kidney_disease.csv

.PHONY: all install test primary outlier ablation missing tuned missingness \
        compare leakage clean

all: primary outlier ablation missing tuned missingness compare
	@echo "All experiments complete. Tables and figures are in results/."

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -q

# --- Experiment 1: primary analysis ----------------------------------------
primary:
	$(PYTHON) src/ckd_pipeline.py --csv $(CSV) --experiment no_outlier

# --- Experiment 2: fold-safe IQR winsorisation -----------------------------
outlier:
	$(PYTHON) src/ckd_pipeline.py --csv $(CSV) --experiment outlier_capped

# --- Experiment 3: diagnostic criteria removed -----------------------------
ablation:
	$(PYTHON) src/ckd_pipeline.py --csv $(CSV) --experiment ablation

# --- Experiment 4: missingness as explicit features ------------------------
missing:
	$(PYTHON) src/ckd_pipeline.py --csv $(CSV) --experiment missing_indicator

# --- Experiment 5: nested CV hyperparameter tuning -------------------------
tuned:
	$(PYTHON) src/ckd_pipeline.py --csv $(CSV) --experiment no_outlier --tune \
	    --stability-repeats 0

# --- Is missingness itself predictive? -------------------------------------
missingness:
	$(PYTHON) src/missingness_report.py --csv $(CSV)

# --- Side-by-side table across experiments ---------------------------------
compare:
	$(PYTHON) src/build_comparison.py

# --- Standalone leakage demonstration --------------------------------------
leakage:
	PYTHONPATH=src $(PYTHON) src/leakage_demo.py

clean:
	rm -rf results/*/ results/table_experiment_comparison.csv logs/
