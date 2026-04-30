# Makefile for thesis pipeline
# Usage: make data | train | eval | figures | tables | thesis | archive | all

.PHONY: data train ensemble eval figures tables thesis archive all clean

data:
	python src/scripts/01_prepare_data.py

train:
	python src/scripts/02_train_base_learners.py
	python src/scripts/03_train_foundation_models.py

ensemble:
	python src/scripts/04_train_meta_learner.py

eval:
	python src/scripts/05_compute_intervals_cqr.py
	python src/scripts/06_run_hpa_simulation.py
	python src/scripts/07_run_bcf_evaluation.py
	python src/scripts/08_compute_shap.py
	python src/scripts/09_run_statistical_tests.py

figures:
	python src/scripts/10_generate_figures.py

tables:
	python src/scripts/11_generate_tables.py

thesis:
	cd thesis/manuscript && latexmk -pdf -output-directory=build main.tex

# Frozen submission snapshot
# Usage: make archive DATE=2026-05-08 LABEL=first_version
archive:
	@if [ -z "$(DATE)" ] || [ -z "$(LABEL)" ]; then \
		echo "Usage: make archive DATE=YYYY-MM-DD LABEL=first_version"; exit 1; \
	fi
	mkdir -p submission/$(DATE)_$(LABEL)
	cp -r data/processed results reports thesis docs src submission/$(DATE)_$(LABEL)/
	cd submission/$(DATE)_$(LABEL) && find . -type f -exec sha256sum {} \; > MANIFEST.md
	chmod -R a-w submission/$(DATE)_$(LABEL)
	@echo "Frozen snapshot at submission/$(DATE)_$(LABEL)/"

all: data train ensemble eval figures tables thesis

clean:
	rm -rf thesis/manuscript/build
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
