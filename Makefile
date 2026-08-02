BUCKET  = ids-datalakec48eb2cab942494ba5059fac3b3527d9
DATASET ?= cic-ids-2017
# prefer the repo venv if present, else system python
PY := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

DATASETS = cic-ids-2017 cic-ids-2018 cic-ddos-2019

# EDA guard rails for the 16 GB box: cap the DuckDB spill so a long scan cannot
# fill the root disk and wedge sshd, and keep threads/memory off the settings
# that thrash-froze it during conversion. Defaults live here rather than in the
# command line so the safe run is the one you get without remembering anything.
DUCK_TMP  ?= /tmp/talos_duckdb
EDA_FLAGS ?= --temp-dir $(DUCK_TMP) --threads 4 --memory-limit 8GB
# Validation scans the same lake with the same box underneath it, so it gets the
# same guard rails. Kept separate from EDA_FLAGS because the two stages will
# diverge — the deep tiers need more temp than a profile pass ever does.
VAL_FLAGS ?= --temp-dir $(DUCK_TMP) --threads 4 --memory-limit 8GB

.PHONY: help extract convert discover label label-all \
        eda eda-all eda-smoke eda-compare eda-render \
        validate validate-all validate-deep validate-cross validate-render validate-gate \
        validate-sample validate-score oracle-fetch oracle-stage oracle-join

help:            ## show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

extract:         ## run Zeek over the raw pcaps (on the EC2 box; RAW_ONLY=… to scope)
	bash src/data/zeek_batch.sh

convert:         ## mirror one dataset's Zeek logs to parquet, 1:1, same tree (DATASET=…)
	$(PY) src/data/to_parquet.py \
		--input  s3://$(BUCKET)/extracted/$(DATASET) \
		--output s3://$(BUCKET)/parquets/$(DATASET)

discover:        ## profile the lake by log type -> reports/lake_features_report.txt
	$(PY) src/preprocess/lake_feature_discovery.py --non_interactive \
		> reports/lake_features_report.txt

label:           ## label one dataset -> lake labelled zone (DATASET=…)
	$(PY) src/preprocess/label/label_flows.py --dataset $(DATASET)

label-all:       ## label every dataset
	@for d in $(DATASETS); do echo "=== $$d ==="; \
		$(PY) src/preprocess/label/label_flows.py --dataset $$d || exit 1; done

eda-smoke:       ## 200k-row dry run of every dataset -> /tmp, proves the pipeline in seconds
	@mkdir -p $(DUCK_TMP)
	@for d in $(DATASETS); do echo "=== $$d ==="; \
		$(PY) src/eda/profile_dataset.py --dataset $$d --limit 200000 \
			--out /tmp/eda_smoke_$$d.json --no-cascade $(EDA_FLAGS) || exit 1; done
	@echo "smoke ok — schema, spec and query all agree with the lake; now: make eda-all"

eda:             ## profile one dataset -> its report + regen every comparison (DATASET=…)
	@mkdir -p $(DUCK_TMP)
	$(PY) src/eda/profile_dataset.py --dataset $(DATASET) $(EDA_FLAGS)

eda-all:         ## profile every dataset, then rebuild all reports once at the end
	@mkdir -p $(DUCK_TMP)
	@for d in $(DATASETS); do echo "=== $$d ==="; \
		$(PY) src/eda/profile_dataset.py --dataset $$d --no-cascade $(EDA_FLAGS) \
			|| exit 1; done
	$(PY) src/eda/compare.py --render

eda-compare:     ## rebuild every comparison from existing profiles (no lake access)
	$(PY) src/eda/compare.py --render

eda-render:      ## rebuild the HTML from existing JSON
	$(PY) src/eda/render.py

# The validation runner exits 1 when it finds something at or above the policy's
# blocking severity. That is a result, not a failure, so these targets keep going
# and rebuild the report either way; `validate-gate` is the one place a result
# turns into an exit code.
validate:        ## label-validation tiers 0-3 for one dataset -> its report (DATASET=…)
	@mkdir -p $(DUCK_TMP)
	@$(PY) src/label_validation/runner.py --dataset $(DATASET) $(VAL_FLAGS); s=$$?; \
		$(PY) src/label_validation/report/render.py; exit $$s

validate-all:    ## validate every dataset, then rebuild all reports once at the end
	@mkdir -p $(DUCK_TMP)
	@for d in $(DATASETS); do echo "=== $$d ==="; \
		$(PY) src/label_validation/runner.py --dataset $$d $(VAL_FLAGS) || true; done
	$(PY) src/label_validation/report/render.py
	@echo "reports -> reports/validation/index.html — now decide: make validate-gate"

validate-deep:   ## tier 4: contradictions provable from the labels alone (DATASET=…)
	@mkdir -p $(DUCK_TMP)
	@$(PY) src/label_validation/runner.py --dataset $(DATASET) --tier 4 $(VAL_FLAGS); s=$$?; \
		$(PY) src/label_validation/report/render.py; exit $$s

validate-cross:  ## tier 6: cross-dataset class semantics from existing JSON (no lake access)
	@$(PY) src/label_validation/runner.py --dataset $(DATASET) --tier 6; s=$$?; \
		$(PY) src/label_validation/report/render.py; exit $$s

validate-render: ## rebuild the validation HTML from existing JSON
	$(PY) src/label_validation/report/render.py

# The external oracle is the DistriNet CNS2022 corrected label set -- an
# independently produced, hand-forensicated labelling of 2017 and 2018. It is the
# only source in this pipeline that did not derive from the CIC schedule, so it is
# the one thing that can catch an error we inherited rather than introduced.
# Fetching is opt-in because the 2018 archive is 9.7 GB.
oracle-fetch:    ## download the DistriNet corrected label sets (large; prints the command by default)
	$(PY) src/label_validation/oracle/stage.py --fetch --dataset $(DATASET)

oracle-stage:    ## unzip + convert the corrected CSVs to parquet (DATASET=…)
	$(PY) src/label_validation/oracle/stage.py --stage --dataset $(DATASET)

oracle-join:     ## join the oracle to our labelled flows -> reports/validation/oracle_<ds>.json
	@mkdir -p $(DUCK_TMP)
	$(PY) src/label_validation/oracle/stage.py --join --dataset $(DATASET) $(VAL_FLAGS)

# Certification is what turns "the checks are quiet" into a precision figure with
# an interval on it. The sample carries its evidence so a flow can be adjudicated
# from the row rather than by going back to the lake.
validate-sample: ## export a stratified certification sample for adjudication (DATASET=…)
	@mkdir -p $(DUCK_TMP)
	$(PY) src/label_validation/certify/sample.py --dataset $(DATASET) $(VAL_FLAGS)
	@echo "adjudicate reports/validation/samples/$(DATASET).csv, then: make validate-score DATASET=$(DATASET)"

validate-score:  ## score an adjudicated sample -> per-class precision + Wilson intervals
	$(PY) src/label_validation/certify/score.py --dataset $(DATASET)

validate-gate:   ## exit non-zero if any validated dataset carries a blocking finding
	$(PY) src/label_validation/report/gate.py --all
