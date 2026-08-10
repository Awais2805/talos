# Lake location lives in config.yml (lake.root) -- no target carries a bucket
# name any more; every stage resolves its own paths through Config.
DATASET ?= cic-ids-2017
# prefer the repo venv if present, else system python
PY := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
# the console script installed by `pip install -e .`
TALOS := $(shell test -x .venv/bin/talos && echo .venv/bin/talos || echo talos)

DATASETS = cic-ids-2017 cic-ids-2018 cic-ddos-2019

# EDA guard rails: cap the DuckDB spill so a long scan cannot fill the root disk
# and wedge sshd, and keep threads/memory off the settings that thrash-froze the
# box during conversion. Defaults live here rather than in the command line so
# the safe run is the one you get without remembering anything.
#
# These are sized for a 16 GB machine. The EC2 box has been resized more than
# once -- check `nproc` and `free -g` and override rather than assuming.
DUCK_TMP  ?= /tmp/talos_duckdb
EDA_FLAGS ?= --temp-dir $(DUCK_TMP) --threads 4 --memory-limit 8GB

# ---------------------------------------------------------------------------
# Labelling and label validation were removed on 2026-08-04 and are being
# rebuilt from scratch. The post-mortem on what was removed and why is in
# docs/label_pipeline_audit.md -- read it before re-adding targets here.
#
# The EDA targets below are the only stage downstream of `discover`. Nothing
# downstream of EDA (map/merge/clean/encode/split/train) exists yet. The lake
# still holds the raw, extracted and parquet zones; the labelled zone was
# cleared when the labelling module was removed.
# ---------------------------------------------------------------------------

.PHONY: help install test init config extract convert discover \
        eda eda-all eda-smoke eda-compare eda-render

help:            ## show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

init:            ## create a local lake + config (LAKE=./lake to place it)
	$(TALOS) init $(or $(LAKE),./lake)

config:          ## show where every zone resolves to
	$(TALOS) config

install:         ## create .venv and install the package in editable mode
	python3 -m venv .venv
	.venv/bin/python -m pip install -e ".[dev]"

test:            ## golden regression over the committed EDA fixtures (no lake needed)
	$(PY) -m pytest tests/ -q

extract:         ## run the configured extractor over the raw zone (DATASET=…)
	$(TALOS) extract $(DATASET)

convert:         ## convert extracted logs to parquet/csv (DATASET=…, FORMAT=parquet|csv|both)
	$(TALOS) convert --dataset $(DATASET) $(if $(FORMAT),--format $(FORMAT),)

discover:        ## profile the lake by log type -> reports/lake_features_report.txt
	$(TALOS) discover --non_interactive \
		> reports/lake_features_report.txt

eda-smoke:       ## 200k-row dry run of every dataset -> /tmp, proves the pipeline in seconds
	@mkdir -p $(DUCK_TMP)
	@for d in $(DATASETS); do echo "=== $$d ==="; \
		$(TALOS) eda --dataset $$d --limit 200000 \
			--out /tmp/eda_smoke_$$d.json --no-cascade $(EDA_FLAGS) || exit 1; done
	@echo "smoke ok — schema, spec and query all agree with the lake; now: make eda-all"

eda:             ## profile one dataset -> its report + regen every comparison (DATASET=…)
	@mkdir -p $(DUCK_TMP)
	$(TALOS) eda --dataset $(DATASET) $(EDA_FLAGS)

eda-all:         ## profile every dataset, then rebuild all reports once at the end
	@mkdir -p $(DUCK_TMP)
	@for d in $(DATASETS); do echo "=== $$d ==="; \
		$(TALOS) eda --dataset $$d --no-cascade $(EDA_FLAGS) \
			|| exit 1; done
	$(TALOS) compare --render

eda-compare:     ## rebuild every comparison from existing profiles (no lake access)
	$(TALOS) compare --render

eda-render:      ## rebuild the HTML from existing JSON
	$(TALOS) render
