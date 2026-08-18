# Lake location lives in config.yml (lake.root) -- no target carries a bucket
# name any more; every stage resolves its own paths through Config.
DATASET ?= cic-ids-2017
# prefer the repo venv if present, else system python
PY := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
# the console script installed by `pip install -e .`
TALOS := $(shell test -x .venv/bin/talos && echo .venv/bin/talos || echo talos)

DATASETS = cic-ids-2017 cic-ids-2018 cic-ddos-2019

# Resource limits (memory, threads, spill directory and its cap) are detected
# per machine and declared in config.yml under `resources:`. They are no longer
# guessed at here -- `talos config` shows what this box resolved to.

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

.PHONY: help install test init config ingest extract convert label label-report discover \
        screen validity relevance explain \
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

ingest:          ## put captures into a source's raw zone (DATASET=… SOURCE=… FILES=…)
	$(TALOS) ingest --dataset $(DATASET) $(if $(SOURCE),--source $(SOURCE),) $(FILES)

extract:         ## run the configured extractor over the raw zone (DATASET=…)
	$(TALOS) extract $(DATASET)

convert:         ## convert extracted logs to parquet/csv (DATASET=…, FORMAT=parquet|csv|both)
	$(TALOS) convert --dataset $(DATASET) $(if $(FORMAT),--format $(FORMAT),)

label:           ## attach ground truth from the attack schedule (DATASET=…)
	$(TALOS) label --dataset $(DATASET)

label-report:    ## same, but report only — writes nothing (DATASET=…)
	$(TALOS) label --dataset $(DATASET) --no-write

label-method:    ## label by a chosen method (DATASET=… METHOD=ae|tabcl|fused)
	$(TALOS) label --dataset $(DATASET) --method $(METHOD)

label-all-methods: ## the whole Path B chain for one dataset (DATASET=…)
	$(TALOS) label --dataset $(DATASET) --method schedule
	$(TALOS) label --dataset $(DATASET) --method ae
	$(TALOS) label --dataset $(DATASET) --method tabcl
	$(TALOS) label --dataset $(DATASET) --method fused

screen:          ## label-free feature screen over one or more datasets (DATASETS="a b")
	$(TALOS) screen --dataset $(or $(DATASETS),$(DATASET)) \
		$(if $(FEATURES),--features $(FEATURES),) $(if $(EMIT),--emit,)

validity:        ## rows that cannot be true of a real flow (DATASETS="a b")
	$(TALOS) validity --dataset $(or $(DATASETS),$(DATASET))

relevance:       ## task relevance + domain probe -> schema v1 (DATASETS="a b")
	$(TALOS) relevance --dataset $(or $(DATASETS),$(DATASET)) \
		$(if $(FEATURES),--features $(FEATURES),) $(if $(EMIT),--emit,)

explain:         ## which line of Eq. 17 decided each row (DATASET=… METHOD=fused)
	$(TALOS) label --dataset $(DATASET) --method $(or $(METHOD),fused) --explain

audit:           ## emit audit candidates for a person (DATASET=… FLOOR=…)
	$(TALOS) audit emit --dataset $(DATASET) $(if $(FLOOR),--floor $(FLOOR),)

audit-page:      ## rebuild the adjudication page from an existing audit file (DATASET=…)
	$(TALOS) audit render --dataset $(DATASET)

audit-status:    ## how much of the audit file has been adjudicated (DATASET=…)
	$(TALOS) audit status --dataset $(DATASET)

benchmark:       ## score methods against the adjudicated slice (DATASET=… METHODS="a b")
	$(TALOS) audit benchmark --dataset $(DATASET) --compare $(METHODS)

discover:        ## profile the lake by log type -> reports/lake_features_report.txt
	$(TALOS) discover --non_interactive \
		> reports/lake_features_report.txt

eda-smoke:       ## 200k-row dry run of every dataset -> /tmp, proves the pipeline in seconds
	@for d in $(DATASETS); do echo "=== $$d ==="; \
		$(TALOS) eda --dataset $$d --limit 200000 \
			--out /tmp/eda_smoke_$$d.json --no-cascade || exit 1; done
	@echo "smoke ok — schema, spec and query all agree with the lake; now: make eda-all"

eda:             ## profile one dataset -> its report + regen every comparison (DATASET=…)
	$(TALOS) eda --dataset $(DATASET)

eda-all:         ## profile every dataset, then rebuild all reports once at the end
	@for d in $(DATASETS); do echo "=== $$d ==="; \
		$(TALOS) eda --dataset $$d --no-cascade \
			|| exit 1; done
	$(TALOS) compare --render

eda-compare:     ## rebuild every comparison from existing profiles (no lake access)
	$(TALOS) compare --render

eda-render:      ## rebuild the HTML from existing JSON
	$(TALOS) render
