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
---------------------

.PHONY: help install test init config ingest extract convert label label-report discover \
        screen verify relevance explain \
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

screen:          ## label-free feature screen, pre-label by default (DATASETS="a b", ZONE=parquet|labelled, CAPTURES="Monday Tuesday", EMIT=1, OUT=path)
	$(TALOS) screen --dataset $(or $(DATASETS),$(DATASET)) --zone $(or $(ZONE),parquet) \
		$(if $(CAPTURES),--captures $(CAPTURES),) \
		$(if $(FEATURES),--features $(FEATURES),) $(if $(EMIT),--emit $(OUT),)

verify:          ## rows that cannot be true of a real flow, pre-label by default (DATASETS="a b", ZONE=parquet|labelled, CAPTURES="Monday Tuesday")
	$(TALOS) verify --dataset $(or $(DATASETS),$(DATASET)) --zone $(or $(ZONE),parquet) \
		$(if $(CAPTURES),--captures $(CAPTURES),)

relevance:       ## task relevance + domain probe -> schema v1 (DATASETS="a b", EMIT=1, OUT=path)
	$(TALOS) relevance --dataset $(or $(DATASETS),$(DATASET)) \
		$(if $(FEATURES),--features $(FEATURES),) $(if $(EMIT),--emit $(OUT),)

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
			--out /tmp/eda_smoke_$$d.json || exit 1; done
	@echo "smoke ok — schema, spec and query all agree with the lake; now: make eda-all"

eda:             ## profile one dataset, no comparison by default (DATASET=…; WITH=<dataset> for pairwise, ALL=1 for the whole directory)
	$(TALOS) eda --dataset $(DATASET) \
		$(if $(WITH),--compare-with $(WITH),) $(if $(ALL),--compare-all,)

eda-all:         ## profile every dataset, then rebuild all comparisons once at the end
	@for d in $(DATASETS); do echo "=== $$d ==="; \
		$(TALOS) eda --dataset $$d \
			|| exit 1; done
	$(TALOS) compare --render

eda-compare:     ## rebuild every comparison from existing profiles (no lake access)
	$(TALOS) compare --render

eda-render:      ## rebuild the HTML from existing JSON
	$(TALOS) render
