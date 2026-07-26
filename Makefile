BUCKET  = ids-datalakec48eb2cab942494ba5059fac3b3527d9
DATASET ?= cic-ids-2017
# prefer the repo venv if present, else system python
PY := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

.PHONY: help extract convert discover
help:            ## show available targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

extract:         ## run Zeek over the raw pcaps (run on the EC2 box)
	bash src/data/zeek_batch.sh

convert:         ## mirror one dataset's Zeek logs to parquet, 1:1, same tree (DATASET=…)
	$(PY) src/data/to_parquet.py \
		--input  s3://$(BUCKET)/extracted/$(DATASET) \
		--output s3://$(BUCKET)/parquets/$(DATASET)

discover:        ## profile the parquet datasets in the lake
	$(PY) src/preprocess/lake_feature_discovery.py
