BUCKET  = ids-datalakec48eb2cab942494ba5059fac3b3527d9
DATASET ?= cic-ids-2017
TAG     ?=

.PHONY: help extract convert discover
help:            ## show available targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

extract:         ## run Zeek over the raw pcaps (run on the EC2 box)
	bash src/data/zeek_batch.sh

convert:         ## convert one dataset's Zeek logs to parquet (DATASET=… TAG=v2)
	python3 src/data/to_parquet.py \
		--input  s3://$(BUCKET)/extracted/$(DATASET) \
		--output s3://$(BUCKET)/parquets/$(DATASET) \
		$(if $(TAG),--tag $(TAG))

discover:        ## profile the parquet datasets in the lake
	python3 src/preprocess/lake_feature_discovery.py
