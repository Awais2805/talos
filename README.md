# Talos
A research project on cross-domain NIDS's (Network Intrusion Detection System's)

## Overview
Talos is comprised of two main components: data and the machine learning pipeline. The data is sourced from 3 main sources - which are comprised of:
- Captured flows from the network emulator
- Captured flows from live honeypot
- Flows from publicly available datasets (such as CIC-IDS)

All captured flows are then extracted using Zeek. My previous attempt on IDS used CICFlowMeter, but I find that Zeek extracts richer features from pcap files. 

The extracted flows are then fed into the ML pipeline. 

ML pipline flow:
1. Map 
2. Merge
3. Clean
4. Encode
5. Split
6. Train

Models are then saved into a registry and undergo evaluation. 

Strong models are used in continual learning loops - where models are fed live packet feeds and infer on them (in training capacity only). 



## Motivation & Goals
### The Detection Ladder (the three heads)
<!-- known/known · known/unknown-env · unknown/unknown -->
After a previous attempt on creating an IDS, I realised that the model of an IDS needs to have the ability to detect attacks across domains. It is trivial fitting xgboost to a single dataset, but then applying that same model in the real world will lead to collapse. 

In the scope of this project, a good ids would perform well in 3 main areas: 1. Detecting familiar attacks in familiar enviroments. 2. Detecting familiar attacks in novel enviroments. 3. Detecting anomlic/novel attacks in familiar enviroments. 4. Detecting anomlic behaviour in novel enviroments (layer 2 NN)

For this reason we have to make the model resitant to domain shift - which is where the data sources play a key component. If we train the model across multiple domains, there is a higher chance of it being able to somewhat perform well with new domains (ips-xgboost repo). Though this can help the model adapt to new domains - there is no guarantee that for any new enviroment the model will be able to adapt effectively. 

One effective method to reduce domain shift is to train the model on captured network traffic from the enviroment it is about to be deloyed in. This would then increase the chances of the model performing well, as it has some famililarity with the domain - coupled with the fact that the model should already (best-case) be resistant to domain shift. 


## Architecture
<!-- link to docs/architecture.md + the high-level diagram -->
### Data Layer


### Schema — the Contract
- tdb

### ML Pipeline
**IN-PROGRESS**

### Continual Learning
**POST-PIPELINE**

### Evaluation
**Methods** 
- Cross evaluation...

## Repository Structure
.
├── data
│   ├── adapters
│   ├── emulation
│   │   └── requirements.txt
│   └── honeypot
├── docs
├── eval
├── extraction
├── infra
├── model
├── README.md
├── schema
│   └── requirements.txt
└── training
    └── requirements.txt

12 directories, 4 files


## Data Sources
### Public Datasets
- CIC-IDS 2017 + 2018
- CIC-DDoS 2019 
...
### netemV2 Emulator
**INCOMPLETE**
### Honeypot
**INCOMPLETE**


## Schema
<!-- pointer to schema/, canonical feature contract, versioning -->

## Getting Started
### Prerequisites
### Installation
<!-- per-component: emulation (EC2) vs training (ICF) -->

## Usage
### Feature Extraction (Zeek → canonical)
### Training Pipeline
### Evaluation & Cross-Env Matrix

## Infrastructure
### Compute (EC2 / ICF)
### Storage

## Results & Findings
<!-- cross-dataset transfer results, the realism-canary outcomes -->

## Roadmap & Status
### Status
- Setting up






## References
<!-- datasets, Zeek, prior work -->

## License