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
After a previous attempt on creating an IDS, I realised that the model of an IDS needs to have the ability to detect attacks across domains. It is trivial fitting xgboost to a single dataset, but then applying that model in the real world will lead to collapse. 

For this reason, a model has to have a diverse and rich source of data to train on - which has lead me to gather as 

## Architecture
<!-- link to docs/architecture.md + the high-level diagram -->
### Data Layer
### Schema — the Contract
### ML Pipeline
### Continual Learning
### Evaluation

## Repository Structure
<!-- the directory tree + one line per component -->

## Data Sources
### Public Datasets
### netemV2 Emulator
### Honeypot

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
<!-- current phase; deferred: live engine, prevention -->

## References
<!-- datasets, Zeek, prior work -->

## License
