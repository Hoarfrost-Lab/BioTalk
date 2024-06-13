
# Data Processing

## Overview

This repository contains notebooks for downloading and processing Uniprot data. The process involves downloading raw data files, processing them, and splitting datasets for further use.

## Download Raw Data

To download and unzip the required data files, use the following commands:
## Download and unzip ID mapping file
wget -b https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/idmapping_selected.tab.gz
gunzip -c idmapping_selected.tab.gz > idmapping_selected.tab &

## Download Uniprot Trembl
wget -b https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.tab.gz
gunzip -c uniprot_trembl.tab.gz > uniprot_trembl.tab &

## Download Uniprot Swissprot
wget -b https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.tab.gz
gunzip -c uniprot_swissprot.tab.gz > uniprot_swissprot.tab &

## Intructions
Generate dataset by following the functions in DatasetGeneration.ipynb. The final files generated will contain following columns: AC, EC, OC, UniRef100, UniRef90, UniRef50, EmblCdsId, Sequence
Split the dataset and create four benchmark datasets by following the functions in DatasetSplitting.ipynb



