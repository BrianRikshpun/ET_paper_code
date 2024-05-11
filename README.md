# Innovative Construction of Evolutionary Taxonomy by Applying Machine Learning to Nucleotides Distributions

## Description
This project investigates the application of machine learning to analyze codon/codon-pair distributions and their relationship to evolutionary taxonomy. Supervised models, including SVM, Decision Trees, Random Forests, and Logistic Regression, were used to classify taxonomic groups, while unsupervised clustering methods like K-Means and Self-Organizing Maps identified similarity clusters. Codon-pair data consistently outperformed individual codons, and the non-linear SVM achieved the best classification results. Novel evaluation metrics, such as the Taxonomy Closeness score, highlighted the discriminative power of codon-pair data for well-separated clusters.

Codon & Bicodon data source - https://dnahive.fda.gov/dna.cgi?cmd=codon_usage&id=537&mode=tisspec <br>
NCBI Taxonomy data source - https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)


## Installation
### Prerequisites
- The code can be run on any Python IDE console (Pycharm,
- Required Python 3.9 or above

### Setting Up
1. Clone the repository
   ```bash
   git clone https://github.com/username/repo-name.git

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   

## Usage

### Data Preparation
Prepare the dataset (Data cleaning and merging with the NCBI taxonomy)
```bash
python DataPrep.py --input o537-genbank_species.tsv --rankedlineage122.csv
```

### Running Supervised Models
Run the supervised models (SVM, Decision Trees, etc.) to classify taxonomic groups using on the data after using DataPrep.py on codon/bicodon features:
```bash
python Classification.py --ready_to_run_codon.csv 
```

### Running Supervised Models
Execute the unsupervised clustering (K-Means, SOM) to identify similarity clusters on the data after using DataPrep.py on codon/bicodon features:
```bash
python Clustering.py --ready_to_run_codon.csv 
```








