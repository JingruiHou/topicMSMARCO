# Topic MSMARCO Dataset

The Topic MSMARCO dataset is a specialized subset of the original MSMARCO Passage Ranking Dataset, created to evaluate the continual lifelong learning performance in neural information retrieval systems. It is designed to facilitate research into how neural IR systems can adapt to new information over time without forgetting previously learned knowledge.

## Publication

This dataset was introduced in the paper titled "Advancing continual lifelong learning in neural information retrieval: definition, dataset, framework, and empirical evaluation" available at [arXiv](https://arxiv.org/abs/2308.08378).

## Dataset Description

The Topic MSMARCO dataset clusters the training set of the MSMARCO dataset into distinct topics using a combination of Word2Vec embeddings and KMeans clustering. This approach was chosen for its interpretability over methods like BERTtopic. After cleaning and preprocessing the text, embeddings were trained and clustering experiments were conducted. Optimal clusters were selected based on perplexity evaluations, with noisy clusters being removed and similar thematic clusters merged.

The dataset comprises six thematic categories:
- IT
- Furnishing
- Food
- Health
- Tourism
- Finance

Semantic distances between these subsets are computed to explore the relationship between topic distance and continual learning effectiveness.

## Reproducing the Dataset

Due to the large size of the dataset, it is not feasible to host it fully online. However, we provide index files and the methodology to reconstruct the dataset locally.

### Prerequisites

1. **Download the MSMARCO Dataset**: Go to the [MSMARCO official website](https://microsoft.github.io/msmarco/Datasets) and download:
   - `collection.tar.gz`
   - `queries.tar.gz`

2. **Download Index Files**: Download the `TopicMSMARCO_index_files.7z` from [Google Drive](https://drive.google.com/file/d/1K3gP_KcFVbBuIvPHbqL8yRZ459q7jT6W/view?usp=sharing).

### Setup

1. **Pre-Tokenization**: To facilitate efficient data handling, especially for models testing backward and forward transfer abilities, pre-tokenize queries and documents.
   - For traditional IR models like DRMM, KRNM, ConvKNRM, Duet, use [Glove](https://nlp.stanford.edu/projects/glove/) for dictionary and pre-trained vectors.
   - For advanced pretrained models like ColBERT, use the respective tokenizer.
   - For models requiring local feature inputs like DUET, pre-store the paired query-document local representations.

2. **Dataset Construction**: Refer to the `TopicMSMARCO_ir_data_building.py` script in this repository for the process of constructing training sets, test sets, and local representations.

## Usage

For more details on using this dataset, refer to the CLNIR repository at [GitHub CLNIR](https://github.com/JingruiHou/CLNIR/tree/main).

## Citation

If you use the Topic MSMARCO dataset or refer to our work in your research, please cite us using the following BibTeX entry:

```bibtex
@article{hou2023advancing,
  title={Advancing continual lifelong learning in neural information retrieval: definition, dataset, framework, and empirical evaluation},
  author={Hou, Jingrui and Cosma, Georgina and Finke, Axel},
  journal={arXiv preprint arXiv:2308.08378},
  year={2023},
  url={https://arxiv.org/abs/2308.08378}
}
