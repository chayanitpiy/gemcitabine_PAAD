# Gemcitabine_PAAD
This code performs stratified 10X10 nested-CV to evaluate optimized predictive models from PAAD paients treated with gemcitabine datasets derived from the TCGA-PAAD project (https://gdc.cancer.gov/).

To run the code examples, python version 3.7.3 is required, which can be downloaded from https://www.python.org/downloads/?fbclid=IwAR1VrxINLd7djn_4UuQTxshpya9D2wmt_H0jnUaJrEXVCoalD3viphyLRyI

Specific python libraries used in this study are listed below and can be installed using python’s in-build pip module by executing the code directly from Linux terminal command line.

Pandas version 1.2.3

Scikit-learn version 0.20.1

Imbalanced-learn version 0.3.3

Dtreeviz version 1.2

Joblib version 1.0.1

Graphviz version 0.16

Matplotlib version 3.3.4

Numpy version 1.20.2

Seaborn version 0.11.1

Code example should be executed directly in the working directory from the Linux terminal command line

Usage: python PYTHON CODE [-h] -i DATASET -c CLASSIFIER [-p HYPERPARAM] [-r NB_RANDOM_SEEDS] [-m METRIC] [-SW] [-log2] [-imp] [-t]

Optional arguments:

-h, --help : Show this help message and exit.

-i DATASET, --dataset: The tabulated pharmaco-omic dataset, where columns specify patients, their corresponding drug response (Responder or Non Responder), and molecular features for the selected profile.

-c CLASSIFIER, --classifier: supervised learning algorithm employed to build the classifiers.

-r NB_RANDOM_SEEDS, --random-seeds: Number of random seeds used to run cross-validation tests (Default = 5).

-m METRIC, --metric: Metric used to select optimized models (Default = MCC). Possibilities are: AUC, MCC, AUCPR, AVGPRE, PRE, REC, F1.

-sw, --sample-weighting: this option applies sample weighting.

-n, --normalize: this option normalizes the dataset prior to training and testing the classifier.

As an example, the following command line will carry out five nested 10-fold cross-validation runs of a RF model with feature selection on the mRNA(FPKM) dataset (provided in this repository):

$ python code_ML_model.py -i mRNA(FPKM)_TCGA_PAAD_gemcitabine.tsv -c PRF -r 5 -m MCC
