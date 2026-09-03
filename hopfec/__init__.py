"""hopfec: whole-brain effective connectivity with Hopf whole-brain models.

Sub-packages
------------
config      YAML configuration with portable (env-var / relative) paths
bids        BIDS dataset + participants.tsv helpers
atlases     catalogue of parcellation atlases (NIfTI + label tables)
preprocess  fMRIPrep launcher (container) and BOLD post-processing/parcellation
inputs      discovery of fMRIPrep and HALFpipe derivatives
sc          structural connectivity: QSIPrep/QSIRecon, tractography, normative dTOR-985
models      linear and non-linear Hopf models, GEC fitting, parameter search
pipeline    participant- and group-level effective-connectivity workflow
"""
__version__ = "0.1.0"
