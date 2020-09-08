# Named Entity Recognition (PyTorch)

- NER task is to tag sequence of tokens to a sequence of labels.
    - Labels are 
        1. ORG - Organization
        2. Per - Person 
        3. O - Other
        4. Loc - Locations

- Dataset files can be found in `Data/*.tsv` tab separated files.
- To Install repository requirements `pip install -r requirements.txt`
- To use the repo
    - to configure models hyper-parameters, edit values in file `models/hyperparameters.py`
    - `main.py` is used to train Baseline model
    - `main_crf.py` is used to train CRF model

- Quick Tour:
    - `data_loader/TSVDatasetParser.py` is the file containing the main data parser which:
        - starts by reading tsv file, to create 2 lists data_x, data_y (all data_x tokens are lowercase)
        - Create vocabulary from unique tokens, to create `word2idx` dict. Same applies for labels `label2idx` dict
        - Converts data from tokens to indices to be NN friendly `encode_dataset(...)`
    - `models/`
        - `hyperparameters.py` it holds the model configuration values
        - `models.py` contains BaseLine Model, CRF Model Architectures
            - each model has its own functions `forward(...)`, `save(...)`, `load(...)`, `predict_sentences(...)`
    
    - `training/`
        - `train.py` contain trainer objects for both baseline and CRF models
    
    - other folders names implies their functional