

# Download the traning data from
http://www.statmt.org/europarl/v10/training/europarl-v10.de-en.tsv.gz 


Install the following
pip install sentencepiece torch datasets tqdm

python ./prepare_data.py            # Run it once to get the data ready
./tokenizer/train_tokenizer.py 

# Run it multiple times to train the model
python ./train.py
