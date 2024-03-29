# word-embeddings

Package contents:
- frequency_data folder: CSVs that are the result of frequency_experiment.py. Format is "[token index], [number of times token was generated]"
    - frequencies_of_random_vectors.csv: Result of using random token embeddings instead of GPT-J's token embeddings (unsaved random seed)
    - frequencies_old.csv: My first run of the frequencies_experiment.py (unsaved random seed) (WARNING: this is formatted differently, and in particular tokens that were never generated are not in the csv)
    - frequencies_random_seed_1.csv: Results of frequency_experiment.py with the random seed 1.
- frequency_plots folder: Analysis plots are saved here
- README.md: The README
- analysis.py: Calls several methods of analysis on data once it has been generated by frequency_experiment.py
- vocab.json: GPT's vocabulary (the mapping of tokens to indexes), from https://huggingface.co/EleutherAI/gpt-j-6B/tree/main

Generating Frequency Data:
1. Download the GPT-J parameters. They are in the file 'tf_model.h5' at https://huggingface.co/EleutherAI/gpt-j-6B/tree/main. 
1. Run the main method of frequency_experiment.py. Optionally, change its random seed and the outfile name.
1. If you wish to generate frequencies from random token embeddings instead of GPT-J's embeddings, use the  weights=np.random.normal line

Analyzing Frequency Data:
1. Run analysis.py.
1. If you are analyzing differently named files, you may have to change the file_name lines.

