import h5py
import numpy as np
import time
# requires the tf_model.h5 file from https://huggingface.co/EleutherAI/gpt-j-6B/tree/main

def test_from_random_direction(weights, d_model, n_directions):
    # creates n_directions random d_model-dimensional vectors
    # for each vector, multiplies it by weights to find which token GPT would output if that vector were the final state of the residual stream
    test_vectors=np.random.normal(loc=0, scale=1, size=(d_model, n_directions))
    logits=np.matmul(weights, test_vectors)
    index_frequencies=counts(np.argmax(logits, axis=0))
    return index_frequencies

def sum_dicts(dict1, dict2):
    # creates a new dict with all the keys in dict1 or dict2, whose entries are the sum of their entries
    joined_dict={}
    joined_keys=set(dict1).union(set(dict2))

    for key in joined_keys:
        if key in dict1:
            value1=dict1[key]
        else:
            value1=0
        if key in dict2:
            value2=dict2[key]
        else:
            value2=0
        joined_dict[key]=value1+value2
    return joined_dict

def counts(arr):
    # creates a dict of how often a term appears in arr
    # ex: [a,b,b] -> {a:1, b:2}
    counts_dict={}
    for x in arr:
        if x not in counts_dict:
            counts_dict[x]=0
        counts_dict[x]+=1
    return counts_dict

if __name__=="__main__":
    path='tf_model.h5'
    random_seed=1
    np.random.seed(random_seed)
    # weights downloaded from https://huggingface.co/EleutherAI/gpt-j-6B/tree/main
    with h5py.File(path, 'r') as f:
        key='/transformer/tfgptj_for_causal_lm/transformer/wte/weight:0'
        weights=f.get(key)[:]
        # weights=np.random.normal(loc=0, scale=1, size=(50400, 4096)) #use this to make "random word embeddings"
        num_tokens=weights.shape[0]
        dimensions=weights.shape[1]

        # experimentation found that it's faster to generate samples in batches of ~600
        batch_size=600
        batches=5000

        cumulative_frequencies={num:0 for num in range(50400)}

        for i in range(batches):
            t_start=time.time()
            batch_frequencies=test_from_random_direction(weights, dimensions, batch_size)
            cumulative_frequencies=sum_dicts(cumulative_frequencies, batch_frequencies)
            t_end=time.time()
            print(f"For batch {i} it took {t_end-t_start:.2f} seconds, or {(t_end-t_start)/batch_size:.2e} seconds per test")

        outfile_prefix="frequency_data/"
        outfile_location="frequencies_random_seed_1.csv"
        with open(f"{outfile_prefix}{outfile_location}", 'w') as f:
            out_string="\n".join([f"{key},{cumulative_frequencies[key]}" for key in sorted(list(cumulative_frequencies))])
            f.write(out_string)
        print("done)")
