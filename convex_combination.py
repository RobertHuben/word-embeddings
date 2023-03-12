import numpy as np
import h5py
import time
import pandas as pd

from analysis import load_data


def generate_self_activation_data():
    # for each token, checks if it is "self-activated"
    # meaning that the direction from the vector centroid to that vector makes this vector the most activated
    # being self-activated means you are not interior (converse may fail)
    path='tf_model.h5'
    with h5py.File(path, 'r') as f:
        key='/transformer/tfgptj_for_causal_lm/transformer/wte/weight:0'
        weights=f.get(key)[:]

        centroid=np.average(weights, axis=0) #center point of all embedding vectors
        directions_from_centroid=weights-centroid #direction from centroid to each token 
        tokens_not_self_activating=[]
        for i in range(weights.shape[0]):
            test_direction=directions_from_centroid[i]
            activations=np.matmul(weights, test_direction)
            most_active_component=np.argmax(activations)
            if (most_active_component ==i):
                pass
                # print(f"Normal: {i}")
            else:
                tokens_not_self_activating.append(i)
                # print(f"Weird: The token {i} most activates the token {most_active_component}.")
            if i%1000==0:
                print(f"Finished checking token {i}, found {len(tokens_not_self_activating)} tokens that fail to self-activate so far")

        with open("tokens_not_self_activating_tokens.txt", 'w') as f_out:
            f_out.write(str(tokens_not_self_activating))
        print("Done!")

def load_self_activation_dataframe(self_activation_file_name="failed_self_activation_tokens.txt", frequency_file_name="frequencies_random_seed_1.csv"):
    # makes a dataframe by reading existing files
    # resulting dataframe has columns token number, frequency, token, "weird", and "fails_self_activation"
    df=load_data(frequency_file_name, file_prefix="frequency_data/", token_names_location="vocab.json", include_weird_flags=True)
    with open(self_activation_file_name, 'r') as f:
        non_activated_tokens=[int(n) for n in f.read()[1:-1].split(",") if int(n)<50257]
    df["fails_self_activation"]=[n in non_activated_tokens for n in df["index"]]
    return df

if __name__=="__main__":
    generate_self_activation_data() 
    df=load_self_activation_dataframe()
    failed_self_activate_df=df.loc[df['fails_self_activation']==1]
    weird_df=df.loc[df['weird']==1]

    pd.set_option('display.max_rows', 200)
    print(weird_df)