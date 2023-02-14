import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


def load_data(file_name, labels_location="vocab.json", file_prefix="frequency_data/"):
    with open(f"{file_prefix}{file_name}", 'r') as frequency_file:
        df = pd.read_csv(frequency_file, names=['index', 'frequency'])
        if labels_location:
            with open(labels_location, 'r') as label_file:
                token_encoding = reverse_dict(json.load(label_file))
                max_tokens = max(list(token_encoding))
                df = df.loc[df['index'] < max_tokens+1]
                token_labels = [token_encoding[n] for n in range(max_tokens+1)]
                df['token'] = token_labels
        return df


def weird_token_indices():
    # from https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation
    weird_tokens = [188, 189, 190, 191, 192, 193, 194, 195, 196, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 221, 3693, 5815, 9364, 12781, 17405, 17629, 17900, 18472, 20126, 21807, 23090, 23282, 23614, 23785, 24200, 24398, 24440, 24934, 25465, 25992, 28666, 29372, 30202, 30208, 30209, 30210, 30211, 30212, 30213, 30897, 30898, 30899, 30905, 30906, 31032, 31576, 31583, 31666, 31708, 31727, 31765, 31886, 31957, 32047, 32437,
                    32509, 33454, 34713, 35207, 35384, 35579, 36130, 36173, 36174, 36481, 36938, 36940, 37082, 37444, 37574, 37579, 37631, 37842, 37858, 38214, 38250, 38370, 39165, 39177, 39253, 39446, 39749, 39752, 39753, 39755, 39756, 39757, 39803, 39811, 39821, 40240, 40241, 40242, 41380, 41383, 41441, 41551, 42066, 42089, 42090, 42202, 42424, 42470, 42586, 42728, 43065, 43177, 43361, 43453, 44686, 45544, 45545, 46600, 47182, 47198, 47571, 48193, 49781, 50009]
    return weird_tokens

def reverse_dict(dict_to_reverse):
    # changes key:value dict to value:key dict
    # make sure the values are distinct in the original dict!
    new_dict = {value: key for key, value in dict_to_reverse.items()}
    if len(dict_to_reverse) > len(new_dict):
        print("When you reversed your dict, you lost some entries!")
    return new_dict


def compute_number_of_zero_frequency(df):
    return len(df[df['frequency']==0])

def df_at_weird_tokens(df):
    weird_tokens=weird_token_indices()
    indexes_as_vector=[idx in weird_tokens for idx in df['index']]
    return df[indexes_as_vector]

def analyze_number_of_zero_tokens():
    file_name = "frequencies_random_seed_1.csv"
    random_data_file_name="frequencies_of_random_vectors.csv"
    df = load_data(file_name)
    df_of_weird_tokens=df_at_weird_tokens(df)
    df_initial_93=df[df['index']<93]
    df_random=load_data(random_data_file_name, labels_location=None)
    print(f"Of the whole {len(df)} tokens, {compute_number_of_zero_frequency(df)} ({compute_number_of_zero_frequency(df)/len(df):.1%}) had zero frequency")
    print(f"Of the {len(df_of_weird_tokens)} weird tokens, {compute_number_of_zero_frequency(df_of_weird_tokens)} ({compute_number_of_zero_frequency(df_of_weird_tokens)/len(df_of_weird_tokens):.1%}) had zero frequency")
    print(f"Of the first {len(df_initial_93)} tokens, {compute_number_of_zero_frequency(df_initial_93)} ({compute_number_of_zero_frequency(df_initial_93)/len(df_initial_93):.1%}) had zero frequency")
    print(f"In a random dataset of {len(df_random)} tokens, {compute_number_of_zero_frequency(df_random)} ({compute_number_of_zero_frequency(df_random)/len(df_random):.1%}) had zero frequency")
    
def analyze_most_frequent_tokens(top_n=10):
    file_name = "frequencies_random_seed_1.csv"
    df = load_data(file_name)
    df=df.sort_values(by="frequency", ascending=False)
    cutoff_threshold=df.iloc[top_n-1]['frequency']
    df_top_n=df[df['frequency']>=cutoff_threshold]
    print("The top {top_n} most frequent tokens are:")
    print(df_top_n)

def plot_frequencies_descending(df, log_scale=False, fig_title="Frequency of Tokens", save_file_name=None):
    # plt.close()
    df=df.sort_values(by="frequency", ascending=False)
    df['x']=list(range(len(df)))
    if log_scale:
        rectified_frequencies=[max(freq, 0.5) for freq in df['frequency']]
        df['frequency']=rectified_frequencies
        plt.yscale('log')

    plt.plot(df['x'], df['frequency'])
    plt.title(fig_title)
    plt.xlabel("Token (descending by frequency)")
    plt.ylabel("Frequency")
    if save_file_name:
        plt.savefig(save_file_name, bbox_inches='tight')

def make_frequency_plots():
    file_name = "frequencies_random_seed_1.csv"
    df = load_data(file_name)
    plt.subplots(2,2)
    plt.tight_layout()
    plt.subplot(221)
    plot_frequencies_descending(df, log_scale=False, fig_title="GPT-J Token Frequencies", save_file_name='')
    plt.subplot(223)
    plot_frequencies_descending(df, log_scale=True, fig_title="", save_file_name='')
    random_data_file_name="frequencies_of_random_vectors.csv"
    random_df = load_data(random_data_file_name)
    plt.subplot(222)
    plot_frequencies_descending(random_df, log_scale=False, fig_title="Random Embeddings Token Frequencies", save_file_name='')
    plt.subplot(224)
    plot_frequencies_descending(random_df, log_scale=True, fig_title="", save_file_name='frequency_plots/four_quadrant_plot.png')

def make_frequency_plots_showing_weird_tokens():
    file_name = "frequencies_random_seed_1.csv"
    df = load_data(file_name)
    plot_frequencies_descending(df, log_scale=False, fig_title="GPT-J Token Frequencies")
    df=df.sort_values(by="frequency", ascending=False)
    df['x']=list(range(len(df)))
    df=df_at_weird_tokens(df)
    plt.scatter(x=df['x'], y=df['frequency'], color='red', marker='x', alpha=0.5, label='Weird Tokens')
    plt.legend()
    save_file_name="frequency_plots/freq_with_weird_tokens"
    plt.savefig(save_file_name, bbox_inches='tight')

def print_zero_frequency_indices():
    file_name = "frequencies_random_seed_1.csv"
    df = load_data(file_name)
    zero_frequency_df=df[df['frequency']==0]
    print("Indexes of tokens with zero frequency:")
    print(zero_frequency_df['index'].values)    

if __name__ == "__main__":
    print("\n")
    analyze_number_of_zero_tokens()
    analyze_most_frequent_tokens()
    make_frequency_plots()
    make_frequency_plots_showing_weird_tokens()
    print_zero_frequency_indices()
    
