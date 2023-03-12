from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

token_subdirectory="tokens/"
tokenizer.save_vocabulary(token_subdirectory)
x=tokenizer("Hello world")['input_ids']
print(x)