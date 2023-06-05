from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# Initialize model
remote_name = "tiiuae/falcon-7b"
local_name = "../falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(local_name)
model = AutoModelForCausalLM.from_pretrained(local_name, trust_remote_code=True)
print('Model Initialized')

# Tokenize input
# batch_sentences = [
#     "But what about second breakfast?",
#     "Don't think he knows about second breakfast, Pip.",
#     "What about elevensies?",
# ]
# encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")

# TODO: in padding/truncation needed? or maybe only for finetuning
input_text = "The last time I went to jail"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
print('Input Encoded')

# input_text = "Once upon a time"
# input_ids = tokenizer.encode(input_text, return_tensors='pt')

attention_mask = torch.ones(input_ids.shape)

output = model.generate(
    input_ids, 
    attention_mask=attention_mask, 
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
print('Output Generated')

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
