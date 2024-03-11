from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import string
from tqdm import tqdm
import pdb, os
import torch
import time
import argparse
import json
import warnings
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
parser.add_argument('--batch_size', type=int, default=2)
args = parser.parse_args()

NAME_MODEL = args.model
TOKENIZER_MAX_SEQ_LEN = 2048

BATCH_SIZE = args.batch_size  # Could be much bigger depending on your model and your GPU. To be tuned for speed performance

MAX_NEW_TOKENS = 256  # Increase for a task with long output

device = "cuda:0" if torch.cuda.is_available() else "cpu"

##Initializing in_context examples
prompt_generator = PromptGenerator(args.num_examples)


## TODO: replace qa_gender dataset in above line with the pormpts csv file given to your group
qa_gender_ds = load_dataset("vipulgupta/CALM", "qa_gender", split='test', trust_remote_code=True)


## TODO: Create tokenizer user hugingface AutoTokenizer module
tokenizer =



model = AutoModelForCausalLM.from_pretrained(NAME_MODEL, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained(NAME_MODEL, torch_dtype=torch.float16, trust_remote_code=True)

def save_result(dataset, out_file, batch_size) :
    all_generated_answers = []
    idx_batches = [list(range(idx, min(idx + batch_size, len(dataset)))) for idx in range(0, len(dataset), batch_size)]
    for idx_batch in tqdm(idx_batches):
        batch = dataset.select(idx_batch)
        ## TODO : convert your inputs into prompts. Note for this assignment all models are instruction finetnued and no additional information needs to be added to prompts
        texts = []

        ## TODO: use tokenizer defined on line 15 to create prompts into tokens and .
        tokens =

        ## This is to make sure you do not face device mismatch error. You can comment below line if you have shofted tokens to device in previous line
        tokens = tokens.to(device)

        ## TODO: use model to generate outputs for the prompts. You can use model.generate function
        generated_tokens_org =

        generated_tokens = generated_tokens_org[:, tokens.input_ids.shape[1]:]  # We truncate the original prompts from the generated texts

        ## TODO: convert generated tokens to text using tokenizer
        generated_texts =
        #print ('output : ', generated_texts)
        all_generated_answers.append(generated_texts)

    ## save list in text file
    ##TODO : change the saving format according to your preferred file format
    with open(out_file, 'w+') as f:
        json.dump(all_generated_answers,f,indent=4)


if not os.path.exists('results'):
    os.makedir('results')

file_name = args.model.split('/')[1]

save_result(qa_gender_ds,'results/' + file_name + '-results.json', int(BATCH_SIZE))
