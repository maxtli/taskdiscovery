from typing import *
import datasets
from torch.utils.data import DataLoader
from transformer_lens.utils import tokenize_and_concatenate
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def retrieve_owt_data(
    batch_size,
    ctx_length,
    tokenizer,
    split="train",
    from_saved=False,
    ds_name="Elriggs/openwebtext-100k",
):
    dataset = datasets.load_dataset(ds_name, split="train")
    tokens_dataset = tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=ctx_length,
        column_name="text",
        add_bos_token=True,
        num_proc=4,
    )
    tokens_dataset
    data_loader = DataLoader(
        tokens_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return data_loader


def create_fewshot_primer(prompt_data) -> str:
    """Creates the primer string for GPT in-context learning
    
    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information

    Returns:
    prompt: the constructed ICL prompt primer as a string
    """       
    prompt = ''
    prompt += prompt_data['prefixes']['instructions'] + prompt_data['instructions'] + prompt_data['separators']['instructions']
    
    for example in prompt_data['examples']:
        
        prompt += prompt_data['prefixes']['input'] + example['input'] + prompt_data['separators']['input']
        prompt += prompt_data['prefixes']['output'] + example['output'] + prompt_data['separators']['output']
        
    return prompt
    
def create_prompt(prompt_data, sentence=None) -> str:
    """Creates a prompt using the specified sentence for GPT in-context learning
    
    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    sentence: a query string (sentence/word) to include in the ICL prompt

    Returns:
    prompt: the constructed ICL prompt as a string
    """
    if sentence is None and prompt_data['query_target'] is not None:
        sentence = prompt_data['query_target']['input']

    if isinstance(sentence, list):
        sentence = sentence[0]

    prompt_init = create_fewshot_primer(prompt_data)    
    prompt = prompt_init + prompt_data['prefixes']['input'] + sentence + prompt_data['separators']['input']
    prompt += prompt_data['prefixes']['output']
    
    return prompt   

# Partial primer & prompt functions
def create_partial_fewshot_primer(prompt_data, include = np.arange(8)) -> str:
    """Creates the primer string for GPT in-context learning, filtering to include a subset of specified priming strings
    
    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    include: an iterable of ints indicating which examples to include in the ICL prompt
    
    Returns:
    prompt: the constructed ICL prompt primer as a string
    """
    prompt = ''
    prompt += prompt_data['prefixes']['instructions'] + prompt_data['instructions'] + prompt_data['separators']['instructions']

    # Grab each priming example in the specified order.
    for i in include:
        example = prompt_data['examples'][i]
        prompt += prompt_data['prefixes']['input'] + example['input'] + prompt_data['separators']['input']
        prompt += prompt_data['prefixes']['output'] + example['output'] + prompt_data['separators']['output']
        
    return prompt

def create_partial_prompt(prompt_data, sentence=None, include=np.arange(8)) -> str:
    """Creates a prompt using the specified sentence and partial list of in-context primer sentences
    
    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    sentence: a query string (sentence /word) to include in the ICl prompt
    include: an iterable of ints indicating which examples to include in the ICL prompt
    
    Returns:
    prompt: the prompt as a string
    """
    if sentence is None and prompt_data['query_target'] is not None:
        sentence = prompt_data['query_target']['input']
    if isinstance(sentence, list):
        sentence = sentence[0]
        
    prompt_init = create_partial_fewshot_primer(prompt_data, include)
    
    prompt = prompt_init + prompt_data['prefixes']['input'] + sentence + prompt_data['separators']['input']
    prompt += prompt_data['prefixes']['output']
    
    return prompt


# UTILS FOR GENERATING PROMPT META LABELS
def get_prompt_parts_and_labels(prompt_data, query_sentence=None):
    """
    Generates high-level labels for ICL prompts according to its ICL role, such as demonstration, label, separator, structural, etc.
    The JSON prompt format should include 'instructions', 'examples' with ('input', 'output') pairs, 
    'prefixes', and 'separators' for 'input', 'output', and 'instructions'.
    Used in conjunction with tokenize_labels

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    query_sentence: optional (if contained in prompt_data) str containing a query for an ICL prompt

    Returns:
    prompt_parts: structured list of words to be flattened and tokenized
    prompt_part_labels: structured list of labels to be flattened & extended over tokenization
    """
    if query_sentence is None and prompt_data['query_target'] is not None:
        query_sentence = prompt_data['query_target']['input']
    if isinstance(query_sentence, list):
        query_sentence = query_sentence[0]
    n_examples = len(prompt_data['examples'])
    assemble_icl_example = lambda example, prompt_data: [prompt_data['prefixes']['input'], example['input'], prompt_data['separators']['input'], prompt_data['prefixes']['output'], example['output'], prompt_data['separators']['output']]
    assemble_icl_query = lambda query, prompt_data: [prompt_data['prefixes']['input'], query, prompt_data['separators']['input'], prompt_data['prefixes']['output']]

    prompt_instructions = [prompt_data['prefixes']['instructions'], prompt_data['instructions'], prompt_data['separators']['instructions']] 
    prompt_icl_examples = [assemble_icl_example(prompt_data['examples'][i], prompt_data) for i in range(n_examples)]
    prompt_icl_query = [assemble_icl_query(query_sentence, prompt_data)]

    prompt_instructions_labels = ['bos_token', 'instructions_token', 'separator_token']
    prompt_icl_examples_labels = [['structural_token', f'demonstration_{i+1}_token', 'separator_token', 'structural_token', f'demonstration_{i+1}_label_token', 'separator_token'] for i in range(n_examples)]
    prompt_icl_query_labels = [['query_structural_token', 'query_demonstration_token', 'query_separator_token', 'query_structural_token']]

    prompt_parts = prompt_instructions + prompt_icl_examples + prompt_icl_query

    prompt_part_labels = prompt_instructions_labels + prompt_icl_examples_labels + prompt_icl_query_labels

    return prompt_parts, prompt_part_labels

def extend_labels(sentence_parts, text_labels, tokenizer):
    """
    Extends ICL component labels across words that are tokenized into multiple tokens for non-llama-style (sentence-piece) tokenizers

    Parameters:
    sentence_parts: list, where each element is either a token (str), phrase (str), or list of tokens/phrases
    text_labels: list with the same structure as 'sentence_parts', with a corresponding label for that level of the input sentence.
    tokenizer: huggingface tokenizer
    
    Returns:
    final_labels: flattened/extended list of token labels for an ICL prompt (split into parts, contained in sentence_parts and text_labels)
    """    
    prompt_builder = ''
    final_labels = []
    for i,(word,label) in enumerate(zip(sentence_parts, text_labels)):
        
        if isinstance(word, list):
            for j, (word2,label2) in enumerate(zip(word,label)):
                if len(word2) == 0:
                    continue
                pre = tokenizer(prompt_builder, return_length=True).length[0]
                prompt_builder += word2
                post = tokenizer(prompt_builder, return_length=True).length[0]
                n_tokens = tokenizer(word2, return_length=True).length[0]
                actual_tokens = post-pre
                if n_tokens != actual_tokens and n_tokens < actual_tokens:
                    if 'end_of_example' in final_labels[-1]:
                        final_labels.extend(['separator_token']*(actual_tokens - n_tokens))
                    else:
                        final_labels.extend([final_labels[-1]]*(actual_tokens - n_tokens))
                final_labels.extend([label2] * (n_tokens))
                if j==3:
                    final_labels[-1] = final_labels[-1].replace('structural', 'predictive')
                if j==5:
                    final_labels[-n_tokens] = final_labels[-n_tokens].replace('separator', 'end_of_example')
        else:
            if len(word) == 0:
                continue
            pre = tokenizer(prompt_builder, return_length=True).length[0]
            prompt_builder += word
            post = tokenizer(prompt_builder, return_length=True).length[0]
            n_tokens = tokenizer(word, return_length=True).length[0]
            actual_tokens = post-pre
            if n_tokens != actual_tokens and n_tokens < actual_tokens:
                    final_labels.append(final_labels[-1]*(actual_tokens - n_tokens))
            final_labels.extend([label] * (n_tokens))

    return final_labels

def extend_labels_llama(sentence_parts, text_labels, tokenizer):
    """
    Extends ICL component labels across words that are tokenized into multiple tokens for llama-style (sentence-piece) tokenizers

    Parameters:
    sentence_parts: list, where each element is either a token (str), phrase (str), or list of tokens/phrases
    text_labels: list with the same structure as 'sentence_parts', with a corresponding label for that level of the input sentence.
    tokenizer: huggingface tokenizer
    
    Returns:
    final_labels: flattened/extended list of token labels for an ICL prompt (split into parts, contained in sentence_parts and text_labels)
    """
    prompt_builder = ''
    final_labels = ['bos_token']
    for i,(word,label) in enumerate(zip(sentence_parts, text_labels)):
        
        if isinstance(word, list):
            for j, (word2,label2) in enumerate(zip(word,label)):
                if len(word2) == 0:
                    continue
                pre = tokenizer(prompt_builder, return_length=True).length
                prompt_builder += word2
                post = tokenizer(prompt_builder, return_length=True).length
                if word2.startswith(' '):  
                    n_tokens = len(tokenizer.tokenize(word2.replace(" ","",1)))
                else:
                    n_tokens = tokenizer(word2, return_length=True).length -1
                actual_tokens = post-pre
                if n_tokens != actual_tokens:
                    if n_tokens < actual_tokens:
                        if prompt_builder.startswith(' '):
                            final_labels.append(label2)
                        else:
                            if 'end_of_example' in final_labels[-1]:
                                final_labels.extend(['separator_token']*(actual_tokens - n_tokens))
                            else:
                                final_labels.extend([final_labels[-1]]*(actual_tokens - n_tokens))
                    elif n_tokens > actual_tokens: 
                        n_tokens = min(actual_tokens, n_tokens)
                
                final_labels.extend([label2] * (n_tokens))
                if j==3:
                    final_labels[-1] = final_labels[-1].replace('structural', 'predictive')
                if j==5:
                    final_labels[-n_tokens] = final_labels[-n_tokens].replace('separator', 'end_of_example')
                
        else:
            if len(word) == 0:
                continue
            pre = tokenizer(prompt_builder, return_length=True).length
            prompt_builder += word
            post = tokenizer(prompt_builder, return_length=True).length
            n_tokens = tokenizer(word, return_length=True).length -1
            actual_tokens = post-pre
            if n_tokens != actual_tokens and n_tokens < actual_tokens:
                    final_labels.append(final_labels[-1]*(actual_tokens - n_tokens))
            final_labels.extend([label] * (n_tokens))

    return final_labels

def tokenize_labels(sentence_parts, text_labels, tokenizer):
    """
    Extends phrase-level labels across tokenization for in-context learning prompts. Tested with GPT-2's tokenizer from huggingface.
    Parameters:
    sentence_parts: list, where each element is either a token (str), phrase (str), or list of tokens/phrases
    text_labels: list with the same structure as 'sentence_parts', with a corresponding label for that level of the input sentence.
    tokenizer: huggingface tokenizer
    
    Returns: 
    labels: flattened/extended list of token labels for an ICL prompt (split into parts, contained in sentence_parts and text_labels)

    based on the tokenize_and_preserve_labels function from:
    https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
    """
    
    is_llama = 'llama' in tokenizer.name_or_path

    if is_llama:
        labels = extend_labels_llama(sentence_parts, text_labels, tokenizer)
    else:
        labels = extend_labels(sentence_parts, text_labels, tokenizer)

    return labels

def get_token_meta_labels(prompt_data, tokenizer, query=None):
    """
    Computes the ICL meta-labels for every token in a prompt.
    
    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    tokenizer: huggingface tokenizer
    query: str of the query input

    Return:
    token_labels: list of tuples (prompt token index, token, label)  
    prompt_string: full prompt as a string
    """
    if query is None and prompt_data['query_target'] is not None:
        query = prompt_data['query_target']['input']
    if isinstance(query, list):
        query = query[0]
        
    prompt_parts, prompt_part_labels = get_prompt_parts_and_labels(prompt_data, query_sentence=query)
    token_meta_labels = tokenize_labels(prompt_parts, prompt_part_labels, tokenizer)
    prompt_string = create_prompt(prompt_data=prompt_data, sentence=query)
    tokens = [tokenizer.decode(x) for x in tokenizer(prompt_string).input_ids]
    token_labels = list(zip(np.arange(len(tokens)), tokens, token_meta_labels))

    return token_labels, prompt_string

def get_dummy_token_labels(n_icl_examples, tokenizer, prefixes=None, separators=None):
    """
    Computes the ground-truth meta labels & indices for an ICL prompt with the specified number of example pairs
    These GT labels assume each word gets a single token

    Parameters:
    n_icl_examples: number of ICL example pairs
    tokenizer: huggingface tokenizer
    prefixes: ICL template prefixes
    separators: ICL template separators

    Return:
    final_token_labels: list of tuples containing a token's index and label name [(int, str), ... ]
    """
    is_llama = 'llama' in tokenizer.name_or_path
    prepend_bos = not is_llama
    if prefixes is not None and separators is not None:
        dummy_prompt_data = word_pairs_to_prompt_data({'input': ['a']*n_icl_examples, 'output':['a']*n_icl_examples}, 
                                                    query_target_pair={'input':['a'], 'output':['a']}, prepend_bos_token=prepend_bos,
                                                    prefixes=prefixes, separators=separators)
    else:
        dummy_prompt_data = word_pairs_to_prompt_data({'input': ['a']*n_icl_examples, 'output':['a']*n_icl_examples}, 
                                                  query_target_pair={'input':['a'], 'output':['a']}, prepend_bos_token=prepend_bos)
    final_token_labels, _ = get_token_meta_labels(dummy_prompt_data,tokenizer)
    final_token_labels = [(x[0],x[-1]) for x in final_token_labels]
    return final_token_labels

def compute_duplicated_labels(token_labels, gt_labels):
    """
    Computes a map between duplicated labels and ground truth label positions for localized averaging

    Parameters:
    token_labels: token labels of actual prompt being used
    gt_labels: token labels for a "ground truth" prompt that assumes each input & output is a single token

    Returns:
    index_map: a dict mapping prompt label indices to ground truth label indices
    dup_label_ranges: indices where labels should be duplicated
    """
    check_inds = list(filter(lambda x: 'demo' in x[2], token_labels))
    dup_ranges = pd.DataFrame(check_inds).groupby(2)[0].aggregate(lambda x: (x.min(), x.max()))
    dup_labels = [v for v,x in dup_ranges.items() if (x[1] - x[0]) > 0]

    dup_label_ranges = dup_ranges[dup_labels].to_dict()
    dup_inds = pd.DataFrame(check_inds)[pd.DataFrame(check_inds)[2].duplicated()][0].values

    index_map = {k:v[0] for (k,v) in zip([x[0] for x in token_labels if x[0] not in dup_inds], gt_labels)}

    return index_map, dup_label_ranges

def update_idx_map(idx_map, idx_avg) -> dict:
    """
    Updates the idx_map to map duplicate tokens to its gt token position    
    """
    update_map = {}
    for (i,j) in idx_avg.values():
        for k in range(i,j+1):
            if k not in idx_map.keys():
                update_map[k] = idx_map[i]

    update_map = {**idx_map, **update_map} 
    return update_map


def word_pairs_to_prompt_data(word_pairs : dict,
                              instructions: str = "",
                              prefixes: dict = {"input":"Q:", "output":"A:","instructions":""},
                              separators: dict = {"input":"\n", "output":"\n\n", "instructions":""},
                              query_target_pair: dict = None, prepend_bos_token=False,
                              shuffle_labels=False, prepend_space=True) -> dict:
    """Takes a dataset of word pairs, and constructs a prompt_data dict with additional information to construct an ICL prompt.
    Parameters:
    word_pairs: dict of the form {'word1':['a', 'b', ...], 'word2':['c', 'd', ...]}
    instructions: prefix instructions for an ICL prompt
    prefixes: dict of ICL prefixes that are prepended to inputs, outputs and instructions
    separators: dict of ICL separators that are appended to inputs, outputs and instructions
    query_target_pair: dict with a single input-output pair acting as the query for the prompt
    prepend_bos_token: whether or not to prepend a BOS token to the prompt
    shuffle_labels: whether to shuffle the ICL labels
    prepend_space: whether to prepend a space to every input and output token

    Returns: 
    prompt_data: dict containing ICL prompt examples, and template information
    """
    prompt_data = {}
    prompt_data['instructions'] = instructions
    prompt_data['separators'] = separators
    if prepend_bos_token:
        prefixes = {k:(v if k !='instructions' else '<|endoftext|>' + v) for (k,v) in prefixes.items()}
    prompt_data['prefixes'] = prefixes

    if query_target_pair is not None:
        query_target_pair = {k:(v[0] if isinstance(v, list) else v) for k,v in query_target_pair.items()}
    prompt_data['query_target'] = query_target_pair
        
    if shuffle_labels:
        randomized_pairs = [np.random.permutation(x).tolist() if i==1 else x for (i,x) in enumerate(list(word_pairs.values()))] # shuffle labels only
        if prepend_space:
            prompt_data['examples'] = [{'input':' ' + w1, 'output':' ' + w2} for (w1,w2) in list(zip(*randomized_pairs))]
            prompt_data['query_target'] = {k:' ' + v for k,v in query_target_pair.items()} if query_target_pair is not None else None
        else:
            prompt_data['examples'] = [{'input':w1, 'output':w2} for (w1,w2) in list(zip(*randomized_pairs))]
    else:    
        if prepend_space:
            prompt_data['examples'] = [{'input':' ' + w1, 'output':' ' + str(w2)} for (w1,w2) in list(zip(*word_pairs.values()))]
            prompt_data['query_target'] = {k:' ' + str(v) for k,v in query_target_pair.items()} if query_target_pair is not None else None
        else:
            prompt_data['examples'] = [{'input':w1, 'output':w2} for (w1,w2) in list(zip(*word_pairs.values()))]
    
    return prompt_data


# DATASET UTILS
class ICLDataset:
    """
    A simple dataset class containing input-output pairs, used for ICL prompt construction.
    """
    def __init__(self, dataset):    
        if isinstance(dataset, str):
            self.raw_data = pd.read_json(dataset)
        elif isinstance(dataset, dict):
            self.raw_data = pd.DataFrame(dataset)
        self.raw_data = self.raw_data[['input', 'output']]

    def __getitem__(self,i):       
        if isinstance(i, int):
            return self.raw_data.iloc[i].to_dict()
        elif isinstance(i, slice):
            return self.raw_data.iloc[i].to_dict(orient='list')
        elif isinstance(i, list) or isinstance(i, np.ndarray):            
            return self.raw_data.iloc[i].to_dict(orient='list')
        elif isinstance(i, str):
            if i not in self.raw_data.columns:
                raise KeyError(f"Column '{i}' not in the dataset. Current columns in the dataset: {self.raw_data.columns.to_list()}")
            else:
                return self.raw_data[i].to_list()
        else:
            raise ValueError(f"{i} is not a valid index type. Expected one of: [int, list, np.ndarray, slice, str]")

    def __len__(self):
        return len(self.raw_data)
    
    def __repr__(self):
        s = "ICLDataset" + "({\n\tfeatures: " + f"{self.raw_data.columns.to_list()},\n\tnum_rows: {self.__len__()}" + "\n})"
        return s
    
def split_icl_dataset(dataset, train_size=None, test_size=0.3, seed=42) -> Dict[str,ICLDataset]:
    """
    Uses scikit-learn's train_test split to create train, valid, test dataset from provided dataset.

    Parameters:
    dataset: ICL dataset
    train_size: percentage of data (float between 0 and 1) to put in the training data split
    test_size: percentage of data (float between 0 and 1) to put into the test data split
    seed: seed used for splitting the data

    Returns:
    dict containing train, valid, test ICL datasets
    """
    if train_size is None and test_size is None:
        train_size = 0.7
        test_size = 0.3

    elif train_size is not None and test_size is None:
        test_size = 1-train_size

    elif train_size is None and test_size is not None:
        train_size = 1-test_size
    
    elif train_size is not None and test_size is not None:
        assert train_size + test_size == 1
    
    train, valid = train_test_split(dataset.raw_data, test_size=test_size, random_state=seed)
    test, valid = train_test_split(valid, test_size=test_size, random_state=seed)

    train = ICLDataset(train.to_dict(orient='list'))
    valid = ICLDataset(valid.to_dict(orient='list'))
    test = ICLDataset(test.to_dict(orient='list'))

    return {'train':train, 'valid':valid, 'test':test}


def load_ICL_dataset(task_name: str,
                 root_data_dir: str = 'function_vectors_data',
                 test_size = 0.3, 
                 seed=32
                ) -> Dict[str,ICLDataset]:
    """
    Loads a dataset with input/output pairs

    Parameters:
    task_name: the name of the task dataset
    root_data_dir: the root directory where the data comes from
    test_size: fraction used in train/test split
    
    Return:
    dataset: the dict contain the train/valid/test dataset splits
    """

    data_folders = ['abstractive', 'extractive']
    assert test_size <= 1.0

    path = Path(root_data_dir)
    d_group_map = [(dataset_type, os.path.exists(os.path.join(root_data_dir, dataset_type, task_name+'.json'))) for dataset_type in data_folders]

    d_group = list(filter(lambda x: x[1], d_group_map))

    assert len(d_group) !=0 and len(d_group) == 1, f"Error! 'task_name'={task_name}.json must be uniquely contained in one of these directories:{data_folders}. Please check the root_data_dir"
    dataset_folder = d_group[0][0]
    
    d_path = os.path.join(path, dataset_folder, f'{task_name}.json')
    
    dataset = ICLDataset(d_path)
    dataset = split_icl_dataset(dataset, test_size=test_size, seed=seed)

    return dataset

class FVDataset(Dataset):
    def __init__(self, data_array, tasks, labels):
        self.data_array = data_array
        self.tasks = tasks
        self.labels = labels

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        data = self.data_array[idx]
        input_ids = data['input_ids'].squeeze()
        attention_mask = data['attention_mask'].squeeze()

        # Convert input_ids and attention_mask to torch tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return input_ids, attention_mask, self.tasks[idx], self.labels[idx]