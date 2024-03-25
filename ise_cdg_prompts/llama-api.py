#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
# Load dataset containing markdown and code cells
dataset_path = "/kaggle/input/cnn-rnn-final-experiment/final_dataset.csv"
dataset = pd.read_csv(dataset_path)
def dataset_get_markdown(index: int):
    return str(dataset.iloc[index]['markdown'])
def dataset_get_source(index: int):
    return str(dataset.iloc[index]['source'])


# In[83]:


class Pipeline:
    def __init__(self, l):
        self.l = l
    
    def to_map(self, f):
        return Pipeline(list(map(f, self.l)))
    
    def to_reduce(self, f, initial=None):
        from functools import reduce
        return reduce(f, self.l, initial)
    
    def to_list(self):
        return self.l


# In[144]:


class Template:
    def __init__(self, index: int, row_index: int):
        self.index = index
        self.row_index = row_index
        
    def representative_index(template):
        return str(template.index + 1)
        
    def generate_prompt(template):
        return (f"Start Markdown {template.representative_index()}: {dataset_get_markdown(template.row_index)}\n"
                + f"Start Code {template.representative_index()}: {dataset_get_source(template.row_index)}\n"
                )


class Sample:
    def __init__(self, template_indices, question_index):
        self.templates = (Pipeline(range(len(template_indices)))
                          .to_map(lambda template_index: Template(template_index, template_indices[template_index]))
                          .to_list())
        self.question_index = question_index
        
    def generate_prompt(sample):
        return ("For now, Just read these template Markdown and Code pairs. \n"
                + "".join(Pipeline(sample.templates)
                           .to_map(lambda template: template.generate_prompt())
                           .to_list())
                + f"\n Then, Generate markdown for the below code according to the pairs.\n Code: {dataset_get_source(sample.question_index)}"
               )
    
    def get_ground_truth(sample):
        return dataset_get_markdown(sample.question_index)

def generate_prompt_data(samples):
    prompt_list = Pipeline(samples).to_map(lambda sample: sample.generate_prompt()).to_list()
    grund_truth = Pipeline(samples).to_map(lambda sample: sample.get_ground_truth()).to_list()
    return prompt_list, grund_truth


# In[145]:


import random
sample_size = 10
shot_size = 4
prompt_list, ground_truth = generate_prompt_data(
    samples = [Sample(
        template_indices=random.sample(range(0, dataset.shape[0]), shot_size), 
        question_index=random.sample(range(0, dataset.shape[0]), 1)) for i in range(sample_size)]
)


# In[146]:


print(prompt_list[0])


# In[91]:


print(ground_truth[0])


# In[4]:


get_ipython().run_line_magic('pip', 'install llama-index-program-openai')
get_ipython().run_line_magic('pip', 'install llama-index-llms-llama-api')
get_ipython().system('pip install llama-index')

def get_llama(api_key: str):
    from llama_index.llms.llama_api import LlamaAPI
    return LlamaAPI(api_key=api_key)

llama = get_llama(api_key = "LL-U2BMZaeGCgEdNQ56j2UyBNN8bztei5whjdqxLyMCtaMdcaSVCkm70Faq5WYF2KF4")


# In[101]:


def get_llama_response(prompt: str):
#     try:
        return llama.complete(prompt).text
#     except:
#         return "fuck"


# In[6]:


get_llama_response(prompt="Paul Graham is ")


# In[102]:


print(get_llama_response(prompt=prompt_list[0]))


# In[96]:


print(prompt_list[0])


# In[52]:


def get_fewshot_response(prm222: str) -> str:
    resp = get_llama_response(prm222)
#     resp = resp.replace(prm222, '')
    return resp

response_list = list(map(get_fewshot_response, prompt_list))


# In[53]:


list(map(print, response_list))


# In[ ]:




