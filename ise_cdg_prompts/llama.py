
# In[145]:

from prompt_generator import generate_prompt_data, Sample, dataset
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


# get_ipython().run_line_magic('pip', 'install llama-index-program-openai')
# get_ipython().run_line_magic('pip', 'install llama-index-llms-llama-api')
# get_ipython().system('pip install llama-index')

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
