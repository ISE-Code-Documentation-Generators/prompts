from ise_cdg_prompts.prompt_generator import generate_prompt_data, Sample, dataset_len
import random
sample_size = 10
shot_size = 4
prompt_list, ground_truth = generate_prompt_data(
    samples = [Sample(
        template_indices=random.sample(range(0, dataset_len()), shot_size), 
        question_index=random.sample(range(0, dataset_len()), 1)) for i in range(sample_size)]
)


print(prompt_list[0])
print(ground_truth[0])


# pip install llama-index-program-openai llama-index-llms-llama-api llama-index
def get_llama(api_key: str):
    from llama_index.llms.llama_api import LlamaAPI
    return LlamaAPI(api_key=api_key)

llama = get_llama(api_key = "LL-U2BMZaeGCgEdNQ56j2UyBNN8bztei5whjdqxLyMCtaMdcaSVCkm70Faq5WYF2KF4")


def get_llama_response(prompt: str):
#     try:
        return llama.complete(prompt).text
#     except:
#         return "fuck"


get_llama_response(prompt="Paul Graham is ")

print(get_llama_response(prompt=prompt_list[0]))

print(prompt_list[0])


def get_fewshot_response(prm222: str) -> str:
    resp = get_llama_response(prm222)
#     resp = resp.replace(prm222, '')
    return resp

response_list = list(map(get_fewshot_response, prompt_list))

list(map(print, response_list))
