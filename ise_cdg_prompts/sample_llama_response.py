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


from ise_cdg_prompts.llm_api import LLM_API
from ise_cdg_prompts.llm_api.llama import Llama_API
llm_api: "LLM_API" = Llama_API()

response_list = list(map(llm_api.get_response, prompt_list))

list(map(print, response_list))
