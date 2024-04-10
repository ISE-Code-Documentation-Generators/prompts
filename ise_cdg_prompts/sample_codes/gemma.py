# get_ipython().system("pip install pandas")
# get_ipython().system('pip install -U "transformers==4.38.1" --upgrade')
# get_ipython().system("pip install accelerate")
# get_ipython().system("pip install -i https://pypi.org/simple/ bitsandbytes")


from typing import List
import pandas as pd
import random

from ise_cdg_prompts.dataset import CodeMarkdown, PromptDataset
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor
from ise_cdg_prompts.task import Task
from ise_cdg_prompts.utils.pipeline import Pipeline


batch_size = 10
shot_size = 4


class AlirezaDataset(PromptDataset):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.df: "pd.DataFrame" = pd.read_csv(path)
        self.df.dropna(subset=["source", "markdown"], inplace=True)

    def __getitem__(self, index) -> "CodeMarkdown":
        return CodeMarkdown(
            str(self.df.loc[index]["source"]),
            str(self.df.loc[index]["markdown"]),
        )

    def __len__(self) -> int:
        return self.df.shape[0]


# Load dataset containing markdown and code cells
dataset = AlirezaDataset(path="./final_dataset.csv")


class AlirezaPromptGenerationVisitor(PromptGenerationVisitor):
    def visit_task(self, task: "Task") -> str:
        return (
            self.__visit_templates(templates=task.templates)
            + "\nGenerate markdown for the bottom code according to the four samples above\n Code: "
            + task.get_ground_truth()
        )

    def __visit_template(self, code_markdown: "CodeMarkdown", index):
        result = "Start Markdown " + str(index) + ": " + code_markdown.markdown + "\n"
        result = result + "Start Code " + str(index) + ": " + code_markdown.code + "\n"
        return result


    def __visit_templates(self, templates: List[CodeMarkdown]):
        prompt = ""
        for index, template in enumerate(templates):
            prompt = prompt + self.__visit_template(
                code_markdown=template,
                index=index + 1,
            )
        return prompt

prompt_generation_visitor = AlirezaPromptGenerationVisitor()


random.seed(0)
randomlist = [
    random.sample(range(0, len(dataset)), shot_size + 1) for i in range(batch_size)
]
tasks: List["Task"] = []
for batch in range(batch_size):
    task_list = randomlist[batch]
    tasks.append(
        Task(
            question=dataset[task_list[shot_size]],
            templates=Pipeline(task_list[:shot_size])
            .to_map(lambda ind: dataset[ind])
            .to_list(),
        )
    )

prompt_list = []
grund_truth = []
for batch in range(batch_size):
    prompt_list.append(prompt_generation_visitor.visit_task(tasks[batch]))
    grund_truth.append(tasks[batch].get_ground_truth())

i = 9
# with open("./prompt_{}.txt".format(i + 1), "w") as f:
#     f.write(str(prompt_list[i]))


len(prompt_list)


# ## Load Gemma 2b


# get_ipython().system("pip install accelerate")
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.environ["HF_TOKEN"] = "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"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")

input_text = """
                from transformers import AutoTokenizer, AutoModelForCausalLM

                tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
                model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

                # This code is doing:
              """
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_length=600)
print(tokenizer.decode(outputs[0]))


input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_length=50)
print(tokenizer.decode(outputs[0]))

input_text = """
                input_text = "Write me a poem about Machine Learning."
                input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

                outputs = model.generate(**input_ids)
                print(tokenizer.decode(outputs[0]))

                # This code is doing:
              """
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_length=800)
print(tokenizer.decode(outputs[0]))


# ## Quantized loading 7b
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import os

os.environ["HF_TOKEN"] = "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"

model = "google/gemma-7b-it"
# use quantized model
pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
    },
)


messages = [
    {
        "role": "user",
        "content": """

Start Markdown 1: # Exercises
You could write the function `get_mae` yourself. For now, we'll supply it. This is the same function you read about in the previous lesson. Just run the cell below.
Start Code 1: def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
Start Markdown 2: Keras NN model can be evaluated on lots of metrics with just passing the metric name, but it is not the case for ROC_AUC score, so we will define our own auc metric function:
Start Code 2: def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
Start Markdown 3: # Wordcloud
Start Code 3: #Custom function to extract text from variety column
def get_text(column):
    words = ''
    for text in column:
        words += text
    return words
Start Markdown 4: Therefore, we'll create this little function to just return the single number we need given a pair of variables:
Start Code 4: def corr(x,y): return np.corrcoef(x,y)[0][1]

corr(housing.MedInc, housing.MedHouseVal)

Generate markdown for the bottom code according to the four samples above
 Code: def tr_plot(tr_data, start_epoch):
    #Plot the training and validation data
    tacc=tr_data.history['accuracy']
    tloss=tr_data.history['loss']
    vacc=tr_data.history['val_accuracy']
    vloss=tr_data.history['val_loss']
    Epoch_count=len(tacc)+ start_epoch
    Epochs=[]
    for i in range (start_epoch ,Epoch_count):
        Epochs.append(i+1)
    index_loss=np.argmin(vloss)#  this is the epoch with the lowest validation loss
    val_lowest=vloss[index_loss]
    index_acc=np.argmax(vacc)
    acc_highest=vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label='best epoch= '+ str(index_loss+1 +start_epoch)
    vc_label='best epoch= '+ str(index_acc + 1+ start_epoch)
    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(20,8))
    axes[0].plot(Epochs,tloss, 'r', label='Training loss')
    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )
    axes[0].scatter(index_loss+1 +start_epoch,val_lowest, s=150, c= 'blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
    axes[1].scatter(index_acc+1 +start_epoch,acc_highest, s=150, c= 'blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    #plt.style.use('fivethirtyeight')
    plt.show()


    """,
    },
]
prompt = pipeline.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
outputs = pipeline(
    prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95
)
# print(outputs[0]["generated_text"])


def generate_response(prompt_content):
    messages = [
        {
            "role": "user",
            "content": prompt_content,
        },
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipeline(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return outputs[0]["generated_text"]
