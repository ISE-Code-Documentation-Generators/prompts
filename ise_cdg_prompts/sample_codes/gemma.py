# get_ipython().system("pip install pandas")
# get_ipython().system('pip install -U "transformers==4.38.1" --upgrade')
# get_ipython().system("pip install accelerate")
# get_ipython().system("pip install -i https://pypi.org/simple/ bitsandbytes")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import PromptDataset
    from ise_cdg_prompts.sample.main import TaskSampler
    from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor

from ise_cdg_prompts.utils.pipeline import Pipeline


import random

random.seed(0)


from ise_cdg_prompts.alireza_dataset import AlirezaDataset
from ise_cdg_prompts.prompt_generation_visitor.alireza import (
    AlirezaPromptGenerationVisitor,
)
from ise_cdg_prompts.sample.random import RandomTaskSampler

# Load dataset containing markdown and code cells
dataset: "PromptDataset" = AlirezaDataset(path="./final_dataset.csv")
task_sampler: "TaskSampler" = RandomTaskSampler(
    dataset=dataset,
    shot_size=4,
    sample_size=10,
)
prompt_generation_visitor: "PromptGenerationVisitor" = AlirezaPromptGenerationVisitor()


tasks = task_sampler.generate_samples()
prompt_list = (
    Pipeline(tasks)
    .to_map(lambda task: prompt_generation_visitor.visit_task(task))
    .to_list()
)
grund_truth = Pipeline(tasks).to_map(lambda task: task.get_ground_truth()).to_list()


# ## Load Gemma 2b
from ise_cdg_prompts.llm_api.gemma_test import KossherLLMTest


# ## Quantized loading 7b
model = None
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import os

os.environ["HF_TOKEN"] = "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"

# use quantized model
pipeline = pipeline(
    "text-generation",
    model="google/gemma-7b-it",
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
    },
)

kossher2 = []

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
print(outputs[0]["generated_text"])
kossher2.append(outputs[0]["generated_text"])


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

print(generate_response(prompt_list[-1]))
kossher2.append(generate_response(prompt_list[-1]))