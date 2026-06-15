# How to Run an Hugging Face Model


### Pipeline Function
The easiest way to do this is with the pipeline function. IT handles all the processes and makes the process smooth. We just have to use the pipeline function. 

But to understand whats actually happening inside the function, we have to do it manually. 

### pipeline() vs manual loading

`pipeline()` does everything in one line. The manual path does the exact same work, just spelled out step by step. Same input, same answer.

```python
# pipeline()                                  # manual (what pipeline does under the hood)
from transformers import pipeline             # AutoTokenizer.from_pretrained(model_name)  -> load tokenizer
                                              # tokenizer(text, return_tensors="pt")       -> tokenize (text -> numbers)
clf = pipeline("sentiment-analysis",          # AutoModelForSequenceClassification.from_pretrained(model_name) -> load model
               model=model_name)              # model(**inputs)                            -> forward pass (-> logits)
clf("I love this movie!")                     # torch.softmax(logits, dim=-1)              -> softmax (logits -> probabilities)
```

Both print `POSITIVE` with a score of about `0.9999` for `"I love this movie!"`. The pipeline just hides the five manual steps: load tokenizer, tokenize, load model, forward pass, softmax.

### The process

We first get the raw input. We preprocess this using tokenizers and convert it into numbers.  
These numbers are then fed into the model. The model does its processing and after lots of changes to the numbers, we are given some final numbers, logits. These logits are the raw outputs given to us.  
Then with the help of softmax, we convert these logits into human readable form. We can convert it into probability/percentages. 

## AutoTokenizer

AI models have different tokenizers for different LLMs. When we want to run them, we dont have to worry about which model uses which tokenizer method. Autotokenizer takes care of that for us.

## AutoModel

When we use automodel, hugging face automaically assigns a model relevant to the task we have asked for. For example if we asked for an automodel for sentiment analysis, HF gives us distilbert.

## HF Cache

When we run some code which requires a HF model, it downloads it from the HF website. But if we need to continuously download it, its a big task, so HF saves it to the cache of the folder.  
This makes it faster when we want to run repetitive tasks.  

But we have to be careful, when I tried doing it with multiple model in different notebooks, my laptop started lagging a lot and slowed down. So we have to make sure the laptop is compatible enough for it. There was a python feature called pylanche. This is responsible for autocomplete features in python. This was occupying approx 2GB of my RAM. So, I had to disable it completely.

```
(.venv) chaitanya@LAPTOP-M79K4R8E:~/bharatgen-speech-intern-2026$ ls -la ~/.cache/huggingface/hub/
total 40
drwxr-xr-x 9 chaitanya chaitanya 4096 Jun  8 05:33 .
drwxr-xr-x 5 chaitanya chaitanya 4096 Jun  5 11:10 ..
drwxr-xr-x 9 chaitanya chaitanya 4096 Jun  8 05:33 .locks
-rw-r--r-- 1 chaitanya chaitanya  191 May 15 11:54 CACHEDIR.TAG
drwxr-xr-x 6 chaitanya chaitanya 4096 Jun  5 11:10 datasets--ai4bharat--Shrutilipi
drwxr-xr-x 6 chaitanya chaitanya 4096 May 26 07:15 models--HuggingFaceTB--SmolLM2-360M
drwxr-xr-x 5 chaitanya chaitanya 4096 Jun  8 05:33 models--bharatgenai--Shrutam-2
drwxr-xr-x 6 chaitanya chaitanya 4096 May 15 11:54 models--distilbert--distilbert-base-uncased-finetuned-sst-2-english
drwxr-xr-x 6 chaitanya chaitanya 4096 May 27 05:42 models--distilbert-base-uncased-finetuned-sst-2-english
drwxr-xr-x 6 chaitanya chaitanya 4096 Jun  1 05:53 models--openai--whisper-tiny
```

Each model gets stored here after downloading, so we dont have to download it again and again.

## Training Mode vs Inference

Training here refers to the process of the neural network learning.  

Inference is when the model only gives the answers

## model.eval() and torch.no_grad()

We dont always need the model to keep learning.  

To tell pytorch that we want to switch to inference mode, we can use these two commands.

### model.eval()

This mainly affects the layers of the NN.  

1. Dropout
While training, some neurons may be switched off to improve efficiency and accuracy. But for inference mode, we need all of them to be active

2. Batchnorm
WHile training, the model will use weights that are currently ongoing in training. But for inference, we need it to be the average of all.

Here, model.eval() changes these two factors into their inference modes.

### torch.no_grad()

This tells pytorch to stop tracking gradients. Gradients are time and memory consuming. They require more time to process. SO by turning them off, we are improving the overall speed of the model.

### Logits

Logits are the raw outputs we have gotten from the model after processing it. They can be any number, negative posotive, decimal. It gives an output number for every input number given

### Softmax

When we get the logits from the model, they are of no use to us because we can't understand them. They are just raw numbers. We need to understand what each of these numbers mean.  
Softmax gives us the probability for each of the logits. So then the numbers are between 0 and 1. So we can use it to determine some function. 


## trust_remote_code=True

HF libraries have built in code for architecture like BERT. But if some model uses a different architecture, so the library doesn't know how to run them.  

So the developer also uploads a python code along with the model. This python code contains the weights, the code which tells us the architecture of the code and also settings.  

BUt we can't always be sure that the code is safe, so the transformer blocks it for security concerns.  
This is where the trust_remote_code=True command comes. This tells the transformer that you trust the code and gives permission to run it.  

This command is mainly a security feature. 

## config.json

This is basically the blueprint of the model. IT tells us about the model,, things like the size, or layers and how they can be arrranged. Without this, the model is just ranodm numbers and we wont be able to make sense of it.  

When we load a model from HF, it first reads the json file to get an understanding of how the model is doing it, then it downloads the weights.  

### hidden_size

This is the width of the model. 

Inside a model, each token gets converted to a vector. hidden size tells us how long the list of numbers should be. 

If its large, the model will be more accurate but slower. If smaller, model will be faster but less accurate.  

### num_hidden_layers

This tells us how many layers should be there. Each layer is a block of the transformer. 

Its similar to hidden size, if more layers then more efficient, but also slower.

### vocab_size

THis refers to the size of the tokenizer library, tells us how many individual tokens exist. If vocab size is 1000, then there are 1000 tokens with their each individual token ID.

It is basically the number of unique tokens the model can understand.