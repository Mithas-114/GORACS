# Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems

---

This is the pytorch implementation of our proposed GORACS framework.


## Enviroment
---

* Anaconda 3
* Python 3.8.0
* Cuda 11.8
* PyTorch 2.0.0

Please run the following code to install the necessary packages.

```bash
pip install -r requirements.txt
```

## Usage
---

This section provides a guide on how to use the ITRA algorithm from the GORACS framework to minimize the POO score (Eq. 10). 

Subsequently, we offer a detailed explanation of our experimental procedure and step-by-step instructions for running the provided code, using the Games dataset as an example.

### Dataset

* **The Games and the Movies dataset are provided by [Amazon Review Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), and the Food dataset is provided by [Amazon Review Data].**(https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).
* Download the 5-core data for Video Games (Video_Games_5.json.gz) and the metadata (meta_Video_Games.json.gz). 
* Place the extracted **Video_Games_5.json** and **meta_Video_Games.json** files into **"./datasets/games/"** folder.

### Models

* Download [Roberta-base](https://huggingface.co/FacebookAI/roberta-base) and place the files into **"./models/Encoders/Roberta-base/"** folder. 
* Download [LLaMa-7B](https://huggingface.co/huggyllama/llama-7b) and place the files into **"./models/LLMs/LLaMa-7B/"** folder.


### 1. Example of the ITRA Algorithm

* Please follow the ```example.ipynb``` in **"./codes/select/"**. 

* Due to GitHub's 25MB upload limit, we randomly sample 15,000 training samples and 5,000 validation samples from the SeqRec task on Games, saving the embeddings in **"./codes/select/example_data/"**. Please run the following demo to generate the complete dataset.


### 2. Demo of SeqRec

#### (1) Preprocess

In the folder **./codes/data/** :

* Follow ```data_process_SeqRec.ipynb``` to process dataset. 
* Follow ```generate_item_embeddings.ipynb``` to generate item title embeddings for BIGRec.
* Follow ```generate_data_embeddings.ipynb``` with ```task="SeqRec"``` to generate sample embeddings.

The processed data are all saved in folder **"./datasets/games/SeqRec"**.

#### (2) Computing Gradient Norms

The codes for computing gradient norms. In the folder **./codes/grads/** :

* Use ```compute_grads_seqrec.sh```: 
    
    ```shell
    sh compute_grads_seqrec.sh games
    ```
* The result is saved as **"./datasets/games/SeqRec/grads/all"**.

#### (3) Coreset Selection

The codes for coreset selection using ITRA algorithm (Algorithm 1) to minimize POO score. In the folder **./codes/select/** :

* Use ```select_seqrec.sh```: 
    ```shell
    sh select_seqrec.sh <dataset> <lambda> <budget> <num_max_exchange> <num_exchange_candidates>
    ```
    e.g., 
    ```shell
    sh select_seqrec.sh games 0.1 1024 100 30
    ```

* The selected coresets are saved in **"./inputs/games/SeqRec/"**.


#### (4) Finetuning and Evaluation


The codes for finetuning [BIGRec](https://github.com/SAI990323/BIGRec). In the folder **"./codes/finetune/"** :

* Use ```finetune_and_evaluate_seqrec.sh```: 
    ```shell
    sh finetune_and_evaluate_seqrec.sh
        - Enter dataset : <dataset>
        - Enter seed : <seed>
        - Enter lambda value : <lamda>
        - Enter selection budget : <budget>
        - Enter max exchange iterations : <num_max_exchange>
    ```
    e.g., 
    ```shell
    sh finetune_and_evaluate_seqrec.sh
        - Enter dataset : games
        - Enter seed : 0
        - Enter lambda value : 0.1
        - Enter selection budget : 1024
        - Enter max exchange iterations : 100
    ```

* The results, including the finetuned model LoRA weights, inferences for generating recommendations, and recommendation scores, are saved in the following directories:
  
    1. weights : **./outputs/games/SeqRec/weights/**
    2. inferences : **./outputs/games/SeqRec/inferences/**
    3. scores: **./outputs/games/SeqRec/scores/**

#### (5) Test

* We also present the LoRA weights trained on Games dataset for test in **./models/LoRAs/demo-games-seqrec/**.
* Please run ```evaluate_seqrec.sh``` in the folder **./codes/finetune/** :
  
  ```shell
  sh evaluate_seqrec.sh
  ```

* The resulted scores are saved as **./outputs/games/SeqRec/scores/evaluate_demo.json**.

### 3. Demo of CTRPre

#### (1) Preprocess

In the folder **./codes/data/** :

* Follow ```data_process_CTRPre.ipynb``` to process dataset. 
* Follow ```generate_data_embeddings.ipynb``` with ```task="CTRPre"``` to generate sample embeddings.

The processed data are all saved in folder **"./datasets/games/CTRPre"**.

#### (2) Computing Gradient Norms

The codes for computing gradient norms. In the folder **./codes/grads/** :

* Use ```compute_grads_ctrpre.sh```: 
    
    ```shell
    sh compute_grads_ctrpre.sh games
    ```
* The result is saved as **"./datasets/games/CTRPre/grads/all"**.

#### (3) Coreset Selection

The codes for coreset selection using Label-enhanced ITRA algorithm (Algorithm 2). In the folder **./codes/select/** :

* Use ```select_ctrpre.sh```: 
    ```shell
    sh select_ctrpre.sh <dataset> <lambda> <budget> <num_max_exchange> <num_exchange_candidates>
    ```
    e.g., 
    ```shell
    sh select_ctrpre.sh games 0.5 64 20 3
    ```

* The selected coresets are saved in **"./inputs/games/CTRPre/"**.


#### (4) Finetuning and Evaluation


The codes for finetuning [TALLRec](https://github.com/SAI990323/TALLRec). In the folder **./codes/finetune/** :

* Use ```finetune_and_evaluate_ctrpre.sh```: 
    ```shell
    sh finetune_and_evaluate_ctrpre.sh
        - Enter dataset : <dataset>
        - Enter seed : <seed>
        - Enter lambda value : <lamda>
        - Enter selection budget : <budget>
        - Enter max exchange iterations : <num_max_exchange>
    ```
    e.g., 
    ```shell
    sh finetune_and_evaluate_ctrpre.sh
        - Enter dataset : games
        - Enter seed : 0
        - Enter lambda value : 0.5
        - Enter selection budget : 64
        - Enter max exchange iterations : 20
    ```

* The results, including the finetuned model LoRA weights and recommendation scores, are saved in the following directories:
  
    1. weights : **./outputs/games/CTRPre/weights/**
    2. scores: **./outputs/games/CTRPre/scores/**


#### (5) Test

* We also present the LoRA weights trained on Games dataset for test in **./models/LoRAs/demo-games-ctrpre/**.
* Please run ```evaluate_ctrpre.sh``` in the folder **./codes/finetune/** :
  
  ```shell
  sh evaluate_ctrpre.sh
  ```
  
* The resulted scores are saved as **./outputs/games/CTRPre/scores/evaluate_demo.json**.

