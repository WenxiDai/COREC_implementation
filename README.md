# Token-Controlled Re-ranking for Sequential Recommendation via LLMs

This repository contains the official implementation for the paper **Token-Controlled Re-ranking for Sequential Recommendation via LLMs**.

## Contribution

We design and implement COREC, a comprehensive token-augmented re-ranking framework. COREC integrates control token construction with textual user information, enabling flexible control schemes to augment recommendation results through explicit user control.

## Framework Pipeline

The overall architecture of the COREC framework is illustrated below.

![COREC Framework Pipeline](img/COREC_pipeline.jpg)


## How to Run the Code

Follow the steps below to set up the environment and reproduce the results.

### 1. Environment Setup
First, create the conda environment from the `environment.yml` file.

```bash
conda env create -f environment.yml
conda activate corec_rec
```

### 2. Preprocess Data
Next, run the preprocessing script to prepare the dataset.

```bash
./preprocess.sh
```

### 3. Train Model

Once the data is preprocessed, use the following command to start training the COREC model.

```bash
./train.sh
```

### 4. Run Inference
After the model is trained, you can generate recommendations using the inference script.

```bash
./inference.sh
```

### 5. Evaluation
To evaluate the performance of the generated recommendations, run the evaluation script.

```bash
./evaluate.sh
```

## Reference

If you use COREC or find our work helpful, please cite:

```
@misc{dai2025tokencontrolledrerankingsequentialrecommendation,
      title={Token-Controlled Re-ranking for Sequential Recommendation via LLMs}, 
      author={Wenxi Dai and Wujiang Xu and Pinhuan Wang and Dimitris N. Metaxas},
      year={2025},
      eprint={2511.17913},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2511.17913}, 
}
```
