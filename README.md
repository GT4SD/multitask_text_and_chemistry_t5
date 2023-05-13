# Multitask Text and Chemistry T5

## Requirements

Install requirements:

```sh
pip install -r requirements.txt
```

Create a dedicated kernel:

```sh
python -m ipykernel install --name text_chem_t5_demo
```

Good to go :rocket:

## Model training

The training process is carried out using the language modeling trainer based on Hugging Face transformers (Wolf et al., 2020) and PyTorch Lightning (Falcon
and The PyTorch Lightning team, 2019) from the GT4SD library (Manica et al., 2022). To reproduce the training, you need first to install the GT4SD library. For more information regarding the installation process of the GT4SD library, you can visit its [page](https://github.com/GT4SD/gt4sd-core). Once GT4SD is installed, you can use the following command to launch our training. Note that the provided dataset splits in the `dataset-sample` directory contain just a small subset of our actual dataset splits.
To regenerate our full training dataset, we refer the interested reader to the respective section of our paper and the references that are provided there.

```buildoutcfg

gt4sd-trainer --training_pipeline_name language-modeling-trainer \
    --model_name_or_path t5-base \
    --lr 6e-4 \
    --lr_decay 0.99 \
    --batch_size 8 \
    --train_file dataset-sample/train.jsonl \
    --validation_file dataset-sample/valid.jsonl \
    --default_root_dir text_chem_t5_base \
    --type cgm \
    --val_check_interval 20000  \
    --max_epochs 1 \
    --limit_val_batches 500 \
    --accumulate_grad_batches 4 \
    --log_every_n_steps 5000 \
    --monitor val_loss \
    --save_top_k 1 \
    --mode min \
    --every_n_train_steps 20000 \
    --accelerator 'ddp' 

```

The prompt templates that we have used for the 5 different tasks can be found in the following table, where \<input> represents the actual input for each task. 

|          Task         |                              Template                             |
|:---------------------:|:-----------------------------------------------------------------:|
|   Forward prediction  |       Predict the product of the following reaction: \<input>      |
|     Retrosynthesis    | Predict the reaction that produces the following product: \<input> |
|  Paragraph-to-actions |  Which actions are described in the following paragraph: \<input>  |
| Description-to-smiles |          Write in SMILES the described molecule: \<input>          |
|   Smiles-to-caption   |               Caption the following SMILES: \<input>               |


## Perform predictions using our models

The two T5-small based variants of our model are available via the HuggignFace Hub in the following links: [multitask-text-and-chemistry-t5-small-standard](https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-small-standard) and  [multitask-text-and-chemistry-t5-small-augm](https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-small-augm). In the provided notebook (demo.ipynb), we present examples of how the model can be used for the 5 different tasks.  
