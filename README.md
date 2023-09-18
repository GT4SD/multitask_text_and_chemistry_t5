# [Multitask Text and Chemistry T5](https://arxiv.org/abs/2301.12586)

![clm](https://github.com/GT4SD/multitask_text_and_chemistry_t5/blob/main/assets/clm_pipeline.png)

> **Unifying Molecular and Textual Representations via Multi-task Language Modelling**   
> [Dimitrios Christofidellis*](https://researcher.watson.ibm.com/researcher/view.php?person=zurich-DIC), [Giorgio Giannone*](https://georgosgeorgos.github.io/), [Jannis Born](https://research.ibm.com/people/jannis-born), [Ole Winther](https://olewinther.github.io), [Teodoro Laino](https://research.ibm.com/people/teodoro-laino), [Matteo Manica](https://research.ibm.com/people/matteo-manica)  
> International Conference on Machine Learning (ICML), 2023

[[paper](https://arxiv.org/abs/2301.12586)]
[[gradio app](https://huggingface.co/spaces/GT4SD/multitask-text-and-chemistry-t5)]
[[code](https://github.com/GT4SD/multitask_text_and_chemistry_t5)]


*The recent advances in neural language models have also been successfully applied to the field of chemistry, offering generative solutions for classical problems in molecular design and synthesis planning. These new methods have the potential to fuel a new era of data-driven automation in scientific discovery. However, specialized models are still typically required for each task, leading to the need for problem-specific fine-tuning and neglecting task interrelations. The main obstacle in this field is the lack of a unified representation between natural language and chemical representations, complicating and limiting human-machine interaction.
Here, we propose the first multi-domain, multi-task language model that can solve a wide range of tasks in both the chemical and natural language domains. Our model can handle chemical and natural language concurrently, without requiring expensive pre-training on single domains or task-specific models. Interestingly, sharing weights across domains remarkably improves our model when benchmarked against state-of-the-art baselines on single-domain and cross-domain tasks. In particular, sharing information across domains and tasks gives rise to large improvements in cross-domain tasks, the magnitude of which increase with scale, as measured by more than a dozen of relevant metrics. Our work suggests that such models can robustly and efficiently accelerate discovery in physical sciences by superseding problem-specific fine-tuning and enhancing human-model interactions*.




---------
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

---------
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

---------
## Perform predictions using our models

The four variants of our model are available via the HuggignFace Hub in the following links: 

* [multitask-text-and-chemistry-t5-small-standard](https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-small-standard)  
* [multitask-text-and-chemistry-t5-small-augm](https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-small-augm) 
* [multitask-text-and-chemistry-t5-base-standard](https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-base-standard)  
* [multitask-text-and-chemistry-t5-base-augm](https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-base-augm) 

In the provided notebook (demo.ipynb), we present examples of how the model can be used for the 5 different tasks.  


---------

## Citation

```bibtex
@article{christofidellis2023unifying,
  title={Unifying Molecular and Textual Representations via Multi-task Language Modelling},
  author={Christofidellis*, Dimitrios and Giannone*, Giorgio and Born, Jannis and Winther, Ole and Laino, Teodoro and Manica, Matteo},
  journal={arXiv preprint arXiv:2301.12586},
  year={2023}
}
```
