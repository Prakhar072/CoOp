# Prompt Learning for Vision-Language Models

This repo contains the codebase of a series of research projects focused on adapting vision-language models like [CLIP](https://arxiv.org/abs/2103.00020) to downstream datasets via *prompt learning*:

* [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557), in CVPR, 2022.
* [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134), IJCV, 2022.

# Overleaf Final Project report
Our project report can be found at this link - https://www.overleaf.com/project/69f8c7a6535aa7af0a639017
The report has been generated in the neurips format using the package {neurips_2024}.

# adding new datasets
choose your dataset root
DATA_ROOT=/full/path/to/datasets
mkdir -p "$DATA_ROOT/oxford_pets/images" "$DATA_ROOT/oxford_pets/annotations"

if you downloaded the official archives to ~/Downloads:
tar -xzf ~/Downloads/images.tar.gz -C "$DATA_ROOT/oxford_pets/images"
tar -xzf ~/Downloads/annotations.tar.gz -C "$DATA_ROOT/oxford_pets/annotations"

copy the split JSON (if you have it)
cp ~/Downloads/split_zhou_OxfordPets.json "$DATA_ROOT/oxford_pets/split_zhou_OxfordPets.json"

quick checks
ls -l "$DATA_ROOT/oxford_pets"
ls -l "$DATA_ROOT/oxford_pets/images" | head
ls -l "$DATA_ROOT/oxford_pets/annotations" | head

## How to Install
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `CoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## How to Run

Click a paper below to see the detailed instructions on how to run the code to reproduce the results.

* [Learning to Prompt for Vision-Language Models](COOP.md)
* [Conditional Prompt Learning for Vision-Language Models](COCOOP.md)

compare 2 different output directories:
python interpret_prompt.py --dir1 output_ce --dir2 output_0.2hybrid --out prompt_analysis_out --topk 10

interpret prompts:
python interpret_prompt.py --file output_0.2hybrid/fgvc_aircraft/CoOp/rn50_ep50_4shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200 --topk 1

export prompt tokens to a csv:
python extract_prompt_tokens.py --models models.txt --out tokens.csv --topk 1 --backbone RN50

run training session on all datasets, for all configurations:
bash scripts/coop/run_all.sh

run single training session:
bash scripts/coop/main.sh caltech101 rn50_ep50 end 16 8 False custom


## Models and Results

- The pre-trained weights of CoOp (both M=16 & M=4) on ImageNet based on RN50, RN101, ViT-B/16 and ViT-B/32 can be downloaded altogether via this [link](https://drive.google.com/file/d/18ypxfd82RR0pizc5MM1ZWDYDk4j0BtPF/view?usp=sharing). The weights can be used to reproduce the results in Table 1 of CoOp's paper (i.e., the results on ImageNet and its four variants with domain shift). To load the weights and run the evaluation code, you will need to specify `--model-dir` and `--load-epoch` (see this [script](https://github.com/KaiyangZhou/CoOp/blob/main/scripts/eval.sh) for example).
- The raw numerical results can be found at this [google drive link](https://docs.google.com/spreadsheets/d/12_kaFdD0nct9aUIrDoreY0qDunQ9q9tv/edit?usp=sharing&ouid=100312610418109826457&rtpof=true&sd=true).

## Citation
If you use this code in your research, please kindly cite the following papers

```bash
@inproceedings{zhou2022cocoop,
    title={Conditional Prompt Learning for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}

@article{zhou2022coop,
    title={Learning to Prompt for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    journal={International Journal of Computer Vision (IJCV)},
    year={2022}
}
```
