# Scoring and Filtering

- [Scoring and Filtering](#scoring-and-filtering)
  - [Aesthetic Score](#aesthetic-score)
  - [Matching Score](#matching-score)
  - [OCR](#OCR)
  - [Filtering](#filtering)

## Aesthetic Score

To evaluate the aesthetic quality of videos, we use the 
scoring model from [CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor). 
This model is trained on 176K SAC (Simulacra Aesthetic 
Captions) pairs, 15K LAION-Logos (Logos) pairs, and 
250K AVA (The Aesthetic Visual Analysis) image-text pairs.

The aesthetic score is between 1 and 10, where 5.5 can 
be considered as the threshold for fair aesthetics, 
and 6.5 for high aesthetics. Good text-to-image models 
can achieve a score of 7.0 or higher.

For videos, we extract the first, last, and the middle 
frames for evaluation. The script also supports images 
as input.

Download the scoring model using the following command to `./pretrained_models/aesthetic.pth`.

```bash
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth -O pretrained_models/aesthetic.pth
```

First, download the CLIP model [here](https://github.com/openai/CLIP).

Then, use the following script to convert `.pth` to `.ckpt` for use in MindSpore.

```bash
python -m tools.pth_to_ckpt --model aesthetic \ 
 --pth_path 'pretrained_models/aesthetic.pth' \ 
 --save_path 'pretrained_models/aesthetic.ckpt' \
 --show_pth --show_ckpt --convert --value
```


Then, run the following command if you use CPU. **Make sure** the meta file has column `path` (path to the sample).
```bash
python -m scoring.aesthetic.inference /path/to/meta.csv --use_cpu
```
If running on Ascend, you may use 
```bash
export PYTHONPATH=$(pwd)
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/aesthetic/inference.py \ 
 /path/to/meta.csv 
```
Modify `worker_num` and `local_worker_num` based on your resource.


## Matching Score

Matching scores are calculated to evaluate the alignment between an image/video and its caption.
Here, we use the [CLIP](https://github.com/openai/CLIP) model, which is trained on image-text pairs.
For videos, we extract the first, last, and the middle frame and compare it with the caption. 
We record the highest score among the three as the matching score.

First, download the [CLIP ViT-L/14 model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/clip/clip_vit_l_14.ckpt) 
and the [tokenizer](https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz).

Then, run the following command. **Make sure** the meta file has the column `path` (path to the sample).
For matching scores for captions, the meta file should also have the column `text` (caption of the sample).
For option filtering, the argument `--option` must be provided
```bash
# for option filtering
python -m pipeline.scoring.matching.inference /path/to/meta.csv --use_cpu --option animal 
```
If running on Ascend, you may use 
```bash
export PYTHONPATH=$(pwd)
# calculate the matching scores with captions, the column `text` must be present
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/matching/inference.py \ 
 /path/to/meta.csv 
```
Modify `worker_num` and `local_worker_num` based on your resource.

This should output `/path/to/meta_match.csv` with column `match`. Higher matching scores indicate better image-text/video-text alignment.

## OCR
OCR (Optical Character Recognition) is used to detect and recognize 
text in images and video frames. We use the MindOCR package for 
this task, which supports a variety of state-of-the-art OCR 
algorithms. MindOCR supports both detection and recognition of text 
in natural scenes. By default, we use DB++ for detection and
CRNN for recognition. You can check the [MindOCR](https://github.com/mindspore-lab/mindocr/tree/main/tools/infer/text) 
page for the full list.

Run the following command for inference. **Make sure** the meta file has the column `path` (path to the sample).

Currently, we only support captioning on Ascend on a single chip.
Data parallelism may be supported in future release. 

```bash
export PYTHONPATH=$(pwd)
msrun --worker_num=1 --local_worker_num=1 --join=True \
 --log_dir=msrun_log pipeline/scoring/ocr/inference.py \
 /path/to/meta.csv 
```

You can find the results in the `ocr` column of the csv file. It 
will be stored in the following format:
```angular2html
[{"transcription": "canada", "points": [[430, 148], [540, 148], [540, 171], [430, 171]]}, ...]
```

## Filtering
Once scores are obtained, it is simple to filter samples based on these scores. Here is an example to remove
samples of aesthetic score < 5.0.
```
python -m pipeline.datasets.datautil /path/to/meta.csv --aesmin 5
```
This should output `/path/to/meta_aesmin5.0.csv` with column `aes` >= 5.0
