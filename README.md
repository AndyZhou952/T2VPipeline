# Data Processing
>Automatic T2V HQ Data Curation Pipeline v1.0.

## Overview
This pipeline is designed to train AI models for video generation 
based on text inputs based on MindSpore.

First, raw videos—whether sourced from the internet or public 
datasets—are divided into shorter clips using scene detection 
techniques. We offer an optional filtering mechanism to select 
specific video categories of interest. Following this, we incorporate 
`imagededup` package to remove duplicate videos from the dataset.

Next, these videos undergo an evaluation process where multiple 
scores are predicted using existing models. These scores include 
aesthetic scoring, OCR (Optical Character Recognition) for text 
detection, and optical flow scoring to assess motion. 
Only videos that meet satisfactory evaluation criteria advance 
to the captioning step. 
**(Remark: OCR and optical flow scoring will be supported in future 
release)**

After captioning, a matching score is calculated to assess the 
alignment between video and text. Samples with low matching scores
are filtered out.

In summary, our pipeline generates video-text pairs that exhibit 
high aesthetic quality, significant video motion, and strong 
semantic consistency.

## Requirement:
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

## Example Workflow:

Please first download the models and put them in the designated 
directories for scoring and captioning. More details can be 
found in the **Further Reading** section below.

### 0. Set up
```bash
ROOT_VIDEO="/path/to/video/folder"
ROOT_CLIPS="/path/to/video/clips/folder"
ROOT_META="/path/to/meta/folder"
export PYTHONPATH=$(pwd)
# run the command below to set up deduplication if needed
python pipeline/datasets/imagededup/setup.py build_ext --inplace
```

### 1. Convert dataset to CSV
```bash
# 1.1 Create a meta file from a video folder. This should output ${ROOT_META}/meta.csv
python -m pipeline.datasets.convert video ${ROOT_VIDEO} --output ${ROOT_META}/meta.csv

# 1.2 Get video information and remove broken videos. This should output ${ROOT_META}/meta_info_fmin1.csv
python -m pipeline.datasets.datautil ${ROOT_META}/meta.csv --info --fmin 1
```

### 2. Split video to clips
```bash
# 2.1 Detect scenes. This should output ${ROOT_META}/meta_info_fmin1_timestamp.csv
python -m pipeline.splitting.scene_detect ${ROOT_META}/meta_info_fmin1.csv

# 2.2 Cut video into clips based on scenes. This should produce video clips under ${ROOT_CLIPS}
python -m pipeline.splitting.cut ${ROOT_META}/meta_info_fmin1_timestamp.csv --save_dir ${ROOT_CLIPS}

# 2.3 Create a meta file for video clips. This should output ${ROOT_META}/meta_clips.csv
python -m pipeline.datasets.convert video ${ROOT_CLIPS} --output ${ROOT_META}/meta_clips.csv

# 2.4 Get clips information and remove broken ones. This should output ${ROOT_META}/meta_clips_info_fmin1.csv
python -m pipeline.datasets.datautil ${ROOT_META}/meta_clips.csv --info --fmin 1
```

### 3. Deduplication 
```bash
# 3. Deduplication. This should output ${ROOT_META}/meta_clips_info_fmin1_dedup.csv
python -m pipeline.datasets.deduplication ${ROOT_META}/meta_clips_info_fmin1.csv
```

### 4. Scoring and filtering
```bash
# 4.1.1 Calculate matching scores with an option. This should output ${ROOT_META}/meta_clips_info_fmin1_dedup_{args.option}.csv
python -m pipeline.scoring.matching.inference ${ROOT_META}/meta_clips_info_fmin1_dedup.csv --option animal --use_cpu # cpu
# modify worker_num and local_worker_num based on your resource, same below
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/matching/inference.py \
 ${ROOT_META}/meta_clips_info_fmin1_dedup.csv --option animal # Ascend
 
# 4.1.2 Filter videos based on an option. This should output ${ROOT_META}/meta_clips_info_fmin1_dedup_{args.option}_matchmin20.0.csv 
python -m pipeline.datasets.datautil \
 ${ROOT_META}/meta_clips_info_fmin1_dedup_animal.csv --matchmin 20

# 4.2.1 Predict aesthetic scores. This should output ${ROOT_META}/meta_clips_info_fmin1_dedup_{args.option}_matchmin20.0_aes.csv
python -m scoring.aesthetic.inference ${ROOT_META}/meta_clips_info_fmin1_dedup_animal_matchmin20.0.csv --use_cpu # cpu
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/aesthetic/inference.py \ 
 ${ROOT_META}/meta_clips_info_fmin1_dedup_animal_matchmin20.0.csv # Ascend

# 4.2.2 Filter by aesthetic scores. This should output ${ROOT_META}/meta_clips_info_fmin1_dedup_{args.option}_matchmin20_aesmin4.5.csv
python -m pipeline.datasets.datautil ${ROOT_META}/meta_clips_info_fmin1_dedup_animal_matchmin20.0_aes.csv --aesmin 4.5
```

### 5. Captioning and calculating matching scores
```bash
# 5.1 Generate PLLaVA caption. This should output ${ROOT_META}/meta_clips_info_fmin1_dedup_{args.option}_matchmin20_aesmin4.5_caption.csv
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/captioning/caption_pllava.py \
 ${ROOT_META}/meta_clips_info_fmin1_dedup_animal_matchmin20_aesmin4.5.csv # support Ascend only

# 5.2 Clean caption. This should output ${ROOT_META}/meta_clips_caption_cleaned.csv
python -m pipeline.datasets.datautil \
 ${ROOT_META}/meta_clips_info_fmin1_dedup_animal_matchmin20_aesmin4.5_caption.csv \
 --clean-caption --refine-llm-caption --remove-empty-caption \
 --output ${ROOT_META}/meta_clips_caption_cleaned.csv 

# 5.3 Calculate matching scores with captions. This should output ${ROOT_META}/meta_clips_caption_cleaned_matching.csv 
python -m pipeline.scoring.matching.inference \
 ${ROOT_META}/meta_clips_caption_cleaned.csv --use_cpu # cpu
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/matching/inference.py \
 ${ROOT_META}/meta_clips_caption_cleaned.csv # Ascend

# 5.4 Filter by matching scores. This should output ${ROOT_META}/meta_clips_caption_cleaned_matchmin20.0.csv
python -m pipeline.datasets.datautil \
 ${ROOT_META}/meta_clips_caption_cleaned_matching.csv \
 --matchmin 20
```

## Further Reading:
For more information, please refer to:
- [Dataset Management](./pipeline/datasets/README.md)
- [Scene Detection and Video Splitting](./pipeline/splitting/README.md)
- [Scoring and Filtering](./pipeline/scoring/README.md)
- [Captioning](./pipeline/captioning/README.md)

## TODOs:
- [x] Feature: support PLLaVA captioning
- [ ] Feature: support ShareGPT4V captioning
- [ ] Feature: support optical flow/OCR filtering
- [ ] Precision check: matching/aesthetic (CLIP+MLP)
- [ ] Enhancement: support unsupervised concept balancing
- [ ] Enhancement: support Panda-70M cutting and stitching
- [ ] Enhancement: option filtering based on multiple strings

## Acknowledgement
This pipeline for video/image data processing pipeline in MindSpore is 
based on the work [here](https://github.com/hpcaitech/Open-Sora/blob/main/docs/data_processing.md) by HPC-AI OpenSora. We thank them for their generous
support to the open source community.
