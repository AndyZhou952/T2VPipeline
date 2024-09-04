import os

import pandas as pd
import mindspore as ms
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size, init
from tqdm import tqdm

from pipeline.datasets.utils import extract_frames, pil_loader, is_video
from config import parse_args
from text_system import TextSystem

class VideoTextDataset:
    # by default, we use the middle frame for OCR
    def __init__(self, meta_path, transform = None):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample['path']

        # extract frames
        if not is_video(path):
            images = [pil_loader(path)]
        else:
            num_frames = sample["num_frames"] if "num_frames" in sample else None
            images = extract_frames(path, points = [0.5], backend = "opencv", num_frames = num_frames)

        # transform & stack
        if self.transform is not None:
            images = [self.transform(img) for img in images]

        return index, images

    def __len__(self):
        return len(self.meta)

def main():
    args = parse_args()

    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_ocr{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()


    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
    ms.set_auto_parallel_context(parallel_mode = ms.ParallelMode.DATA_PARALLEL)
    init()

    # initialize the TextSystem for OCR, detection algo DB++, recognition CRNN
    text_system = TextSystem(args)

    raw_dataset = VideoTextDataset(args.meta_path)
    rank_id = get_rank()
    rank_size = get_group_size()
    dataset = ds.GeneratorDataset(source=raw_dataset, column_names=['index', 'images'], shuffle=False,
                                      num_shards = rank_size, shard_id = rank_id)
    # TODO: batch size > 1 only support images with same shapes, or else will raise error
    dataset = dataset.batch(args.bs, drop_remainder=False)
    iterator = dataset.create_dict_iterator(num_epochs = 1)

    # ocr detection and recognition
    indices_list = []
    ocr_results_list  = []
    for batch in tqdm(iterator):
        indices = batch["index"]
        images = batch["images"].asnumpy()

        batch_ocr_results = []
        for image in images:
            image = image.squeeze()
            boxes, text_scores, _ = text_system(image)
            batch_ocr_results.append({"boxes": boxes, "texts": text_scores})

        indices_list.extend(indices.tolist())
        ocr_results_list.extend(batch_ocr_results)

    meta_local = raw_dataset.meta
    meta_local['ocr'] = ocr_results_list
    meta_local.to_csv(out_path, index = False)
    print(meta_local)
    print(f"New meta with OCR scores saved to '{out_path}'.")

if __name__ == "__main__":
    main()
