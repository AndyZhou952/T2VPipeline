# - This file is adapted from https://github.com/hpcaitech/Open-Sora/blob/main/opensora/datasets/read_video.py
# and also https://github.com/hpcaitech/Open-Sora/blob/main/tools/datasets/utils.py

import cv2
import gc
import re
import mindspore as ms
import numpy as np
from PIL import Image

import math
import os
import warnings
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union
import av

MAX_NUM_FRAMES = 2500

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# from torchvision.datasets.folder
def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def is_video(filename):
    ext = os.path.splitext(filename)[-1].lower()
    return ext in VID_EXTENSIONS

def extract_frames(
    video_path,
    frame_inds=None,
    points=None,
    backend="opencv",
    return_length=False,
    num_frames=None,
):
    """
    Args:
        video_path (str): path to video
        frame_inds (List[int]): indices of frames to extract
        points (List[float]): values within [0, 1); multiply #frames to get frame indices
    Return:
        List[PIL.Image]
    """
    assert backend in ["av", "opencv", "decord"]
    assert (frame_inds is None) or (points is None)

    if backend == "av":
        import av

        container = av.open(video_path)
        if num_frames is not None:
            total_frames = num_frames
        else:
            total_frames = container.streams.video[0].frames

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frames = []
        for idx in frame_inds:
            if idx >= total_frames:
                idx = total_frames - 1
            target_timestamp = int(idx * av.time_base / container.streams.video[0].average_rate)
            container.seek(target_timestamp)
            frame = next(container.decode(video=0)).to_image()
            frames.append(frame)

        if return_length:
            return frames, total_frames
        return frames

    elif backend == "decord":
        import decord

        container = decord.VideoReader(video_path, num_threads=1)
        if num_frames is not None:
            total_frames = num_frames
        else:
            total_frames = len(container)

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frame_inds = np.array(frame_inds).astype(np.int32)
        frame_inds[frame_inds >= total_frames] = total_frames - 1
        frames = container.get_batch(frame_inds).asnumpy()  # [N, H, W, C]
        frames = [Image.fromarray(x) for x in frames]

        if return_length:
            return frames, total_frames
        return frames

    elif backend == "opencv":
        cap = cv2.VideoCapture(video_path)
        if num_frames is not None:
            total_frames = num_frames
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frames = []
        for idx in frame_inds:
            if idx >= total_frames:
                idx = total_frames - 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            # HACK: sometimes OpenCV fails to read frames, return a black frame instead
            try:
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
            except Exception as e:
                print(f"[Warning] Error reading frame {idx} from {video_path}: {e}")
                # First, try to read the first frame
                try:
                    print(f"[Warning] Try reading first frame.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                # If that fails, return a black frame
                except Exception as e:
                    print(f"[Warning] Error in reading first frame from {video_path}: {e}")
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame = Image.new("RGB", (width, height), (0, 0, 0))

            # HACK: if height or width is 0, return a black frame instead
            if frame.height == 0 or frame.width == 0:
                height = width = 256
                frame = Image.new("RGB", (width, height), (0, 0, 0))

            frames.append(frame)

        if return_length:
            return frames, total_frames
        return frames
    else:
        raise ValueError

def read_video_av(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "sec",
    output_format: str = "THWC",
) -> Tuple[ms.Tensor, ms.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames (as Mindspore Tensors) and empty audio frames.

    Args:
        filename (str): Path to the video file.
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video.
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time.
        pts_unit (str, optional): Unit in which start_pts and end_pts values will be interpreted.
        output_format (str, optional): The format of the output video tensors, either "THWC" or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames.
        info (Dict): Metadata for the video.
    """
    # format
    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")
    # file existence
    if not os.path.exists(filename):
        raise RuntimeError(f"File not found: {filename}")
    # pts check
    if end_pts is None:
        end_pts = float("inf")
    if end_pts < start_pts:
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")
    # ==get video info==
    # TODO: creating an container leads to memory leak (1G for 8 workers 1 GPU)
    container = av.open(filename, metadata_errors="ignore")
    video_stream = container.streams.video[0]
    # fps
    video_fps = video_stream.average_rate
    info = {'video_fps': float(video_fps)} if video_fps is not None else {}
    total_frames = video_stream.frames if video_stream.frames > 0 else MAX_NUM_FRAMES  # Fallback for total frames

    video_frames = np.zeros((total_frames, video_stream.height, video_stream.width, 3), dtype=np.uint8)
    container.close()
    del container

    # TODO:
    # == read ==
    try:
        # TODO: The reading has memory leak (4G for 8 workers 1 GPU)
        container = av.open(filename, metadata_errors="ignore")
        assert container.streams.video is not None
        video_frames = _read_from_stream(
            video_frames,
            container,
            start_pts,
            end_pts,
            pts_unit,
            container.streams.video[0],
            {"video": 0},
            filename=filename,
        )
    except av.AVError as e:
        print(f"[Warning] Error while reading video {filename}: {e}")

    if output_format == "TCHW":
        video_frames = video_frames.transpose(0, 3, 1, 2)  # [T, H, W, C] to [T, C, H, W]

    return video_frames, info

def _read_from_stream(
    video_frames,
    container: "av.container.Container",
    start_offset: float,
    end_offset: float,
    pts_unit: str,
    stream: "av.stream.Stream",
    stream_name: Dict[str, Optional[Union[int, Tuple[int, ...], List[int]]]],
    filename: Optional[str] = None,
) -> List["av.frame.Frame"]:
    if pts_unit == "sec":
        # TODO: we should change all of this from ground up to simply take
        # sec and convert to MS in C++
        start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
        if end_offset != float("inf"):
            end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
    else:
        warnings.warn("The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")

    should_buffer = True
    max_buffer_size = 5
    if stream.type == "video":
        # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
        # so need to buffer some extra frames to sort everything
        # properly
        extradata = stream.codec_context.extradata
        # overly complicated way of finding if `divx_packed` is set, following
        # https://github.com/FFmpeg/FFmpeg/commit/d5a21172283572af587b3d939eba0091484d3263
        if extradata and b"DivX" in extradata:
            # can't use regex directly because of some weird characters sometimes...
            pos = extradata.find(b"DivX")
            d = extradata[pos:]
            o = re.search(rb"DivX(\d+)Build(\d+)(\w)", d)
            if o is None:
                o = re.search(rb"DivX(\d+)b(\d+)(\w)", d)
            if o is not None:
                should_buffer = o.group(3) == b"p"
    seek_offset = start_offset
    # some files don't seek to the right location, so better be safe here
    seek_offset = max(seek_offset - 1, 0)
    if should_buffer:
        # FIXME this is kind of a hack, but we will jump to the previous keyframe
        # so this will be safe
        seek_offset = max(seek_offset - max_buffer_size, 0)
    try:
        # TODO check if stream needs to always be the video stream here or not
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError as e:
        print(f"[Warning] Error while seeking video {filename}: {e}")
        return []

    # == main ==
    buffer_count = 0
    frames_pts = []
    cnt = 0
    try:
        for _idx, frame in enumerate(container.decode(**stream_name)):
            frames_pts.append(frame.pts)
            video_frames[cnt] = frame.to_rgb().to_ndarray()
            cnt += 1
            if cnt >= len(video_frames):
                break
            if frame.pts >= end_offset:
                if should_buffer and buffer_count < max_buffer_size:
                    buffer_count += 1
                    continue
                break
    except av.AVError as e:
        print(f"[Warning] Error while reading video {filename}: {e}")

    # garbage collection for thread leakage
    container.close()
    del container
    # NOTE: manually garbage collect to close pyav threads
    gc.collect()

    # ensure that the results are sorted wrt the pts
    # NOTE: here we assert frames_pts is sorted
    start_ptr = 0
    end_ptr = cnt
    while start_ptr < end_ptr and frames_pts[start_ptr] < start_offset:
        start_ptr += 1
    while start_ptr < end_ptr and frames_pts[end_ptr - 1] > end_offset:
        end_ptr -= 1
    if start_offset > 0 and start_offset not in frames_pts[start_ptr:end_ptr]:
        # if there is no frame that exactly matches the pts of start_offset
        # add the last frame smaller than start_offset, to guarantee that
        # we will have all the necessary data. This is most useful for audio
        if start_ptr > 0:
            start_ptr -= 1
    result = video_frames[start_ptr:end_ptr].copy()
    return result