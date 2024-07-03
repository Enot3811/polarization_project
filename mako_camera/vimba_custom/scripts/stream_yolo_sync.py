"""Синхронный стрим с камеры на YOLOv7."""


import sys
from pathlib import Path
import json
import argparse
from datetime import datetime

from vimba import Vimba, PixelFormat
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

sys.path.append(str(Path(__file__).parents[3]))
from mako_camera.vimba_custom.vimba_tools import get_camera
from mako_camera.cameras_utils import split_raw_pol
from yolov7.dataset import create_yolov7_transforms
from utils.model_utils import (
    create_yolo, load_yolo_checkpoint, yolo_inference, coco_idx2label)
from utils.torch_utils.torch_functions import (
    image_tensor_to_numpy, draw_bounding_boxes)
from utils.argparse_utils import unit_interval
from utils.data_utils.data_functions import save_image


def main(
    camera_id: str, config_pth: Path, conf_thresh: float, iou_thresh: float,
    save_dir: Path
):
    # Prepare YOLOv7
    if config_pth:
        # Read config
        with open(config_pth, 'r') as f:
            config_str = f.read()
        config = json.loads(config_str)
        
        cls_id_to_name = {val: key for key, val in config['cls_to_id'].items()}
        num_classes = len(cls_id_to_name)
        polarized = config['polarization']
        num_channels = 4 if polarized else 3
        weights_pth = Path(
            config['work_dir']) / 'ckpts' / 'best_checkpoint.pth'
    else:
        # Set COCO parameters
        cls_id_to_name = coco_idx2label
        num_classes = len(coco_idx2label)
        num_channels = 3
    
    pad_colour = (114,) * num_channels

    # Загрузка модели
    if config_pth:
        model = load_yolo_checkpoint(weights_pth, num_classes)
    else:
        # Создать COCO модель
        model = create_yolo(num_classes)

    # Обработка семплов
    process_transforms = create_yolov7_transforms(pad_colour=pad_colour)
    normalize_transforms = A.Compose([ToTensorV2(transpose_mask=True)])

    # Start streaming
    esc_key_code = 27
    with Vimba.get_instance():
        with get_camera(camera_id) as camera:
            pix_form = camera.get_pixel_format()
            if (not polarized and pix_form != PixelFormat.BayerRG8 or
                    polarized and pix_form != PixelFormat.Mono8):
                raise ValueError(
                    'Camera should be in BayerRG8 during RGB imaging or Mono8 '
                    'during polarization imaging but it is '
                    f'{polarized=} and {pix_form}.')
            
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)

            for frame in camera.get_frame_generator(timeout_ms=3000):

                # Подготавливаем картинку
                image = frame.as_numpy_ndarray()
                if save_dir:
                    source_image = image.copy()

                if polarized:
                    image = split_raw_pol(image)  # ndarray (h, w, 4)
                else:
                    # Raw rgb
                    image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2BGR)

                # Предобработка данных
                image = process_transforms(
                    image=image, bboxes=[], labels=[])['image']
                tensor_image = normalize_transforms(image=image)['image']
                tensor_image = tensor_image.to(torch.float32) / 255
                tensor_image = tensor_image[None, ...]  # Add batch dim

                # Запуск модели
                boxes, class_ids, confidences = yolo_inference(
                    model, tensor_image, conf_thresh, iou_thresh)

                image = image_tensor_to_numpy(tensor_image)
                image = image[..., :3]
                image *= 255
                image = image.astype(np.uint8)

                labels = list(map(lambda idx: cls_id_to_name[idx],
                                  class_ids.tolist()[:30]))
                image = draw_bounding_boxes(
                    image[0],
                    boxes.tolist()[:30],
                    labels,
                    confidences.tolist()[:30])

                if save_dir:
                    src_pth = ((save_dir / f'source {datetime.now()}')
                               .with_suffix('.jpg'))
                    res_pth = ((save_dir / f'result {datetime.now()}')
                               .with_suffix('.jpg'))
                    save_image(source_image, src_pth, rgb_to_bgr=False)
                    save_image(image, res_pth, rgb_to_bgr=False)

                msg = 'YOLOv7 predict. Press <Esc> to stop stream.'
                if not polarized:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow(msg, image)
                key = cv2.waitKey(1)
                if key == esc_key_code:
                    break


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('camera_id',
                        choices=['DEV_000F315D630A', 'DEV_000F315D16C4'],
                        help=('Mako camera id. DEV_000F315D630A is RGB and '
                              'DEV_000F315D16C4 is polarization.'))
    parser.add_argument('--config_pth', type=Path, default=None,
                        help=('A path to a train config of a required model. '
                              'If is not provided then default COCO model for '
                              'RGB will be loaded.'))
    parser.add_argument('--conf_thresh', type=unit_interval, default=0.3,
                        help='A confidence threshold of model.')
    parser.add_argument('--iou_thresh', type=unit_interval, default=0.2,
                        help='An IoU threshold for bounding boxes.')
    parser.add_argument('--save_dir', type=Path, default=None,
                        help='Optional path to save results.')
    args = parser.parse_args()

    if args.config_pth is not None and not args.config_pth.exists():
        raise FileExistsError(
            f'Defined config "{str(args.config_pth)}" does not exist.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(camera_id=args.camera_id, config_pth=args.config_pth,
         conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh,
         save_dir=args.save_dir)
