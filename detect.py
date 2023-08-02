from datetime import datetime as date
from loguru import logger
from association import *
from glob import glob
import cv2
import os
import numpy as np
from edgeyolo.utils import get_color
from edgeyolo.detect import Detector
from torchvision import transforms as TR
from pose_utils.general_utils import polys_from_pose,make_parser
from tracker import byte_tracker
from builder.model_builder import build_model

"""=====================================================
Model			    Size	mAPval	mAPval	FPSAGX	Params
--------------------------------------------------------
EdgeYOLO-Tiny-LRELU	416	    33.1	50.5	206	    7.6 / 7.0
			        640	    37.8	56.7	109	
EdgeYOLO-Tiny		416	    37.2	55.4	136	    5.8 / 5.5
			        640	    41.4	60.4	67	
EdgeYOLO-S			640	    44.1	63.3	53	    9.9 / 9.3
EdgeYOLO-M			640	    47.5	66.6	46	    19.0 / 17.8
EdgeYOLO			640	    50.6	69.8	34	    41.2 / 40.5
========================================================"""

# All parameters are defined here as constants instead of using argparse
WEIGHTS = "./models/edgeyolo_tiny_lrelu_coco.pth"
CONF_THRES = 0.25
NMS_THRES = 0.55
MP = False
FP16 = True
NO_FUSE = False
INPUT_SIZE = [416, 640]
SOURCE = "0" #"E:/videos/test.avi"
TRT = False
LEGACY = False
USE_DECODER = False
BATCH = 1
NO_LABEL = False
SAVE_DIR = "./output/detect/imgs/"
FPS = 99999
IS_GPU = True
POSE_PATH = "./models/vitpose-b-multi-coco.pth"

if IS_GPU:
    import torch

def draw(imgs, results, class_names, line_thickness=4, draw_label=True):
    corner_radius = 10
    transparency = 0.3
    single = False
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
        single = True
    out_imgs = []
    tf = max(line_thickness - 1, 1)
    for img, result in zip(imgs, results):
        if result is not None:
            for *xywh, obj, conf, cls in result:
                intbox = [int(i) for i in xywh]
                color = get_color(int(cls))
                
                # Draw the corners
                cv2.ellipse(img, (intbox[0] + corner_radius, intbox[1] + corner_radius), (corner_radius, corner_radius),
                            180, 0, 90, color, thickness=line_thickness)
                cv2.ellipse(img, (intbox[2] - corner_radius, intbox[1] + corner_radius), (corner_radius, corner_radius),
                            270, 0, 90, color, thickness=line_thickness)
                cv2.ellipse(img, (intbox[0] + corner_radius, intbox[3] - corner_radius), (corner_radius, corner_radius),
                            90, 0, 90, color, thickness=line_thickness)
                cv2.ellipse(img, (intbox[2] - corner_radius, intbox[3] - corner_radius), (corner_radius, corner_radius),
                            0, 0, 90, color, thickness=line_thickness)

                # Draw the rounded rectangle
                intbox_poly = np.array([
                    [intbox[0] + corner_radius, intbox[1]],
                    [intbox[2] - corner_radius, intbox[1]],
                    [intbox[2], intbox[1] + corner_radius],
                    [intbox[2], intbox[3] - corner_radius],
                    [intbox[2] - corner_radius, intbox[3]],
                    [intbox[0] + corner_radius, intbox[3]],
                    [intbox[0], intbox[3] - corner_radius],
                    [intbox[0], intbox[1] + corner_radius],
                ])
                sub_img = img.copy()
                cv2.fillPoly(sub_img, [intbox_poly], color=color)
                img = cv2.addWeighted(sub_img, transparency, img, 1-transparency, 0)

                if draw_label:
                    label = f'{class_names[int(cls)]} {obj * conf:.2f}'
                    t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
                    c2 = intbox[0] + t_size[0], intbox[1] - t_size[1] - 3
                    # cv2.rectangle(img, (intbox[0], intbox[1]), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (intbox[0], intbox[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        out_imgs.append(img)
    return out_imgs[0] if single else out_imgs

def pose_points(detector, image, pose, tracker):    
    transform = TR.Compose([
        TR.ToPILImage(),
        TR.Resize((256, 192)),  # (height, width)
        TR.ToTensor(),
        TR.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    detections = detector(image)

    # Only keep detections where class ID (in the last column) is 0 (person)
    TARGETNUM = [0]
    indices = (detections[:,-1] == TARGETNUM).nonzero(as_tuple=True)[0]
    dets = detections[indices]
    
    online_targets = tracker.update(dets, [image.shape[0],image.shape[1]], image.shape)

    online_tlwhs = []
    online_ids = []
    online_scores = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        if tlwh[2] * tlwh[3] > 10:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)

    device = 'cuda'
    nof_people = len(online_ids) if online_ids is not None else 0
    boxes = torch.empty((nof_people, 4), dtype=torch.int32, device=device)
    images = torch.empty((nof_people, 3, 256, 192))  # (height, width)
    heatmaps = np.zeros((nof_people, 17, 64, 48), dtype=np.float32)

    for i, (x1, y1, w, h) in enumerate(online_tlwhs):
        x1, x2, y1, y2 = adjust_box(x1, y1, w, h, image.shape)
        image_crop, boxes[i] = adjust_image_and_boxes(image, x1, y1, x2, y2, transform, device)

    if images.shape[0] > 0:
        images = images.to(device)
        with torch.no_grad():
            out = pose(images)
        pts = calculate_points(out, boxes, device)
    else:
        pts = np.empty((0, 0, 3), dtype=np.float32)
        online_tlwhs, online_ids, online_scores = [], [], []
    
    return [pts] if len(pts) > 1 else pts, online_tlwhs, online_ids, online_scores

def adjust_box(x1, y1, w, h, shape):
    x1 = x1.astype(np.int32)
    x2 = x1 + w.astype(np.int32)
    y1 = y1.astype(np.int32)
    y2 = y1 + h.astype(np.int32)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(shape[1]-1, x2)
    y2 = min(shape[0]-1, y2)
    return x1, x2, y1, y2

def adjust_image_and_boxes(image, x1, y1, x2, y2, transform, device):
    correction_factor = 256 / 192 * (x2 - x1) / (y2 - y1)
    image_crop = image[y1:y2, x1:x2, ::-1]
    if correction_factor > 1:
        y1, y2 = adjust_y(y1, y2, correction_factor)
    else:
        x1, x2 = adjust_x(x1, x2, correction_factor)
    image_crop = transform(image_crop)
    boxes = torch.tensor([x1, y1, x2, y2], device=device)
    return image_crop, boxes

def adjust_y(y1, y2, correction_factor):
    center = y1 + (y2 - y1) // 2
    length = int(round((y2 - y1) * correction_factor))
    return int(center - length // 2), int(center + length // 2)

def adjust_x(x1, x2, correction_factor):
    center = x1 + (x2 - x1) // 2
    length = int(round((x2 - x1) * 1 / correction_factor))
    return int(center - length // 2), int(center + length // 2)

def calculate_points(out, boxes, device):
    pts = torch.empty((out.shape[0], out.shape[1], 3), dtype=torch.float32, device=device)
    dim1= torch.tensor(1. / 64, device=device)
    dim2= torch.tensor(1. / 48, device=device)
    (b, indices) = torch.max(out, dim=2)
    (b, indices) = torch.max(b, dim=2)
    (c, indicesc) = torch.max(out, dim=3)
    (c, indicesc) = torch.max(c, dim=2)
    for i in range(0, out.shape[0]):
        pts[i, :, 0] = indicesc[i, :] * dim1 * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
        pts[i, :, 1] = indices[i, :] * dim2 * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
        pts[i, :, 2] = c[i, :]
    return pts.cpu().numpy()

def detect_single():
    import time
    exist_save_dir = os.path.isdir(SAVE_DIR)

    # detector setup
    detector = Detector
    detect = detector(
        weight_file=WEIGHTS,
        conf_thres=CONF_THRES,
        nms_thres=NMS_THRES,
        input_size=INPUT_SIZE,
        fuse=not NO_FUSE,
        fp16=FP16,
        use_decoder=USE_DECODER
    )

    tracker = byte_tracker.BYTETracker(make_parser().parse_known_args()[0],frame_rate=30)
    pose_estimator = build_model('ViTPose_base_coco_256x192', POSE_PATH)

    # source loader setup
    if os.path.isdir(SOURCE):

        class DirCapture:

            def __init__(self, dir_name):
                self.imgs = []
                for img_type in ["jpg", "png", "jpeg", "bmp", "webp"]:
                    self.imgs += sorted(glob(os.path.join(dir_name, f"*.{img_type}")))

            def isOpened(self):
                return bool(len(self.imgs))

            def read(self):
                print(self.imgs[0])
                now_img = cv2.imread(self.imgs[0])
                self.imgs = self.imgs[1:]
                return now_img is not None, now_img

        source = DirCapture(SOURCE)
        delay = 0
    else:
        source = cv2.VideoCapture(int(SOURCE) if SOURCE.isdigit() else SOURCE)
        delay = 1

    all_dt = []
    dts_len = 300 // BATCH
    success = True

    # start inference
    count = 0
    t_start = time.time()
    while source.isOpened() and success:

        frames = []
        for _ in range(BATCH):
            success, frame = source.read()
            if not success:
                if not len(frames):
                    cv2.destroyAllWindows()
                    break
                else:
                    while len(frames) < BATCH:
                        frames.append(frames[-1])
            else:
                frames.append(frame)

        if not len(frames):
            break

        results = detect(frames, LEGACY)
        print(results)
        dt = detect.dt
        all_dt.append(dt)
        if len(all_dt) > dts_len:
            all_dt = all_dt[-dts_len:]
        print(f"\r{dt * 1000 / BATCH:.1f}ms  "
              f"average:{sum(all_dt) / len(all_dt) / BATCH * 1000:.1f}ms", end="      ")

        key = -1

        # [print(result.shape) for result in results]

        imgs = draw(frames, results, detect.class_names, 2, draw_label=not NO_LABEL)
        # print([im.shape for im in frames])
        for img in imgs:
            # print(img.shape)
            cv2.imshow("EdgeYOLO result", img)
            count += 1

            key = cv2.waitKey(delay)
            if key in [ord("q"), 27]:
                break
            elif key == ord(" "):
                delay = 1 - delay
            elif key == ord("s"):
                if not exist_save_dir:
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    exist_save_dir = True
                file_name = f"{str(date.now()).split('.')[0].replace(':', '').replace('-', '').replace(' ', '')}.jpg"
                cv2.imwrite(os.path.join(SAVE_DIR, file_name), img)
                logger.info(f"image saved to {file_name}.")
        if key in [ord("q"), 27]:
            cv2.destroyAllWindows()
            break

    logger.info(f"\ntotal frame: {count}, total average latency: {(time.time() - t_start) * 1000 / count - 1}ms")

if __name__ == '__main__':
    detect_single()