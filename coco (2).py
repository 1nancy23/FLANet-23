import torch
import os
import torchvision
import numpy as np
import pycocotools
# import voc2coco
import pickle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from data import VOC_CLASSES as labelmap
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from SMENet import build_SMENet
import argparse
from torch.autograd import Variable
import json
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re
from data import VOC_CLASSES as labels
import time


parser = argparse.ArgumentParser(
    description='SMENet Detector Evaluation')
set_type = 'trainval'
#数据集voc转coco参数
parser.add_argument('--ann_dir', type=str, default="D:/TesT_Code2/SMENet/SMENet/VOCdevkit/VOC2012/Annotations",
                    help='path to annotation files directory. It is not need when use --ann_paths_list')

parser.add_argument('--ann_ids', type=str, default=f"D:/TesT_Code2/SMENet/SMENet/VOCdevkit/VOC2012/ImageSets/Main/{set_type}.txt",
                    help='path to annotation files ids list. It is not need when use --ann_paths_list')
parser.add_argument('--ann_paths_list', type=str, default=None,
                    help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
parser.add_argument('--output', type=str, default='cocoresult/voc_tococo.json', help='path to output json file')
parser.add_argument('--ext', type=str, default='xml', help='additional extension of annotation file')
parser.add_argument('--extract_num_from_imgid', action="store_true",
                    help='Extract image number from the image filename')
#coco评价指标
parser.add_argument('--trained_model',
                    default='./weights/fuben_trainval_RSODT6.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.2, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=10, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root',
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--result_file_path', default="D:/TesT_Code2/SMENet/SMENet/SMENet/cocoresult/", type=str,
                    help='结果文件路径')
args = parser.parse_args()


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def cocozhibiao(coco_data_path):
    result_file_path = args.result_file_path
    # 定义变换函数
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    num_classes = len(labelmap) + 1                      # +1 for background
    model = build_SMENet('test', 400, num_classes)
    # load data
    dataset = VOCDetection(args.voc_root, [('2012', set_type)],
                           BaseTransform(400),
                           VOCAnnotationTransform())
    weights_path = args.trained_model
    weights_dict=torch.load(weights_path)
    model.load_state_dict(weights_dict, strict=False)
    print('Finished loading model!')
    if args.cuda:
        model = model.cuda()




    # 遍历所有图片
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # 创建一个空的json列表
    json_list = []
    _t = Timer()
    avg_det_time=0
    for i in range(num_images):
        im, gt, h, w ,img_id = dataset.pull_item(i)
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t.tic()
        ne = model(x)
        detect_time = _t.toc(average=False)
        avg_det_time += detect_time
        detections = ne.data
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]   #each label: [200, 5]
            mask = dets[:, 0].gt(0.5).expand(5, dets.size(0)).t()   # dets[:, 0] is score
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)

            # 将每个检测框和分数添加到json列表中，注意要按照COCOeval()的格式
            for k in range(cls_dets.shape[0]):
                json_dict = {}
                json_dict['image_id'] = img_id # 图像的id，从0开始
                json_dict['category_id'] = j # 类别的id，从1开始
                json_dict['bbox'] = cls_dets[k][:4].tolist() # 检测框的坐标，[x,y,w,h]
                json_dict['score'] = float(cls_dets[k][4]) # 检测框的分数
                json_list.append(json_dict)
        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))
    print('Average Predict Time: {:.3f}s'.format(avg_det_time/num_images))

    # 将json列表写入json文件
    result_file_path = os.path.join(result_file_path, 'all_boxes.json')
    with open(result_file_path, 'w') as f:
        json.dump(json_list, f)

    IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
    # 使用pycocoevalcap库来评估结果文件，并打印每个评价指标的得分



    # 加载结果文件
    coco = COCO(coco_data_path)
    cocoDt = coco.loadRes(result_file_path)
    coco_eval = COCOeval(coco, cocoDt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 获取指标
    stats = coco_eval.stats
    # 定义指标名称
    metrics = ["Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] ", 
               "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] ", 
               "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] ", 
               "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] ", 
               "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] ", 
               "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] ", 
               "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] ", 
               "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] ", 
               "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] ", 
               "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] ", 
               "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] ", 
               "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] "]
    # 打开一个txt文件
    with open("cocoresult/results.txt", "a") as f:
        f.write(f"trained_model:{args.trained_model}    set_type:{set_type}\n")
        # 遍历指标
        for i in range(len(stats)):
            # 写入指标名称和值
            f.write(f"{metrics[i]}= {stats[i]:.3f}\n")
        f.write(f"\n")
        # 关闭文件
        f.close()
        
    # for c in range(len(labelmap)):
    #     # Set the category id
    #     coco_eval.params.catIds = [c]
    #     print(coco_eval.params.catIds)
    #     # Evaluate the results
    #     coco_eval.evaluate()
    #     coco_eval.accumulate()
        # # Print the AP for each IOU threshold and the average AP
        # print(f"Category {c}:")
        # for i in range(len(IOU_THRESHOLDS)):
        #     print(f"AP at IOU {IOU_THRESHOLDS[i]}: {coco_eval.eval['recall'][i, :, :].mean()}")
        # print(f"Average AP: {coco_eval.eval['recall'][:, :, :].mean()}")

    # Evaluate the results for all categories
    coco_eval.params.catIds = list(range(len(labelmap)))

    # Evaluate the results
    coco_eval.evaluate()
    coco_eval.accumulate()

    # # Print the AP for each IOU threshold and the average AP
    # print("All categories:")
    # for i in range(len(IOU_THRESHOLDS)):
    #     print(f"AP at IOU {IOU_THRESHOLDS[i]}: {coco_eval.eval['recall'][i].mean()}")
    # print(f"Average AP: {coco_eval.eval['recall'].mean()}")







def get_label2id() -> Dict[str, int]:
    """id is 1 start"""
    labels_str = labels
    labels_ids = list(range(1, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    # If use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotaion ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid+ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(jpg_filname,annotation_root, extract_num_from_imgid=True):
    filename=jpg_filname
    img_id = os.path.splitext(jpg_filname)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) 
    ymin = int(float(bndbox.findtext('ymin'))) 
    xmax = int(float(bndbox.findtext('xmax'))) 
    ymax = int(float(bndbox.findtext('ymax'))) 
    assert xmax >= xmin and ymax >= ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    ann = {
        'area': (xmax - xmin) * (ymax - ymin),
        'iscrowd': 0,
        'bbox': [xmin, ymin, xmax, ymax],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        jpg_filname="".join([os.path.splitext(os.path.basename(a_path))[0],".jpg"])
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(jpg_filname,annotation_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)
    return output_jsonpath


def main():
    label2id = get_label2id()
    ann_paths = get_annpaths(
        ann_dir_path=args.ann_dir,
        ann_ids_path=args.ann_ids,
        ext=args.ext,
        annpaths_list_path=args.ann_paths_list
    )
    coco_data_path=convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=args.output,
        extract_num_from_imgid=args.extract_num_from_imgid
    )
    print("VOC to COCO Complete!(bbox no change)")
    cocozhibiao(coco_data_path)


if __name__ == '__main__':
    main()
