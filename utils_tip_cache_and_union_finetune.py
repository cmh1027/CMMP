"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from code import interact
from fileinput import filename
from locale import normalize
import os
import torch
import pickle
import numpy as np
import scipy.io as sio
import json

from torchvision.transforms import Resize, CenterCrop

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet
from gta.gta import GTA
from hico_text_label import hico_unseen_index, hico_text_label, hico_obj_text_label
from hico_list import hico_verb_object_list
import sys
sys.path.append('../pocket/pocket')
import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

import sys
sys.path.append('detr')
import detr.datasets.transforms_clip as T
import pdb
import copy 
import pickle
import torch.nn.functional as F
import clip
from PIL import Image
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchvision.transforms import PILToTensor
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont


def custom_collate(batch):
    images = []
    targets = []
    
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
    return images, targets

def draw_bounding_boxes(image, box_h, box_o, objects, verbs, mapping_obj, mapping_verb, src_size, dst_size, path):
    """
    Draws bounding boxes and labels on a given image.

    Args:
        image (torch.Tensor): The image tensor of shape (C, H, W) with values in [0, 1].
        boxes (torch.Tensor): Bounding box tensor of shape (N, 4), where each box is [x_min, y_min, x_max, y_max].
        labels (torch.Tensor): Labels tensor of shape (N,), containing the label for each bounding box.
    """
    
    pil_image = to_pil_image(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default(size=35)
    for bh, bo, obj, v in zip(box_h, box_o, objects, verbs):
        obj = mapping_obj[obj.item()]
        verb = mapping_verb[v.item()]
        bh[..., [0, 2]] *= (dst_size[1] / src_size[1])
        bh[..., [1, 3]] *= (dst_size[0] / src_size[0])
        bo[..., [0, 2]] *= (dst_size[1] / src_size[1])
        bo[..., [1, 3]] *= (dst_size[0] / src_size[0])
        bh = bh.tolist()
        bo = bo.tolist()
        x_min, y_min, x_max, y_max = bh
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        x_min, y_min, x_max, y_max = bo
        draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=2)
        text_bbox = draw.textbbox((x_min, y_min), obj, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x_min, y_min - text_height, x_min + text_width, y_min], fill="blue")
        draw.text((x_min, y_min - text_height), obj, fill="white", font=font)

        bh_center = ((bh[0]+bh[2])/2, (bh[1]+bh[3])/2)
        bo_center = ((bo[0]+bo[2])/2, (bo[1]+bo[3])/2)
        draw.line([bh_center, bo_center], width = 2, fill='red')
        interaction = ((bh_center[0] + bo_center[0])/2, (bh_center[1] + bo_center[1])/2)
        x_min, y_min = interaction[0], interaction[1]
        text_bbox = draw.textbbox(interaction, verb, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x_min, y_min - text_height, x_min + text_width, y_min], fill="green")
        draw.text((x_min, y_min - text_height), verb, fill="white", font=font)

    
    pil_image.save(path)


class DataFactory(Dataset):
    def __init__(self, name, partition, data_root, image_path, clip_model_name, zero_shot=False, zs_type='rare_first', num_classes=600, detr_backbone="R50"): ##ViT-B/16, ViT-L/14@336px
        if name not in ['hicodet', 'gta', 'vcoco']:
            raise ValueError("Unknown dataset ", name)
        assert clip_model_name in ['ViT-L/14@336px', 'ViT-B/16', 'ViT-L/14'], "Unknown CLIP model " + clip_model_name
        self.clip_model_name = clip_model_name
        if self.clip_model_name in ['ViT-B/16', 'ViT-L/14']:
            self.clip_input_resolution = 224
        elif self.clip_model_name == 'ViT-L/14@336px':
            self.clip_input_resolution = 336
        
        if name == 'hicodet':
            self.dataset = HICODet(
                root=os.path.join(data_root, image_path, partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        elif name == "gta":
            self.dataset = GTA(
                root=os.path.join(data_root, image_path, partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            assert not zero_shot
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        # add clip normalization
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
            ])
            self.clip_transforms = T.Compose([
                T.IResize([self.clip_input_resolution,self.clip_input_resolution]),
            ])
        else:   
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
            ])
            self.clip_transforms = T.Compose([
                T.IResize([self.clip_input_resolution,self.clip_input_resolution]),
            ])
        
        self.partition = partition
        self.name = name
        self.count=0
        self.zero_shot = zero_shot
        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            self.zs_type = zs_type
            self.filtered_hoi_idx = hico_unseen_index[self.zs_type]

        device = "cuda"
        _, self.process = clip.load(self.clip_model_name, device=device)

        self.keep = [i for i in range(len(self.dataset))]

        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            self.zs_keep = []
            self.remain_hoi_idx = [i for i in np.arange(600) if i not in self.filtered_hoi_idx]

            for i in self.keep:
                (image, target), filename = self.dataset[i]
                mutual_hoi = set(self.remain_hoi_idx) & set([_h.item() for _h in target['hoi']])
                if len(mutual_hoi) != 0:
                    self.zs_keep.append(i)
            self.keep = self.zs_keep

            self.zs_object_to_target = [[] for _ in range(self.dataset.num_object_cls)]
            if num_classes == 600:
                for corr in self.dataset.class_corr:
                    if corr[0] not in self.filtered_hoi_idx:
                        self.zs_object_to_target[corr[1]].append(corr[0])
            else:
                for corr in self.dataset.class_corr:
                    if corr[0] not in self.filtered_hoi_idx:
                        self.zs_object_to_target[corr[1]].append(corr[2])
        
        
    def __len__(self):
        return len(self.keep)

    # train detr with roi
    def __getitem__(self, i):
        (image, target), filename = self.dataset[self.keep[i]]
        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            _boxes_h, _boxes_o, _hoi, _object, _verb = [], [], [], [], []
            for j, hoi in enumerate(target['hoi']):
                if hoi in self.filtered_hoi_idx:
                    continue
                _boxes_h.append(target['boxes_h'][j])
                _boxes_o.append(target['boxes_o'][j])
                _hoi.append(target['hoi'][j])
                _object.append(target['object'][j])
                _verb.append(target['verb'][j])           
            target['boxes_h'] = torch.stack(_boxes_h)
            target['boxes_o'] = torch.stack(_boxes_o)
            target['hoi'] = torch.stack(_hoi)
            target['object'] = torch.stack(_object)
            target['verb'] = torch.stack(_verb)
        w,h = image.size
        target['orig_size'] = torch.tensor([h,w])
        if self.name in ['hicodet', 'gta']:
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')
            ## TODO add target['hoi']

        image_original = image
        image, target = self.transforms(image, target)
        image_clip, target = self.clip_transforms(image, target)  
        image, _ = self.normalize(image, None)
        image_clip, target = self.normalize(image_clip, target)
        target['filename'] = filename
        return (image,image_clip, PILToTensor()(image_original).to(image.device)/255), target

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def get_region_proposals(self, results,image_h, image_w):
        human_idx = 0
        min_instances = 3
        max_instances = 15
        region_props = []
        bx = results['ex_bbox']
        sc = results['ex_scores']
        lb = results['ex_labels']
        hs = results['ex_hidden_states']
        is_human = lb == human_idx
        hum = torch.nonzero(is_human).squeeze(1)
        obj = torch.nonzero(is_human == 0).squeeze(1)
        n_human = is_human.sum(); n_object = len(lb) - n_human
        # Keep the number of human and object instances in a specified interval
        device = torch.device('cpu')
        if n_human < min_instances:
            keep_h = sc[hum].argsort(descending=True)[:min_instances]
            keep_h = hum[keep_h]
        elif n_human > max_instances:
            keep_h = sc[hum].argsort(descending=True)[:max_instances]
            keep_h = hum[keep_h]
        else:
            # keep_h = torch.nonzero(is_human[keep]).squeeze(1)
            # keep_h = keep[keep_h]
            keep_h = hum

        if n_object < min_instances:
            keep_o = sc[obj].argsort(descending=True)[:min_instances]
            keep_o = obj[keep_o]
        elif n_object > max_instances:
            keep_o = sc[obj].argsort(descending=True)[:max_instances]
            keep_o = obj[keep_o]
        else:
            # keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
            # keep_o = keep[keep_o]
            keep_o = obj

        keep = torch.cat([keep_h, keep_o])

        boxes=bx[keep]
        scores=sc[keep]
        labels=lb[keep]
        hidden_states=hs[keep]
        is_human = labels == human_idx
            
        n_h = torch.sum(is_human); n = len(boxes)
        # Permute human instances to the top
        if not torch.all(labels[:n_h]==human_idx):
            h_idx = torch.nonzero(is_human).squeeze(1)
            o_idx = torch.nonzero(is_human == 0).squeeze(1)
            perm = torch.cat([h_idx, o_idx])
            boxes = boxes[perm]; scores = scores[perm]
            labels = labels[perm]; unary_tokens = unary_tokens[perm]
        # Skip image when there are no valid human-object pairs
        if n_h == 0 or n <= 1:
            print(n_h, n)

        # Get the pairwise indices
        x, y = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device)
        )
        # pdb.set_trace()
        # Valid human-object pairs
        x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
        sub_boxes = boxes[x_keep]
        obj_boxes = boxes[y_keep]
        lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
        rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
        union_boxes = torch.cat([lt,rb],dim=-1)
        sub_boxes[:,0].clamp_(0, image_w)
        sub_boxes[:,1].clamp_(0, image_h)
        sub_boxes[:,2].clamp_(0, image_w)
        sub_boxes[:,3].clamp_(0, image_h)

        obj_boxes[:,0].clamp_(0, image_w)
        obj_boxes[:,1].clamp_(0, image_h)
        obj_boxes[:,2].clamp_(0, image_w)
        obj_boxes[:,3].clamp_(0, image_h)

        union_boxes[:,0].clamp_(0, image_w)
        union_boxes[:,1].clamp_(0, image_h)
        union_boxes[:,2].clamp_(0, image_w)
        union_boxes[:,3].clamp_(0, image_h)
    

        # return sub_boxes.int(), obj_boxes.int(), union_boxes.int()
        return sub_boxes, obj_boxes, union_boxes

    def get_union_mask(self, bbox, image_size):
        n = len(bbox)
        masks = torch.zeros
        pass

class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

def get_flop_stats(model, data_loader):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.
    Returns:
        float: the total number of gflops of the given model.
    """
    inputs = _get_model_analysis_input(data_loader)
    flops = FlopCountAnalysis(model, inputs)
    print("Total FLOPs(G)", flops.total() / 1e9)
    print(flop_count_table(flops, max_depth=4, show_param_shapes=False))
    return flops

def _get_model_analysis_input(data_loader):
    for batch in data_loader:
        inputs = pocket.ops.relocate_to_cuda(batch[0])
        inputs = (inputs,batch[1])
        return inputs

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes

        if net.dataset == "hicodet":
            objs = list(map(lambda s:s.replace("a photo of a ", "").replace("a photo of an ", "").replace("a photo of ", ""), [obj[1] for obj in hico_obj_text_label]))
            objs.remove("nothing")
            self.verbs = list(map(lambda k:k[0], hico_verb_object_list))
        elif net.dataset == "gta":
            from gta.gta_to_hico import object_list, valid_interactions
            objs = object_list
            self.verbs = list(map(lambda k:k[0], valid_interactions))
        else:
            raise NotImplementedError
        
        text_to_idx = {val:i for i, val in enumerate(objs)}
        self.idx_to_text = {val:key for key, val in text_to_idx.items()}

        self._net = net


    def _on_each_iteration(self):
        loss_dict = self._state.net(*self._state.inputs, targets=self._state.targets)
        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")
        
        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()
        
        total_iter = len(self._train_loader.dataset)
        print(f"Epoch {self._state.epoch} : {self._state.iteration % total_iter} / {total_iter}", end=f"      \r")

       
    @torch.no_grad()
    def sam_cache(self, dataloader, save_root):
        dataset = dataloader.dataset.dataset
        
        for iteration, batch in tqdm(enumerate(dataloader), total=len(dataset), desc="caching..."):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            boxes, confidences, labels = self._net.cache_sam(inputs,batch[1])
            name = batch[1][0]['filename'][:-4]
            with open(os.path.join(save_root, name+".pkl"), "wb") as f:
                pickle.dump([boxes, confidences, labels], f)

    @torch.no_grad()
    def test_hico(self, dataloader, args=None):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset

        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        if args.dataset == "hicodet":
            tgt_num_classes = 600
        elif args.dataset == "gta":
            tgt_num_classes = 12
        else:
            raise NotImplementedError

        num_gt = dataset.anno_interaction if args.dataset in ["hicodet", "gta"] else None
        meter = DetectionAPMeter(
            tgt_num_classes, nproc=1,
            num_gt=num_gt,
            algorithm='11P'
        )
        total_times = []
        for iteration, batch in tqdm(enumerate(dataloader), total=len(dataset)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            time_start = time.time()
            outputs = net(inputs,batch[1])
            # debug
            if outputs is not None:
                src_size = outputs[0]['src_size'].tolist()
                dst_size = outputs[0]['dst_size']
                sam = outputs[0]['sam']
            time_end = time.time()
            total_times.append(time_end - time_start)
            # Skip images without detections
            if outputs is None or len(outputs) == 0:
                continue
            for output, target in zip(outputs, batch[-1]):
                output = pocket.ops.relocate_to_cpu(output, ignore=True)
                # Format detections
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
                objects = output['objects']
                scores = output['scores']
                verbs = output['labels']
                if net.module.num_classes==117 or net.module.num_classes==407:
                    interactions = conversion[objects, verbs]
                else:
                    interactions = verbs
                # Recover target box scale
                gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])
                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)
                unique_hoi = interactions.unique()
                for hoi_idx in unique_hoi:
                    gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                    det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                            gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                            boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )
                
                # debug
                # if outputs is not None and (labels==1).sum() > 0:
                #     exp_name = "sam" if sam else "orig"
                #     draw_bounding_boxes(outputs[0]["image"], boxes_h[labels==1], boxes_o[labels==1], objects[labels==1], interactions[labels==1].long(), self.idx_to_text, self.verbs, src_size, dst_size, f"log/gta/{exp_name}/{'%05d'% iteration}.png")

                
                # draw_bounding_boxes(outputs[0]["image"], boxes_h[labels==1], boxes_o[labels==1], objects[labels==1], interactions[labels==1].long(), self.idx_to_text, self.verbs, src_size, dst_size, f"log/{args.dataset}/{'%05d'% iteration}.png")
                meter.append(scores, interactions, labels)   # scores human*object*verb, interaction（600), labels
        time_sum = 0
        for i in total_times:
            time_sum += i
        print("FPS: %f"%(1.0/(time_sum/len(total_times))))
        print(len(total_times))
        return meter.eval()

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
        
        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir) 
        print('saving cache.pkl to', cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)
