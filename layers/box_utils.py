# -*- coding: utf-8 -*-
import torch
def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


# compute IOU
# compute intersect
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    overlaps = jaccard(truths, point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]  # Shape: [num_priors]，+1  0:back
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)  # encode(matches,priors,variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    return loc_t,conf_t

def encode(matched, priors, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def decode(loc, priors, variances):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loc = loc.to(device)
    priors = priors.to(device)

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes[:, 0].new()
    yy1 = boxes[:, 1].new()
    xx2 = boxes[:, 2].new()
    yy2 = boxes[:, 3].new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # torch.index_select(x1, 0, idx, out=xx1)
        # torch.index_select(y1, 0, idx, out=yy1)
        # torch.index_select(x2, 0, idx, out=xx2)
        # torch.index_select(y2, 0, idx, out=yy2)
        # xx1 = torch.clamp(xx1, min=x1[i])
        # yy1 = torch.clamp(yy1, min=y1[i])
        # xx2 = torch.clamp(xx2, max=x2[i])
        # yy2 = torch.clamp(yy2, max=y2[i])
        # w.resize_as_(xx2)
        # h.resize_as_(yy2)
        # w = xx2 - xx1
        # h = yy2 - yy1
        # w = torch.clamp(w, min=0.0)
        # h = torch.clamp(h, min=0.0)
        # inter = w*h
        # rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        # union = (rem_areas - inter) + area[i]
        # IoU = inter/union  # store result in iou
        # idx = idx[IoU.le(overlap)]
        idx= torch.autograd.Variable(idx, requires_grad=False)
        idx = idx.data
        x1 = torch.autograd.Variable(x1, requires_grad=False)
        x1 = x1.data
        y1 = torch.autograd.Variable(y1, requires_grad=False)
        y1 = y1.data
        x2 = torch.autograd.Variable(x2, requires_grad=False)
        x2 = x2.data
        y2 = torch.autograd.Variable(y2, requires_grad=False)
        y2 = y2.data
        xx1.resize_(0)
        xx2.resize_(0)
        yy1.resize_(0)
        yy2.resize_(0)
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        area = torch.autograd.Variable(area, requires_grad=False)
        area = area.data
        idx= torch.autograd.Variable(idx, requires_grad=False)
        idx = idx.data
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]#保留交并比小于阈值的预测边界框的id


    return keep, count
