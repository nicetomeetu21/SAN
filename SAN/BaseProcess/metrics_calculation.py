import torch

from monai.metrics import compute_hausdorff_distance
def cal_metrics(pred_out, pred_label, metric_dict):
    if 'acc' not in metric_dict: metric_dict['acc'] = []
    if 'tpr' not in metric_dict: metric_dict['tpr'] = []
    if 'tnr' not in metric_dict: metric_dict['tnr'] = []
    if 'miou' not in metric_dict: metric_dict['miou'] = []
    if 'dice' not in metric_dict: metric_dict['dice'] = []
    if 'hd' not in metric_dict: metric_dict['hd'] = []

    TP = ((pred_out > 0.5) & (pred_label > 0.5)).sum()
    TN = ((pred_out <= 0.5) & (pred_label <= 0.5)).sum()
    FN = ((pred_out <= 0.5) & (pred_label > 0.5)).sum()
    FP = ((pred_out > 0.5) & (pred_label <= 0.5)).sum()

    acc = (TP + TN) / (TP + FP + FN + TN)
    tpr = TP / (TP + FN)
    tnr = TN / (FP + TN)
    miou = TP / (TP + FN + FP)
    dice = 2*TP / (2*TP + FN + FP)
    # print(TP,TN, FN, FP, TP+TN+FN+FP)
    hd = cal_hd(pred_out.clone(), pred_label.clone())
    metric_dict['acc'].append(acc.item())
    metric_dict['tpr'].append(tpr.item())
    metric_dict['tnr'].append(tnr.item())
    metric_dict['miou'].append(miou.item())
    metric_dict['hd'].append(hd.item())
    metric_dict['dice'].append(dice.item())

def cal_hd(pred, label):
    # pred //= 255
    # label //= 255
    # pred = pred[np.newaxis, np.newaxis,:,:,:]
    # label = label[np.newaxis, np.newaxis,:,:,:]
    pred = pred.unsqueeze(0)
    pred = (pred > 0.5).float()
    label = label.unsqueeze(0)
    # print(torch.max(pred), torch.max(label))
    # print(pred.shape, label.shape)
    return compute_hausdorff_distance(
        y_pred=pred,y=label,percentile=95
    )

def cal_masked_metrics(pred_out, pred_label, mask, metric_dict):
    if 'macc' not in metric_dict: metric_dict['macc'] = []
    if 'mtpr' not in metric_dict: metric_dict['mtpr'] = []
    if 'mtnr' not in metric_dict: metric_dict['mtnr'] = []
    if 'mmiou' not in metric_dict: metric_dict['mmiou'] = []
    if 'mdice' not in metric_dict: metric_dict['mdice'] = []
    if 'mhd' not in metric_dict: metric_dict['mhd'] = []

    TP = ((pred_out > 0.5) & (pred_label > 0.5))
    TN = ((pred_out <= 0.5) & (pred_label <= 0.5))
    FN = ((pred_out <= 0.5) & (pred_label > 0.5))
    FP = ((pred_out > 0.5) & (pred_label <= 0.5))
    TP = (TP.float()*mask).sum()
    TN = (TN.float()*mask).sum()
    FN = (FN.float()*mask).sum()
    FP = (FP.float()*mask).sum()
    # print(TP,TN, FN, FP)
    # print(mask.sum(), TP+TN+FN+FP)
    acc = (TP + TN) / (TP + FP + FN + TN)
    tpr = TP / (TP + FN)
    tnr = TN / (FP + TN)
    miou = TP / (TP + FN + FP)
    dice = 2*TP / (2*TP + FN + FP)

    metric_dict['macc'].append(acc.item())
    metric_dict['mtpr'].append(tpr.item())
    metric_dict['mtnr'].append(tnr.item())
    metric_dict['mmiou'].append(miou.item())
    metric_dict['mdice'].append(dice.item())
    # print(torch.max(mask))
    # mask //= 255
    hd = cal_hd(pred_out.clone()*mask, pred_label.clone()*mask)
    metric_dict['mhd'].append(hd.item())