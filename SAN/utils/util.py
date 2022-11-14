import os
from torchvision.utils import save_image
def find_full_name(target_dir, name_part):
    names = os.listdir(target_dir)
    for name in names:
        if name_part in name: return name

def gen_visual_imgs(img_list):
    imgs = []
    for j in range(min(2, img_list[0].shape[2])):
        for img in img_list:
            imgs.append(img[..., j, :, :])
    return imgs

import json
def save_metrics_to_json(test_metric_dict, global_iter, result_dir, json_name='single_fold_result.json'):
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    json_path = os.path.join(result_dir, json_name)
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        results = dict()
    # print(results)
    def write_to_result(results_dict, iter, metric_dict):
        # print(iter, results_dict)
        iter = str(iter)
        if iter not in results_dict.keys(): results_dict[iter] = dict()
        # print(iter, results_dict)
        for k in metric_dict.keys():
            if k not in results_dict[iter]: results_dict[iter][k] = []
            results_dict[iter][k] += metric_dict[k]

    write_to_result(results, global_iter, test_metric_dict)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

def save_cube_from_tensor(img, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    for j in range(img.shape[2]):
        img_path = os.path.join(result_dir, str(j + 1) + '.png')
        save_image(img[j, :, :], img_path)


import sys
import torch
def load_network(network, save_path=''):
    if not os.path.isfile(save_path):
        print('%s not exists yet!' % save_path)
        raise ('Generator must exist!')
    else:
        # network.load_state_dict(torch.load(save_path))
        try:
            network.load_state_dict(torch.load(save_path)['state_dict'])
        except:
            pretrained_dict = torch.load(save_path)['state_dict']
            model_dict = network.state_dict()
            try:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                network.load_state_dict(pretrained_dict)
                print(
                        'Pretrained network has excessive layers; Only loading layers that are used')
            except:
                print(
                    'Pretrained network has fewer layers; The following are not initialized:')
                for k, v in pretrained_dict.items():
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v

                if sys.version_info >= (3, 0):
                    not_initialized = set()
                else:
                    from sets import Set
                    not_initialized = Set()

                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized.add(k.split('.')[0])

                print(sorted(not_initialized))
                network.load_state_dict(model_dict)



def find_model_by_iter(model_dir, target_iter):
    names = os.listdir(model_dir)
    for name in names:
        if target_iter in name: return os.path.join(model_dir, name)