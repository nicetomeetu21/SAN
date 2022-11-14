# SAN
code for "Choroidal Vessel Segmentation in SD-OCT with 3D Shape-aware Adversarial Networks"

The following data parameters are required for training and testing:

```python
# in ./SAN/BaseProcess/default_setting.py 
parser.add_argument('--image_root', default='path/to/OCT')
parser.add_argument('--label_root', default='path/to/labels')
parser.add_argument('--shape_label_root', default='path/to/shape_labels')
parser.add_argument('--region_mask_root', type=str, default='path/to/choroidal layer mask')
parser.add_argument('--test_data_lists',default=[['test names of fold 1'],['test names of fold 2'],['test names of fold 3']])
parser.add_argument('--result_root', type=str, default='path/to/result')
```
