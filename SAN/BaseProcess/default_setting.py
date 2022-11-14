# -*- coding:utf-8 -*-

def add_server_specific_setting(parser):
    device_dict = {
        0: server0_config,
    }
    temp_args, _ = parser.parse_known_args()
    server_id = temp_args.server_id
    dataset = temp_args.dataset
    return device_dict[server_id](parser, dataset)

def server0_config(parser, dataset):
    if dataset == 'dataset1':
        parser.add_argument('--image_root',
                            default'path/to/OCT')
        parser.add_argument('--label_root',
                            default='path/to/labels')
        parser.add_argument('--region_mask_root', type=str,
                            default='path/to/choroidal layer mask')
        parser.add_argument('--test_data_lists',default=[['test names of fold 1'],['test names fold 2'],['test names fold 3']])
        parser.add_argument('--result_root', type=str, default='path/to/result')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--enable_progress_bar', default=False)
    parser.add_argument('--num_sanity_val_steps', default=-1)
   # parser.add_argument('--strategy', default="ddp_spawn")

