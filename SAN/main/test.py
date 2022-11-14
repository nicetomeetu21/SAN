import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torchio as tio

from BaseProcess.default_setting import add_server_specific_setting
from BaseProcess.init_experiment import initExperiment
from datamodule.multi_tio_3D_datamodule import MultiTioDatamodule
from networks.unet_3d_fix import UNet_2out
from utils.util import save_cube_from_tensor,find_model_by_iter, load_network


class LitModel(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.netS = UNet_2out(**vars(opts))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("UNet3D")
        # train args
        parser.add_argument("--max_steps", type=int, default=100000)
        parser.add_argument("--val_check_interval", type=int, default=2000)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument('--train_per_size', default=(5, 400, 400))
        parser.add_argument('--test_per_size', default=(5, 640, 400))
        parser.add_argument('--test_overlap', default=(2, 0, 0))
        parser.add_argument('--queue_length', default=150)
        parser.add_argument('--samples_per_volume', default=20)
        # model args
        parser.add_argument("--n_in", type=int, default=1)
        parser.add_argument("--n_out", type=int, default=1)
        parser.add_argument("--first_channels", default=64)
        return parent_parser

    def forward(self, img_input):
        pred,_ = self.netS(img_input)
        return pred

    def test_step(self, batch, batch_idx):
        image, label, region_mask, name = batch['image']['data'], batch['label']['data'], batch['region_mask']['data'], \
                                          batch['name']
        pred = self.inference(image)
        prb_out = torch.sigmoid(pred)
        bin_out = (prb_out > 0.5).float()
        save_cube_from_tensor(bin_out.squeeze(),
                              os.path.join(opts.default_root_dir, 'test_'+opts.test_iter,'cubes', name))

    def inference(self, image):
        subject = tio.Subject(image=tio.ScalarImage(tensor=image))
        grid_sampler = tio.inference.GridSampler(subject, self.opts.test_per_size, self.opts.test_overlap)
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
        aggregator = tio.inference.GridAggregator(grid_sampler)
        for patches_batch in patch_loader:
            image, locations = patches_batch['image'][tio.DATA], patches_batch[tio.LOCATION]
            pred = self(image)
            aggregator.add_batch(pred, locations)
        output_tensor = aggregator.get_output_tensor().to(image.device)
        return output_tensor


def main(opts):
    datamodule = MultiTioDatamodule(**vars(opts))
    model = LitModel(opts)
    if opts.command == "test":
        model_save_path = find_model_by_iter(os.path.join(opts.default_root_dir, 'lightning_logs/version_0/checkpoints'), opts.test_iter)
        load_network(model, model_save_path)

        trainer = pl.Trainer.from_argparse_args(opts)
        trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='unet3d_sa_rinner_adv/ndf64frame5_adv0.01_labelConditional')
    parser.add_argument('--fold_id', type=int, default=0)
    parser.add_argument("--test_iter", default='10000')
    parser.add_argument('--devices', default=[0])
    parser.add_argument("--server_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='dataset1')
    parser.add_argument("--command", default="test")
    parser.add_argument('--reproduce', type=int, default=False)
    add_server_specific_setting(parser)
    parser = LitModel.add_model_specific_args(parser)
    opts = parser.parse_args()

    initExperiment(opts)
    main(opts)
