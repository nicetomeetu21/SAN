import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torchio as tio
import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint

from BaseProcess.default_setting import add_server_specific_setting
from BaseProcess.init_experiment import initExperiment
from BaseProcess.metrics_calculation import cal_metrics, cal_masked_metrics
from datamodule.shape_aware_datamodule import MultiTioDatamodule_2
from networks.unet_3d_fix import UNet_2out
from networks.mynet_parts.dice_loss import DiceLoss
from networks.mynet_parts.scheduler import get_scheduler
from networks.mynet_parts.networks_for_Unit3d_fix import MsImageDis
from utils.util import gen_visual_imgs, save_metrics_to_json, save_cube_from_tensor, load_network,find_model_by_iter


class LitModel(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.netS = UNet_2out(**vars(opts))

        if opts.command == 'fit':
            self.save_hyperparameters()
            self.eval_metric_dict = {}
            self.pbar = tqdm.tqdm(total=opts.max_steps)

            # Loss functions
            self.criterion_dice = DiceLoss()
            self.criterion_ce = torch.nn.BCEWithLogitsLoss()
            self.criterion_l2 = torch.nn.MSELoss()
            hyperparameters_dis = dict(dim=32, norm='none', activ='lrelu', n_layer=4, gan_type='lsgan', num_scales=1,
                                       pad_type='zero')
            self.netD2 = MsImageDis(2, hyperparameters_dis)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("UNet3D")
        # train args
        parser.add_argument("--max_steps", type=int, default=100000)
        parser.add_argument("--val_check_interval", type=int, default=2000)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--lr", type=float, default=0.0002)
        parser.add_argument("--b1", type=float, default=0.5)
        parser.add_argument("--b2", type=float, default=0.999)
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

    def training_step(self, batch, batch_idx, optimizer_idx):
        image, label, label_shape = batch['image']['data'], batch['label']['data'], batch['shape_label']['data']
        pred, pred_shape = self.netS(image)
        if optimizer_idx == 0:
            loss_seg = self.criterion_dice(pred, label) + self.criterion_ce(pred, label)
            loss_dist = self.criterion_l2(pred_shape, label_shape)
            input_fake = torch.cat([pred, pred_shape], dim=1)
            loss_adv = self.netD2.calc_gen_loss(input_fake)
            loss_total = loss_seg + loss_dist +loss_adv*0.01
            self.log("losses",
                     {'loss_total': loss_total, 'loss_seg': loss_seg, 'loss_dist': loss_dist, 'loss_adv':loss_adv})
            self.pbar.update(1)
            return loss_total
        elif optimizer_idx == 1:
            input_real = torch.cat([label, label_shape], dim=1)
            input_fake = torch.cat([pred.detach(), pred_shape.detach()], dim=1)
            loss_dis = self.netD2.calc_dis_loss(input_fake=input_fake, input_real=input_real)
            self.log('loss dis', loss_dis)
            self.pbar.update(1)
            return loss_dis

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        if self.global_step % self.opts.val_check_interval == 0:
            image, label, label_shape = batch['image']['data'], batch['label']['data'], batch['shape_label']['data']
            pred, pred_shape = self.netS(image)
            prb_out = torch.sigmoid(pred)
            train_visual = gen_visual_imgs([image, prb_out, label, pred_shape, label_shape])
            self.logger.experiment.add_images('train_visual', torch.cat(train_visual, dim=0), self.global_step)

    def validation_step(self, batch, batch_idx):
        image, label, region_mask = batch['image']['data'], batch['label']['data'], batch['region_mask']['data']
        pred = self.inference(image)
        prb_out = torch.sigmoid(pred)

        eval_metric_dict = {}
        cal_metrics(prb_out, label, metric_dict=eval_metric_dict)
        cal_masked_metrics(prb_out, label, region_mask, metric_dict=eval_metric_dict)
        save_metrics_to_json(eval_metric_dict, self.global_step, self.opts.default_root_dir)
        self.log('mmiou', eval_metric_dict['mmiou'][0])
        if batch_idx == 0:
            visuals = torch.stack(gen_visual_imgs([image, prb_out, label]), dim=0)
            self.logger.experiment.add_images('val_visual', visuals, self.global_step)

    def test_step(self, batch, batch_idx):
        image, label, region_mask, name = batch['image']['data'], batch['label']['data'], batch['region_mask']['data'], \
                                          batch['name']
        pred = self.inference(image)
        prb_out = torch.sigmoid(pred)
        bin_out = (prb_out > 0.5).float()
        save_cube_from_tensor(bin_out.squeeze(),
                              os.path.join(opts.result_root, opts.exp_name, 'test_results/pred_bin/cubes', name))

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

    def configure_optimizers(self):
        lr, b1, b2 = self.opts.lr, self.opts.b1, self.opts.b2
        optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D = torch.optim.Adam(self.netD2.parameters(), lr=lr * 2, betas=(b1, b2))
        scheduler_S = get_scheduler(optimizer_S, n_epochs=self.opts.max_steps, offset=0,
                                  decay_start_epoch=self.opts.max_steps // 2)
        scheduler_D = get_scheduler(optimizer_D, n_epochs=self.opts.max_steps, offset=0,
                                  decay_start_epoch=self.opts.max_steps // 2)
        return [optimizer_S, optimizer_D], [scheduler_S, scheduler_D]


def main(opts):
    datamodule = MultiTioDatamodule_2(**vars(opts))
    model = LitModel(opts)
    if opts.command == "fit":
        ckpt_callback = ModelCheckpoint(save_last=False, save_top_k=-1, every_n_train_steps=opts.val_check_interval,
                                        filename="model-{step:02d}")
        trainer = pl.Trainer.from_argparse_args(opts, callbacks=[ckpt_callback], check_val_every_n_epoch=None,
                                                val_check_interval=opts.val_check_interval)
        trainer.fit(model=model, datamodule=datamodule)
    elif opts.command == "test":
        checkpoint = torch.load(opts.ckpt_path)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'])

        trainer = pl.Trainer.from_argparse_args(opts)
        trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='unet3d_sa_rinner_adv/ndf64frame5_adv0.01_labelConditional')
    parser.add_argument("--dataset", type=str, default='dataset4')
    parser.add_argument('--fold_id', type=int, default=0)
    parser.add_argument("--server_id", type=int, default=0)
    parser.add_argument('--devices', default=[0])
    parser.add_argument("--command", default="fit")
    parser.add_argument("--ckpt_path", default='')
    parser.add_argument('--reproduce', type=int, default=False)
    add_server_specific_setting(parser)
    parser = LitModel.add_model_specific_args(parser)
    opts = parser.parse_args()

    initExperiment(opts)
    main(opts)
