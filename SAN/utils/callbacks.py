from pytorch_lightning.callbacks import TQDMProgressBar
import tqdm

class LitProgressBar(TQDMProgressBar):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        print(self.opts.max_iters)
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.reset(total=self.opts.max_iters)
        return bar