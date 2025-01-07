import logging
import os
import hydra
import torch
from src.models.version_3.arch.loss import cal_loss
from src.executor import start_ddp_workers
from src.utils import  *
import sys
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
import numpy as np
from tqdm import tqdm
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

logger = logging.getLogger(__name__)


def run(args):
    from src import distrib
    from src.models.version_3.arch.NBSS import NBSS
    from src.solver import Solver

    if args.model != "version_4":
        print("wrong version for test!")
        sys.exit(0)

    kwargs = dict(args.version_4)
    kwargs['n_channel'] = args.n_mics
    model = NBSS(**kwargs)
    pkg = torch.load(args.checkpoint_file, map_location=args.device)
    # model.load_state_dict(pkg['model']['state'])
    # model.load_state_dict(pkg['best_state'])
    logger.info("Model Loading completed!")
    from src.data.multi_channel_dataloader import Validset
    mic_prefix = str(args.n_mics) + "mic"
    tt_dataset = Validset(os.path.join(args.json_dir, "test", mic_prefix))
    tt_loader = distrib.loader(
        tt_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    RTF = evaluate(args,model,tt_loader)

def evaluate(args, model, data_loader):

    total_sisdr = 0
    total_cnt = 0
    total_time = 0
    total_length = 0
    logger.debug(model)
    model.eval()
    args.device = "cpu"
    model.to(args.device)
    # Load data
    pendings = []
    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Eval estimates")
        for i, data in tqdm(enumerate(iterator)):
            # Get batch data
            mixture, lengths, sources = [x.to(args.device) for x in data]
            # Forward
            sources = sources.squeeze(dim=2)
            with torch.no_grad():
                mixture /= mixture.max()
                start = time.time()
                estimate = model(mixture)
                end = time.time()
                total_time = total_time + (end-start)
                total_length = total_length + mixture.size(-1)/16000

    RTF = total_time/total_length
    logger.info(
        bold(f'Test set RTF : {RTF:.4f}'))
    return RTF

@hydra.main(config_path="conf/", config_name='config.yaml')
def main(args):
    global __file__
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    if args.ddp and args.rank is None:
        start_ddp_workers()
    else:
        run(args)


if __name__ == '__main__':
    # asdf = torch.randn([1,2,14323])
    # zxcv = torch.randn([1,8,14323])
    # _run_metrics(asdf,asdf,zxcv)
    main()

