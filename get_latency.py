import logging
import os
import hydra
import torch
import sys
from src.executor import start_ddp_workers
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

logger = logging.getLogger(__name__)

def run(args):
    from src.models.version_3.arch.NBSS import NBSS

    if args.model != "version_4":
        print("wrong version for test!")
        sys.exit(0)

    kwargs = dict(args.version_4)
    kwargs['n_channel'] = args.n_mics
    model = NBSS(**kwargs)
    latency = model.get_latency()
    logger.info(f"Latency of this alogrithm is {latency}ms")

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
    main()

