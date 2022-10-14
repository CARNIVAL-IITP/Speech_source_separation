import logging
import os
import hydra
from IITP_MCSS.executor import start_ddp_workers
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)

def run(args):
    from IITP_MCSS.src import distrib
    from IITP_MCSS.src.models.version_1 import CoSNetwork

    if args.model == 'version_1':
        kwargs = dict(args.swave)
        kwargs['sr'] = args.sample_rate
        kwargs['segment'] = args.segment
        model = CoSNetwork(**kwargs)
    elif args.model == 'version_2':
        kwargs = dict(args.swave)
        kwargs['sr'] = args.sample_rate
        kwargs['segment'] = args.segment
        model = CoSNetwork(**kwargs)
    else:
        logger.fatal("Invalid model name %s", args.model)
        os._exit(1)

    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

@hydra.main(config_path="conf", config_name='config.yaml')
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

