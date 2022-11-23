import logging
import os
import hydra
import torch
from src.executor import start_ddp_workers
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)

def run(args):
    from src import distrib
    from src.models.version_1 import CoSNetwork
    from src.models.version_2 import CoSNetwork_spk
    from src.solver import Solver
    from src.data.dataloader import SyntheticDataset
    if args.model == 'version_1':
        kwargs = dict(args.mos)
        kwargs['sr'] = args.sample_rate
        kwargs['segment'] = args.segment
        model = CoSNetwork(**kwargs)
    elif args.model == 'version_2':
        kwargs = dict(args.mos)
        kwargs['sr'] = args.sample_rate
        kwargs['segment'] = args.segment
        model = CoSNetwork_spk(**kwargs)
    else:
        logger.fatal("Invalid model name %s", args.model)
        os._exit(1)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        model.cuda()
    if args.optim=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, args.beta2))
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    tr_dataset = SyntheticDataset(args.train_dir, n_mics=args.n_mics,
                                  sr=args.sample_rate, perturb_prob=1.0,
                                  mic_radius=args.mic_radius)
    tr_loader = distrib.loader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    cv_dataset = SyntheticDataset(args.test_dir, n_mics=args.n_mics,
                                 sr=args.sample_rate, mic_radius=args.mic_radius)
    cv_loader = distrib.loader(
        cv_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    data = {"tr_loader": tr_loader,"cv_loader": cv_loader}
    solver = Solver(data, model, optimizer, args)
    solver.train()
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

