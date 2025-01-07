import logging
import os
import hydra
import torch
from src.executor import start_ddp_workers
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logger = logging.getLogger(__name__)

def run(args):
    from src import distrib
    from src.models.version_1 import CoSNetwork
    from src.models.version_2 import CoSNetwork_spk
    from src.models.version_3 import MCSS_V3
    # from src.models.version_3_folder.arch.NBSS import NBSS
    from src.solver_skim import Solver


    speaker_model = None
    if args.model == 'version_1':
        kwargs = dict(args.mos)
        kwargs['sr'] = args.sample_rate
        kwargs['segment'] = args.segment
        model = CoSNetwork(**kwargs)
    elif args.model == 'version_2':
        kwargs = dict(args.cos)
        kwargs['sr'] = args.sample_rate
        kwargs['segment'] = args.segment
        model = CoSNetwork_spk(**kwargs)
        from src.speaker_model.speaker_network import ECAPA_TDNN
        speaker_model = ECAPA_TDNN(C=64)
        self_state = speaker_model.state_dict()
        loaded_state = torch.load(args.speaker_model_checkpoint)
        for name, param in loaded_state.items():
            if name.split(".")[0] == "speaker_encoder":
                self_state[name.split("speaker_encoder.")[-1]].copy_(param)
    elif args.model == 'version_3':
        kwargs = dict(args.version_3)
        kwargs['n_channel'] = args.n_mics
        model = MCSS_V3(**kwargs)
    elif args.model == 'skim':
        from src.models.SKIM.model import SKIM
        kwargs = dict(args.skim)
        model = SKIM(**kwargs)
    else:
        logger.fatal("Invalid model name %s", args.model)
        os._exit(1)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        model.cuda()
    if args.optim=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, args.beta2))
    elif args.optim=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr, betas=(args.beta1, args.beta2),weight_decay=args.weight_decay)
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    if args.model == 'version_3':
        from src.data.multi_channel_dataloader import Trainset,Validset
        mic_prefix = str(args.n_mics) + "mic"
        tr_dataset = Trainset(os.path.join(args.json_dir,"train",mic_prefix), sample_rate=args.sample_rate, segment=args.segment, stride=args.stride, pad=args.pad)
        tr_loader = distrib.loader(
            tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        cv_dataset = Validset(os.path.join(args.json_dir,"valid",mic_prefix))
        cv_loader = distrib.loader(
            cv_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    elif args.model == 'skim':
        from src.data.skim_dataloader import Train_dataset,Valid_dataset
        tr_dataset = Train_dataset(args.json_dir,args.sitec_path_dir,segment=args.segment,pad=args.pad)
        tr_loader = distrib.loader(
            tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        cv_dataset = Valid_dataset(args.json_dir, segment=args.segment, pad=args.pad)
        cv_loader = distrib.loader(
            cv_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    else:
        from src.data.dataloader import SyntheticDataset
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
    solver = Solver(data, model, speaker_model, optimizer, args)
    solver.train()
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

