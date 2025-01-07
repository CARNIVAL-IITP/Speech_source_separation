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
    from src.models.version_3_dir.arch.NBSS import NBSS
    from src.solver import Solver


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
    elif args.model == 'version_3_folder':
        kwargs = dict(args.version_3)
        kwargs['n_channel'] = args.n_mics
        model = NBSS(**kwargs)
    elif args.model == 'skim':
        from src.models.SKIM.SkiMSeparator import SkiM_Model
        kwargs = dict(args.skim)
        model = SkiM_Model(**kwargs)
    else:
        logger.fatal("Invalid model name %s", args.model)
        os._exit(1)
    if args.model == 'version_3_folder':
        from src.data.multi_channel_dataloader import Validset
        mic_prefix = str(args.n_mics) + "mic"
        tt_dataset = Validset(os.path.join(args.json_dir,"test",mic_prefix))
        tt_loader = distrib.loader(
            tt_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    else:
        from src.data.dataloader import SyntheticDataset
        tt_dataset = SyntheticDataset(args.test_dir, n_mics=args.n_mics,
                                     sr=args.sample_rate, mic_radius=args.mic_radius)
        tt_loader = distrib.loader(
            tt_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)



@hydra.main(config_path="conf/", config_name='config_eval.yaml')
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

