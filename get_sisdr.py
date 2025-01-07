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
    model.load_state_dict(pkg['model']['state'])
    # model.load_state_dict(pkg['best_state'])
    logger.info("Model Loading completed!")
    from src.data.multi_channel_dataloader import Validset
    mic_prefix = str(args.n_mics) + "mic"
    tt_dataset = Validset(os.path.join(args.json_dir, "test", mic_prefix))
    tt_loader = distrib.loader(
        tt_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    SISDRi = evaluate(args,model,tt_loader)
    logger.info(f"SISDRi: {SISDRi}")

def evaluate(args, model, data_loader):

    total_sisdr = 0
    total_cnt = 0
    total_time = 0
    total_length = 0
    logger.debug(model)
    model.eval()
    # args.device = "cpu"
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
                estimate = model(mixture)
            mixture = mixture[:,0,:]
            # sisdr_loss, sdr, estimate, reorder_estimate = cal_loss(
            #     sources, estimate, lengths)
            neg_sisdr_loss, best_perm = pit(preds=estimate, target=sources, metric_func=neg_si_sdr, eval_func='min')
            if best_perm[0][0]==1:
                B,C,T = sources.size()
                reorder_estimate = torch.zeros([B,C,T])
                reorder_estimate[:, 0,: ] = estimate[:,1,:]
                reorder_estimate[:, 1, :] = estimate[:, 0, :]
            else:
                B, C, T = sources.size()
                reorder_estimate = torch.zeros([B, C, T])
                reorder_estimate[:, 0, :] = estimate[:, 0, :]
                reorder_estimate[:, 1, :] = estimate[:, 1, :]
            reorder_estimate = reorder_estimate.cpu()

            sources = sources.cpu()
            mixture = mixture.cpu()
            sisdri = _run_metrics(sources,reorder_estimate,mixture)

            total_sisdr += sisdri
            total_cnt += sources.shape[0]

    sisdr = total_sisdr/total_cnt
    logger.info(
        bold(f'Test set performance: SISDRi={sisdr:.2f}'))
    return sisdr
def _run_metrics(clean, estimate, mix):
    estimate = estimate.numpy()
    clean = clean.numpy()
    mix = mix.numpy()
    sisdr = cal_SISDRi(clean, estimate, mix)

    return sisdr.mean()


def cal_SISDR(ref_sig, out_sig, eps=1e-8):

    assert len(ref_sig) == len(out_sig)
    B, T = ref_sig.shape
    ref_sig = ref_sig - np.mean(ref_sig, axis=1).reshape(B, 1)
    out_sig = out_sig - np.mean(out_sig, axis=1).reshape(B, 1)
    ref_energy = (np.sum(ref_sig ** 2, axis=1) + eps).reshape(B, 1)
    proj = (np.sum(ref_sig * out_sig, axis=1).reshape(B, 1)) * \
        ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2, axis=1) / (np.sum(noise ** 2, axis=1) + eps)
    sisdr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisdr.mean()



def cal_SISDRi(src_ref, src_est, mix):
    avg_SISDRi = 0.0
    B, C, T = src_ref.shape
    for c in range(C):
        sisdr = cal_SISDR(src_ref[:, c], src_est[:, c])
        sisdrb = cal_SISDR(src_ref[:, c], mix)
        avg_SISDRi += (sisdr - sisdrb)
    avg_SISDRi /= C
    return avg_SISDRi
def reorder_source(source,best_perm):

    return 0

def neg_si_sdr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch_size = target.shape[0]
    si_sdr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_sdr_val.view(batch_size, -1), dim=1)
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

