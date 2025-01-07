import logging
import os
import hydra
import torch
import numpy as np
from src.models.version_3_folder.arch.loss import cal_loss
from concurrent.futures import ProcessPoolExecutor
from src.executor import start_ddp_workers
from src.utils import  *
from src import distrib
import librosa
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

logger = logging.getLogger(__name__)


def run(args):
    from src import distrib
    from src.models.SKIM.model import SKIM
    kwargs = dict(args.skim)
    model = SKIM(**kwargs)
    # args.checkpoint_file = "temp_best.pth.tar"
    pkg = torch.load(args.checkpoint_file, map_location=args.device)
    # model.load_state_dict(pkg)
    model.load_state_dict(pkg["model"]["state"])
    if args.model != "skim":
        print("wrong version for test!")
        sys.exit(0)
    from src.data.skim_dataloader import Valid_dataset
    tt_dataset = Valid_dataset(args.json_dir, segment=args.segment, pad=args.pad)
    tt_loader = distrib.loader(
        tt_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    evaluate(args,model,tt_loader)

def evaluate(args, model=None, data_loader=None, sr=None):

    total_sisnr = 0
    total_pesq = 0
    total_stoi = 0
    total_cnt = 0
    updates = 5
    # Load model
    if not model:
        pkg = torch.load(args.model_path, map_location=args.device)
        if 'model' in pkg:
            model = pkg['model']
        else:
            model = pkg
        model = deserialize_model(model)
        if 'best_state' in pkg:
            model.load_state_dict(pkg['best_state'])
    logger.debug(model)
    model.eval()
    model.to(args.device)
    # Load data
    pendings = []
    data_path = "/home/bjwoo/PycharmProjects/data/sitec"
    from glob import glob
    data_list = glob(data_path + "/**/*.wav", recursive=True)
    from src.data.real_time_datamixing import Data_Mixer
    SkiM = Data_Mixer(data_list)
    a, b = SkiM(2)
    with ProcessPoolExecutor(args.num_workers) as pool:
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, name="Eval estimates")
            for i, data in enumerate(iterator):
                # Get batch data
                mixture, lengths, sources = [x.to(args.device) for x in data]
                # Forward
                if i ==0:
                    a = a.to(args.device)
                    b = b.to(args.device)
                    mixture = a
                    sources = b.permute(1,0,2)

                with torch.no_grad():
                    mixture /= mixture.max()
                    estimate = model(mixture)[-1]
                sisnr_loss, snr, estimate, reorder_estimate = cal_loss(
                    sources, estimate.unsqueeze(dim=0), lengths)

                reorder_estimate = reorder_estimate.cpu()
                sources = sources.cpu()
                mixture = mixture.cpu()
                if i ==0:
                    import soundfile as sf
                    sf.write("mixed.wav",mixture[0],16000,subtype="PCM_24")
                    sf.write("source_1.wav",reorder_estimate[0,0],16000,subtype="PCM_24")
                    sf.write("source_2.wav",reorder_estimate[0,1],16000,subtype="PCM_24")
                pendings.append(
                    pool.submit(_run_metrics, sources, reorder_estimate, mixture, None,
                                sr=sr))
                total_cnt += sources.shape[0]

        for pending in LogProgress(logger, pendings, updates, name="Eval metrics"):
            si_snri = pending.result()
            total_sisnr += si_snri
    metrics = [total_sisnr]
    sisnr = distrib.average(
        [m/total_cnt for m in metrics], total_cnt)
    logger.info(
        bold(f'Test set performance: SISNRi={sisnr[0]:.2f}.'))
    return sisnr

def _run_metrics(clean, estimate, mix, model, sr, pesq=False):
    if model is not None:
        torch.set_num_threads(1)
        # parallel evaluation here
        with torch.no_grad():
            estimate = model(estimate)[-1]
    estimate = estimate.numpy()
    clean = clean.numpy()
    mix = mix.numpy()
    sisnr = cal_SISDRi(clean, estimate, mix)

    return sisnr.mean()


def cal_SISDR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    B, T = ref_sig.shape
    ref_sig = ref_sig - np.mean(ref_sig, axis=1).reshape(B, 1)
    out_sig = out_sig - np.mean(out_sig, axis=1).reshape(B, 1)
    ref_energy = (np.sum(ref_sig ** 2, axis=1) + eps).reshape(B, 1)
    proj = (np.sum(ref_sig * out_sig, axis=1).reshape(B, 1)) * \
        ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2, axis=1) / (np.sum(noise ** 2, axis=1) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr.mean()



def cal_SISDRi(src_ref, src_est, mix):

    avg_SISDRi = 0.0
    B, C, T = src_ref.shape
    for c in range(C):
        sisnr = cal_SISDR(src_ref[:, c], src_est[:, c])
        sisnrb = cal_SISDR(src_ref[:, c], mix)
        avg_SISDRi += (sisnr - sisnrb)
    avg_SISDRi /= C
    return avg_SISDRi

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

