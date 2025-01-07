import torch
import onnx
import logging
import os
from tqdm import tqdm
import hydra
import torch
import numpy as np
from src.executor import start_ddp_workers
from src.utils import  *
import librosa
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)


def run(args):
    from src.models.SKIM.model import SKIM ,SKIM_Stream
    kwargs = dict(args.skim)
    model = SKIM(**kwargs)
    stream_model = SKIM_Stream(**kwargs)
    pkg = torch.load(args.checkpoint_file, map_location=args.device)
    model.load_state_dict(pkg["model"]["state"])
    stream_model.load_state_dict(pkg["model"]["state"])
    if args.model != "skim":
        print("wrong version for test!")
        sys.exit(0)
    model.eval()
    stream_model.eval()
    import torchaudio
    wav = torchaudio.load(
        "/home/bjwoo/PycharmProjects/data/wsj0-mix/2speakers/wav16k/min/tr/mix/401o030g_1.9154_01ec020g_-1.9154.wav")[0]
    dummy_input = torch.randn([1,64000]) #.to("cuda")
    res = model(wav)
    print(res.size())
    chunks = stream_model.encoder.streaming_frame(wav)
    output_chunks = [[] for ii in range(2)]
    # for chunk in tqdm(chunks):
    #     # process a single chunk
    #     output = stream_model(chunk)
    #     for channel in range(2):
    #         # append processed chunks to ouput channels
    #         output_chunks[channel].append(output[channel])
    # waves = [stream_model.decoder.streaming_merge(chunks) for chunks in output_chunks]
    # waves = torch.stack(waves,dim=1)
    import soundfile as sf
    sf.write("source_1.wav", res[0, 0].detach().numpy(), 16000, subtype="PCM_24")
    sf.write("source_2.wav", res[0, 1].detach().numpy(), 16000, subtype="PCM_24")
    print(waves.size())
    loss = torch.nn.MSELoss()
    print(loss(waves,res))
    # input_names = ['input'] #, 'input_hidden']
    # output_names = ['output'] #, 'output_hidden']
    # torch.onnx.export(model,
    #                   dummy_input,
    #                   "sitec_single_channel_ss.onnx",
    #                   export_params=True,
    #                   verbose=False,
    #                   opset_version=12,
    #                   do_constant_folding=True,
    #                   input_names = input_names,
    #                   output_names = output_names,
    #                   dynamic_axes = {'input' : {0: 'batch_size', 1:'seq_len'}, 'output':{0: 'batch_size', 1:'num_spk',2:'seq_len'}}) #, 'input_hidden':{0: 'batch_size'}, 'output_hidden':{0: 'batch_size'}})
    #
    # onnx_path = "/home/bjwoo/PycharmProjects/Speech_source_separation/outputs/skim_mixed_re/sitec_single_channel_ss.onnx"
    # onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(onnx_model)



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

