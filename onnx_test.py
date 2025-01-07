import onnx
onnx_path = "/home/bjwoo/PycharmProjects/Speech_source_separation/outputs/skim_mixed_re/sitec_single_channel_ss.onnx"
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
import time
import soundfile as sf
import numpy as np
import onnxruntime
import torch
ort_session = onnxruntime.InferenceSession(onnx_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


for i in range(1):
    frame_time_max = 0
    start = time.time()
    wav = sf.read(
        "/home/bjwoo/PycharmProjects/data/wsj0-mix/2speakers/wav16k/min/tr/mix/401o030g_1.9154_01ec020g_-1.9154.wav")[0]
    wav = torch.FloatTensor(wav)
    wav = wav.unsqueeze(0).cpu()

    wav = wav[:,:640]
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(wav)}
    est = ort_session.run(None, ort_inputs)[0]
    print(est.size())
