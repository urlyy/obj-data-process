__all__ = ['app']

from fastapi import UploadFile, APIRouter, Form
from starlette.responses import FileResponse
from txt2voice.vocoder.hifigan import inference as gan_vocoder
from txt2voice.vocoder.wavernn import inference as rnn_vocoder
import numpy as np
from txt2voice.synthesizer.inference import Synthesizer
from txt2voice.encoder import inference as encoder
import io
from pathlib import Path
from scipy.io.wavfile import write
import re
import noise_reduce

import os

syn_models_dirt = os.path.join(os.getcwd(), "txt2voice", "synthesizer", "saved_models")
synthesizers = list(Path(syn_models_dirt).glob("**/*.pt"))
synthesizers_cache = {}
encoder.load_model(Path(os.path.join(os.getcwd(), "txt2voice", "encoder", "saved_models", "pretrained.pt")))
# rnn_vocoder.load_model(Path("vocoder/saved_models/pretrained/pretrained.pt"))
gan_vocoder.load_model(
    Path(os.path.join(os.getcwd(), "txt2voice", "vocoder", "saved_models", "pretrained", "g_hifigan.pt")))

dir = os.path.join(os.getcwd(), "txt2voice", "tmp_file")
if not os.path.exists(dir):
    os.mkdir(dir)

app = APIRouter()


def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
        Use dtype='float32' for single precision.
        Parameters
        ----------
        sig : array_like
            Input array, must have integral type.
        dtype : data type, optional
            Desired (floating point) data type.
        Returns
        -------
        numpy.ndarray
            Normalized floating point data.
        See Also
        --------
        float2pcm, dtype
        """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


# 删除无声音的第一秒TODO
def remove_first_second(file_path):
    pass
    # audio = AudioSegment.from_file(file_path, format="mp3")
    # # 删除前1秒
    # audio = audio[1000:]
    # # 保存输出文件
    # audio.export(file_path, format="mp3")


@app.post("/audio/clone", summary="语音克隆")
def synthesize(file: UploadFile, text: str = Form(None)):
    # TODO Implementation with json to support more platform
    # Load synthesizer
    synt_path = synthesizers[0]
    if synthesizers_cache.get(synt_path) is None:
        current_synt = Synthesizer(Path(synt_path))
        synthesizers_cache[synt_path] = current_synt
    else:
        current_synt = synthesizers_cache[synt_path]
    print("using synthesizer model: " + str(synt_path))
    # Load input wav
    # if upfile_b64.filename!="":
    #         wav_base64 = upfile_b64
    #         wav = base64.b64decode(bytes(wav_base64, 'utf-8'))
    #         wav = pcm2float(np.frombuffer(wav, dtype=np.int16), dtype=np.float32)
    #         sample_rate = Synthesizer.sample_rate
    # else:
    # wav, sample_rate, = librosa.load(file.file)
    # 读取并降噪
    wav, sample_rate, = noise_reduce._reduce(file)
    # write("temp.wav", sample_rate, wav)  # Make sure we get the correct wav

    encoder_wav = encoder.preprocess_wav(wav, sample_rate)
    embed, _, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

    # Load input text
    texts = text.split("\n")
    # 必须加一段空白音效果才好
    texts.insert(0, " ")
    punctuation = '！，。、,'  # punctuate and split/clean text
    processed_texts = []
    for text in texts:
        for processed_text in re.sub(r'[{}]+'.format(punctuation), '\n', text).split('\n'):
            if processed_text:
                processed_texts.append(processed_text.strip())
    texts = processed_texts

    # synthesize and vocode
    embeds = [embed] * len(texts)
    specs = current_synt.synthesize_spectrograms(texts, embeds)
    spec = np.concatenate(specs, axis=1)
    # wav = rnn_vocoder.infer_waveform(spec)
    wav = gan_vocoder.infer_waveform(spec)
    # Return cooked wav
    out = io.BytesIO()
    write(out, Synthesizer.sample_rate, wav)
    # wav, sample_rate = noise_reduce.reduce(out)
    # 降噪
    # new_out = io.BytesIO()
    # write(new_out, sample_rate, wav)
    file_path = os.path.join(dir, "audio_file_no_noise.wav")
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'wb') as f:
        # 如果要降噪，下面也要改成new_out
        f.write(out.read())
    # 截掉一开始那1秒
    # remove_first_second(file_path)
    return FileResponse(file_path, media_type="audio/wav")
