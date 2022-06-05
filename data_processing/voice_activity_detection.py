from collections import deque

import onnxruntime
import torch
import torch.nn.functional as F


class VADSilero:

    def __init__(self, model_path: str = "/models/model.onnx", use_gpu: bool = False):

        self.sampling_rate = 16000
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        if use_gpu:
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.ort_session = onnxruntime.InferenceSession(model_path, sess_options=session_options,
                                                        providers=providers)

    def segment(self, audio: torch.Tensor):
        speech_timestamps = self._get_speech_ts_adaptive(audio)
        return speech_timestamps

    def _run_model(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            ort_inputs = {'input': inputs.cpu().numpy()}
            outs = self.ort_session.run(None, ort_inputs)
            outs = [torch.Tensor(x) for x in outs]
        return outs[0]

    @staticmethod
    def audio_to_segments(audio, speech_timestamps):
        segments = []
        for speech_timestamp in speech_timestamps:
            start = speech_timestamp['start']
            end = speech_timestamp['end']
            segments.append((start / 16000, end / 16000, audio[start:end]))

        return segments

    @staticmethod
    def audio_to_segments_from_stamps(audio, speech_timestamps):
        segments = []
        for speech_timestamp in speech_timestamps:
            start = speech_timestamp.start
            end = speech_timestamp.end
            segments.append((start / 16000, end / 16000, audio[start:end]))

        return segments

    def _get_speech_ts_adaptive(self,
                                wav: torch.Tensor,
                                batch_size: int = 200,
                                step: int = 200,
                                num_samples_per_window: int = 4000,
                                min_speech_samples: int = 10000,  # samples
                                min_silence_samples: int = 3000,
                                speech_pad_samples: int = 2000,
                                device='cpu'):
        """
        source: https://github.com/snakers4/silero-vad/blob/master/utils_vad.py
        This function is used for splitting long audios into speech chunks using silero VAD
        Attention! All default sample rate values are optimal for 16000 sample rate model,
        if you are using 8000 sample rate model optimal values are half as much!
        Parameters
        ----------
        batch_size: int
            batch size to feed to silero VAD (default - 200)
        step: int
            step size in samples, (default - 500)
        num_samples_per_window: int
            window size in samples (chunk length in samples to feed to NN, default - 4000)
            (4000 for 16k SR, 2000 for 8k SR is optimal)
        min_speech_samples: int
            if speech duration is shorter than this value, do not consider it speech (default - 10000)
        min_silence_samples: int
            number of samples to wait before considering as the end of speech (default - 4000)
        speech_pad_samples: int
            widen speech by this amount of samples each side (default - 2000)
        run_function: function
            function to use for the model call
        device: string
            torch device to use for the model call (default - "cpu")
        Returns
        ----------
        speeches: list
            list containing ends and beginnings of speech chunks (in samples)
        """
        num_samples = num_samples_per_window
        num_steps = int(num_samples / step)
        assert min_silence_samples >= step
        outs = []
        to_concat = []
        for i in range(0, len(wav), step):
            chunk = wav[i: i + num_samples]
            if len(chunk) < num_samples:
                chunk = F.pad(chunk, (0, num_samples - len(chunk)))
            to_concat.append(chunk.unsqueeze(0))
            if len(to_concat) >= batch_size:
                chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
                out = self._run_model(chunks)
                outs.append(out)
                to_concat = []

        if to_concat:
            chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
            out = self._run_model(chunks)
            outs.append(out)

        outs = torch.cat(outs, dim=0).cpu()

        buffer = deque(maxlen=num_steps)
        triggered = False
        speeches = []
        current_speech = {}
        speech_probs = outs[:, 1]  # 0 index for silence probs, 1 index for speech probs
        median_probs = speech_probs.median()

        trig_sum = 0.89 * median_probs + 0.08  # 0.08 when median is zero, 0.97 when median is 1

        temp_end = 0

        for i, predict in enumerate(speech_probs):
            buffer.append(predict)
            smoothed_prob = max(buffer)
            if (smoothed_prob >= trig_sum) and temp_end:
                temp_end = 0
            if (smoothed_prob >= trig_sum) and not triggered:
                triggered = True
                current_speech['start'] = step * max(0, i - num_steps)
                continue
            if (smoothed_prob < trig_sum) and triggered:
                if not temp_end:
                    temp_end = step * i
                if step * i - temp_end < min_silence_samples:
                    continue
                else:
                    current_speech['end'] = temp_end
                    if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                        speeches.append(current_speech)
                    temp_end = 0
                    current_speech = {}
                    triggered = False
                    continue

        if current_speech:
            current_speech['end'] = len(wav)
            speeches.append(current_speech)

        for i, ts in enumerate(speeches):
            if i == 0:
                ts['start'] = max(0, ts['start'] - speech_pad_samples)
            if i != len(speeches) - 1:
                silence_duration = speeches[i + 1]['start'] - ts['end']
                if silence_duration < 2 * speech_pad_samples:
                    ts['end'] += silence_duration // 2
                    speeches[i + 1]['start'] = max(0, speeches[i + 1]['start'] - silence_duration // 2)
                else:
                    ts['end'] += speech_pad_samples
            else:
                ts['end'] = min(len(wav), ts['end'] + speech_pad_samples)

        return speeches

    def get_VAD_matrix(self,
                       wav: torch.Tensor,
                       batch_size: int = 200,
                       step: int = 200,
                       num_samples_per_window: int = 4000,
                       device='cuda'):
        """
        source: https://github.com/snakers4/silero-vad/blob/master/utils_vad.py
        This function is used for splitting long audios into speech chunks using silero VAD
        Attention! All default sample rate values are optimal for 16000 sample rate model,
        if you are using 8000 sample rate model optimal values are half as much!
        Parameters
        ----------
        batch_size: int
            batch size to feed to silero VAD (default - 200)
        step: int
            step size in samples, (default - 500)
        num_samples_per_window: int
            window size in samples (chunk length in samples to feed to NN, default - 4000)
            (4000 for 16k SR, 2000 for 8k SR is optimal)
        min_speech_samples: int
            if speech duration is shorter than this value, do not consider it speech (default - 10000)
        min_silence_samples: int
            number of samples to wait before considering as the end of speech (default - 4000)
        speech_pad_samples: int
            widen speech by this amount of samples each side (default - 2000)
        run_function: function
            function to use for the model call
        device: string
            torch device to use for the model call (default - "cpu")
        Returns
        ----------
        speeches: list
            list containing ends and beginnings of speech chunks (in samples)
        """
        num_samples = num_samples_per_window
        outs = []
        to_concat = []
        for i in range(0, len(wav), step):
            chunk = wav[i: i + num_samples]
            if len(chunk) < num_samples:
                chunk = F.pad(chunk, (0, num_samples - len(chunk)))
            to_concat.append(chunk.unsqueeze(0))
            if len(to_concat) >= batch_size:
                chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
                out = self._run_model(chunks)
                outs.append(out)
                to_concat = []

        if to_concat:
            chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
            out = self._run_model(chunks)
            outs.append(out)

        outs = torch.cat(outs, dim=0).cpu()

        return outs[:, 1]
