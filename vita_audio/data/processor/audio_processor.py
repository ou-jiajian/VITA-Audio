import json
import math
import os

import cv2
import numpy as np
import torch

import natsort
from vita_audio.tokenizer import get_audio_tokenizer


class AudioProcessor:
    def __init__(
        self,
        audio_tokenizer_path=None,
        audio_tokenizer_type=None,
    ):

        self.audio_tokenizer = get_audio_tokenizer(
            audio_tokenizer_path,
            audio_tokenizer_type,
        )

        self.audio_tokenizer_type = audio_tokenizer_type

        # self.load_model()

    def load_model(self):
        if self.audio_tokenizer is not None:
            self.audio_tokenizer.load_model()

    def process_audios(self, audio_path, is_discrete=False, is_contiguous=False, **kwargs):

        assert not (is_discrete and is_contiguous)
        assert is_discrete or is_contiguous

        if is_discrete:
            audio_tokenizer_type = self.audio_tokenizer_type.split("_")[-1]
            cache_path = os.path.splitext(audio_path)[0] + f"_{audio_tokenizer_type}.json"
            try:
                if os.path.isfile(cache_path):
                    with open(cache_path, "r") as f:
                        audio_data = json.load(f)
                    return audio_data
            except Exception as e:
                pass

        audio_data = self.audio_tokenizer.encode(
            audio_path, is_discrete=is_discrete, is_contiguous=is_contiguous, **kwargs
        )
        # print(f"{len(audio_data)=}")

        if is_discrete:
            try:
                if isinstance(audio_data, list):
                    with open(cache_path, "w") as f:
                        json.dump(audio_data, f)
            except Exception as e:
                pass

        return audio_data

    @property
    def is_discrete(self):
        return self.audio_tokenizer.is_discrete

    @property
    def is_contiguous(self):
        return self.audio_tokenizer.is_contiguous

    @property
    def text_audio_interval_ratio(self):
        if self.audio_tokenizer is None:
            return []
        return self.audio_tokenizer.text_audio_interval_ratio

    def apply_to_role(self, role, **kwargs):
        return self.audio_tokenizer.apply_to_role(role, **kwargs)


def add_audio_input_contiguous(input_ids, audio_path, tokenizer, audio_tokenizer):

    from ...constants import (
        AUD_START_TOKEN,
        AUD_END_TOKEN,
        AUD_TAG_TOKEN,
        AUD_CONTEXT_TOKEN,
    )

    AUD_CONTEXT_ID = tokenizer(AUD_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    AUD_TAG_ID = tokenizer(AUD_TAG_TOKEN, add_special_tokens=False).input_ids
    AUD_START_ID = tokenizer(AUD_START_TOKEN, add_special_tokens=False).input_ids
    AUD_END_ID = tokenizer(AUD_END_TOKEN, add_special_tokens=False).input_ids

    AUD_CONTEXT_ID = AUD_CONTEXT_ID[0]
    AUD_TAG_ID = AUD_TAG_ID[0]
    AUD_START_ID = AUD_START_ID[0]
    AUD_END_ID = AUD_END_ID[0]

    aud_positions = [i for i, x in enumerate(input_ids) if x == AUD_TAG_ID]

    audios = []
    audio_indices = []
    new_input_ids = []
    st = 0
    for aud_idx, aud_pos in enumerate(aud_positions):
        audio = audio_tokenizer.encode(audio_path, is_contiguous=True)
        audios.append(audio)
        audio_token_length = audio.size(0) + 4

        new_input_ids += input_ids[st:aud_pos]

        new_input_ids += [AUD_START_ID]

        audio_indice_b = torch.zeros(
            1, audio_token_length, dtype=torch.int64
        )  # This will change in collate_fn
        audio_indice_s = (
            torch.arange(len(new_input_ids), len(new_input_ids) + audio_token_length)
            .unsqueeze(0)
            .repeat(1, 1)
        )
        audio_indice_b_s = torch.stack(
            [audio_indice_b, audio_indice_s], dim=0
        )  # 2, num_image, image_length
        audio_indices.append(audio_indice_b_s)

        new_input_ids += [AUD_CONTEXT_ID] * audio_token_length

        new_input_ids += [AUD_END_ID]

        st = aud_pos + 1

    new_input_ids += input_ids[st:]
    inputs_ids = new_input_ids

    return inputs_ids, audios, audio_indices
