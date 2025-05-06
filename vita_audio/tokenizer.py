from .constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    IMG_TAG_TOKEN,
    PATCH_CONTEXT_TOKEN,
    PATCH_END_TOKEN,
    PATCH_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    VID_CONTEXT_TOKEN,
    VID_END_TOKEN,
    VID_START_TOKEN,
    VID_TAG_TOKEN,
)


def update_tokenizer(tokenizer):
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        VID_START_TOKEN,
        VID_END_TOKEN,
        VID_CONTEXT_TOKEN,
        PATCH_START_TOKEN,
        PATCH_END_TOKEN,
        PATCH_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
        IMG_TAG_TOKEN,
        VID_TAG_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    # print(f"tokenizer {tokenizer}")
    return tokenizer


def update_tokenizer_for_s2s(tokenizer, model_type):

    if model_type is None:
        return update_tokenizer(tokenizer)

    from .tokenizer_cosyvoice2 import update_tokenizer_for_cosyvoice2, CosyVoice2Tokenizer
    from .tokenizer_glm4voice import update_tokenizer_for_glm4voice, GLM4VoiceTokenizer
    from .tokenizer_snac import update_tokenizer_for_snac, SNACTokenizer
    from .tokenizer_sensevoice_sparktts import (
        update_tokenizer_for_sensevoice_sparktts,
        SenseVoiceSparkTTSTokenizer,
    )
    from .tokenizer_sensevoice_glm4voice import (
        update_tokenizer_for_sensevoice_glm4voice,
        SenseVoiceGLM4VoiceTokenizer,
    )

    if model_type == "glm4voice":
        return update_tokenizer_for_glm4voice(tokenizer)

    if model_type == "cosyvoice2":
        return update_tokenizer_for_cosyvoice2(tokenizer)

    if model_type == "snac24khz":
        return update_tokenizer_for_snac(tokenizer)

    if model_type == "sensevoice_sparktts":
        return update_tokenizer_for_sensevoice_sparktts(tokenizer)

    if model_type == "sensevoice_glm4voice":
        return update_tokenizer_for_sensevoice_glm4voice(tokenizer)

    raise NotImplementedError


def get_audio_tokenizer(model_name_or_path, model_type, flow_path=None, rank=None):

    if model_type is None:
        return None

    from .tokenizer_cosyvoice2 import update_tokenizer_for_cosyvoice2, CosyVoice2Tokenizer
    from .tokenizer_glm4voice import update_tokenizer_for_glm4voice, GLM4VoiceTokenizer
    from .tokenizer_snac import update_tokenizer_for_snac, SNACTokenizer
    from .tokenizer_sensevoice_sparktts import (
        update_tokenizer_for_sensevoice_sparktts,
        SenseVoiceSparkTTSTokenizer,
    )
    from .tokenizer_sensevoice_glm4voice import (
        update_tokenizer_for_sensevoice_glm4voice,
        SenseVoiceGLM4VoiceTokenizer,
    )

    if model_type == "glm4voice":
        return GLM4VoiceTokenizer(model_name_or_path, flow_path=flow_path, rank=rank)

    if model_type == "cosyvoice2":
        return CosyVoice2Tokenizer(model_name_or_path, rank=rank)

    if model_type == "snac24khz":
        return SNACTokenizer(model_name_or_path, rank=rank)

    if model_type == "sensevoice_sparktts":
        return SenseVoiceSparkTTSTokenizer(model_name_or_path, rank=rank)

    if model_type == "sensevoice_glm4voice":
        return SenseVoiceGLM4VoiceTokenizer(model_name_or_path, flow_path=flow_path, rank=rank)

    raise NotImplementedError
