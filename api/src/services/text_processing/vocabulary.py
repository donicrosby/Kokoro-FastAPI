"""Vocabulary utilities. Tokenization uses kokoro_onnx.tokenizer (see text_processor.tokenize)."""


def get_vocab():
    """Legacy vocab dict (symbol -> id). Prefer kokoro-onnx tokenizer for tokenize."""
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»"" '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    return {symbol: i for i, symbol in enumerate(symbols)}


VOCAB = get_vocab()


def decode_tokens(tokens: list[int]) -> str:
    """Convert token IDs back to phonemes string

    Args:
        tokens: List of token IDs

    Returns:
        String of phonemes
    """
    # Create reverse mapping
    id_to_symbol = {i: s for s, i in VOCAB.items()}
    return "".join(id_to_symbol[t] for t in tokens)
