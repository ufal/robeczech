import os
from tokenizers import ByteLevelBPETokenizer

class RobeCzechTokenizer:
    def __init__(self, path):
        """
        Initialize tokenizer.
        :param path: Path to folder storing tokenizer files: vocab.json, merges.txt and dict.txt.
        """
        self._tokenizer = ByteLevelBPETokenizer(os.path.join(path, "vocab.json"), os.path.join(path, "merges.txt"))

        vocab = self._tokenizer.get_vocab()
        self.cls_index = vocab.get("[CLS]")
        self.pad_index = vocab.get("[PAD]")
        self.sep_index = vocab.get("[SEP]")
        self.unk_index = vocab.get("[UNK]")
        self.eol_index = vocab.get("[EOL]")
        self.mask_index = vocab.get("[MASK]")
        self.special_tokens = {self.cls_index, self.pad_index, self.sep_index, self.unk_index, self.eol_index, self.mask_index}

        self._append_eol = self.eol_index != self.unk_index

    def encode(self, text1: str, text2: str = None, max_length: int = 512, pad_to_max_length: bool = False, add_special_tokens: bool = True):
        """
        Encode given text using this tokenizer.

        If text2 is provided, it is concatenated after text1, possibly with suitable special symbols.

        :param text1: Required
        :param text2: Optional
        :param max_length: Maximum number of tokens to return.
        :param pad_to_max_length: Whether to pad tokens to max_length if encoded representation is shorter.
        :param add_special_tokens: Whether to add special tokens.
        :return: dict with encoded tokens (input_ids) and attention mask (attention_mask) marking relevant tokens.
        """
        # encode text
        text_encoded = []

        if add_special_tokens:
            text_encoded.append(self.cls_index)

        text_encoded.extend(self._tokenizer.encode(text1).ids)

        if add_special_tokens:
            if self._append_eol:
                text_encoded.append(self.eol_index)
            text_encoded.append(self.sep_index)

        if text2 is not None:
            text_encoded.extend(self._tokenizer.encode(text2).ids)
            if add_special_tokens:
                if self._append_eol:
                    text_encoded.append(self.eol_index)
                text_encoded.append(self.sep_index)

        # attention mask
        attention_mask = [1] * len(text_encoded)

        # pad
        if pad_to_max_length and len(text_encoded) < max_length:
            to_pad_by = max_length - len(text_encoded)
            text_encoded = text_encoded + [self.pad_index] * to_pad_by
            attention_mask = attention_mask + [0] * to_pad_by

        # truncate
        text_encoded = text_encoded[:max_length]
        attention_mask = attention_mask[:max_length]

        return {
            "input_ids": text_encoded,
            "attention_mask": attention_mask,
        }

    def decode(self, decoded_tokens, skip_special_tokens:bool = True):
        if skip_special_tokens:
            decoded_tokens = [token for token in decoded_tokens if token not in self.special_tokens]

        if self.unk_index not in decoded_tokens:
            return self._tokenizer.decode(decoded_tokens)

        decoded = []
        while self.unk_index in decoded_tokens:
            index = decoded_tokens.index(self.unk_index)
            decoded.append(self._tokenizer.decode(decoded_tokens[:index]))
            decoded_tokens = decoded_tokens[index + 1:]
        decoded.append(self._tokenizer.decode(decoded_tokens))
        return "[UNK]".join(decoded)
