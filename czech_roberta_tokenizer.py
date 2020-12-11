import os
from tokenizers.implementations import ByteLevelBPETokenizer

class CzechRobertaTokenizer:
    def __init__(self, path):
        '''
        Initialize tokenizer.
        :param path: Path to folder storing tokenizer files: vocab.json, merges.txt and dict.txt.
        '''
        self.tokenizer = ByteLevelBPETokenizer(os.path.join(path, "vocab.json"), os.path.join(path, "merges.txt"))

        self.bos_index = 0
        self.pad_index = 1
        self.eos_index = 2
        self.unk_index = 3
        self.num_special_tokens = 4

        self.internal_token_mapping, self.internal_token_mapping_inverse = self._build_internal_token_mapping(path)

        self.eol_index = 4
        self.mask_index = len(self.internal_token_mapping) - 1

    def _build_internal_token_mapping(self, path):
        d = {}
        d_inverse = {}
        with open(os.path.join(path, "dict.txt")) as reader:
            for line_i, line in enumerate(reader):
                line = line.strip()
                k, _ = line.split(' ')
                k = int(k)
                d[k] = line_i + self.num_special_tokens
                d_inverse[line_i + self.num_special_tokens] = k

        return d, d_inverse

    def encode(self, text1: str, text2: str = None, max_length: int = 512, pad_to_max_length: bool = True,
               append_eol_symbol: bool = True):
        '''
        Encode given text using this tokenizer.

        If text2 is provided, it is concatenated to text1 with starting <s> and ending </s> and is marked by 1's
        in token_type_ids.

        :param text1: Required
        :param text2: Optional
        :param max_length: Maximum number of tokens to return.
        :param pad_to_max_length: Whether to pad tokens to max_length if encoded representation is shorter.
        :param append_eol_symbol: Whether to append \n to the text.
        :return: dict with encoded tokens (input_ids), attention mask marking relevant tokens and token type ids that
        distinguish between tokens from text1 and tokens from text2.
        '''
        text_encoded = [self.bos_index] + [self.internal_token_mapping.get(id, self.unk_index) for id in
                                           self.tokenizer.encode(text1).ids]
        text1_encoded_len = len(text_encoded) + 1  # include not yet added eos

        if text2:
            text2_encoded = [self.eos_index, self.bos_index] + [self.internal_token_mapping.get(id, self.unk_index) for
                                                                id in self.tokenizer.encode(text2).ids]
            text2_encoded_len = len(text2_encoded) + int(append_eol_symbol)
            text_encoded.extend(text2_encoded)
        else:
            text1_encoded_len += int(append_eol_symbol)

        if append_eol_symbol:
            text_encoded.append(self.eol_index)

        text_encoded.append(self.eos_index)
        attention_mask = [1] * len(text_encoded)
        token_type_ids = [0] * text1_encoded_len

        if text2:
            token_type_ids.extend([1] * text2_encoded_len)

        # pad
        if pad_to_max_length:
            to_pad_by = max_length - len(text_encoded)
            text_encoded = text_encoded + [self.pad_index] * to_pad_by
            attention_mask = attention_mask + [0] * to_pad_by
            token_type_ids = token_type_ids + [1] * to_pad_by

        # truncate
        text_encoded = text_encoded[:max_length]
        attention_mask = attention_mask[:max_length]
        token_type_ids = token_type_ids[:max_length]

        return_dict = {
            'input_ids': text_encoded,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

        return return_dict

    def decode(self, decoded_tokens):
        decoded_tokens = [self.internal_token_mapping_inverse[id] for id in decoded_tokens if
                          id >= self.num_special_tokens]
        return self.tokenizer.decode(decoded_tokens)
