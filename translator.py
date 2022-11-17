import torch
from transformers import BertTokenizer, XLMRobertaTokenizer, DistilBertTokenizer, MT5Tokenizer


class Translator:

    def __init__(
            self,
            model_name: str,
            max_length=100
    ) -> None:
        device = 'cpu'
        model = torch.load(f'models/{model_name}', map_location=device)
        model = model.to(device)
        model.encoder.device = device
        model.decoder.device = device
        self.model = model
        self.tokenizer = self.__init_tokenizer__(model_name)
        self.max_length = max_length


    def __init_tokenizer__(self, model_name):
        if 'distilbert' in model_name.lower():
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        elif 'xlmroberta' in model_name.lower():
            tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        elif 'mt5' in model_name.lower():
            tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
        tokenizer.add_tokens([f"[2{lang}]" for lang in ["en", "it", "es", "de", "fr"]])
        return tokenizer


    def __call__(self, src_sentence: str) -> str:
        self.model.eval()
        src_tensor = (
            self.tokenizer(
                src_sentence,
                return_tensors="pt",
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            .data["input_ids"]
        )
        trg_indexes = self.tokenizer.convert_tokens_to_ids(["[CLS]"])
        with torch.no_grad():
            enc_src = self.model.encoder(src_tensor)
        for _ in range(self.max_length):
            trg_tensor = torch.tensor(trg_indexes).unsqueeze(0)
            with torch.no_grad():
                output, _ = self.model.decoder(trg_tensor, enc_src)
            pred_index = torch.argmax(output[0, -1], dim=-1).item()
            trg_indexes.append(pred_index)
            trg_tokens = self.tokenizer.convert_ids_to_tokens(trg_indexes)
            if trg_tokens[-1] == "[SEP]":
                break
        trg_tokens = trg_tokens[1:-1]
        res_sent = self.tokenizer.convert_tokens_to_string(trg_tokens)
        return res_sent
