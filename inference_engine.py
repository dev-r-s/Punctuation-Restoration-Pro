import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
from typing import List, Tuple
import os


class PunctuationRestorer:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------------
        # 1️⃣ Load HuggingFace Model
        # -------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # -------------------------------
        # 2️⃣ Load Runtime Config
        # -------------------------------
        runtime_config_path = os.path.join(model_path, "runtime_config.json")

        if not os.path.exists(runtime_config_path):
            raise FileNotFoundError("runtime_config.json not found.")

        with open(runtime_config_path, "r") as f:
            runtime_config = json.load(f)

        self.max_len = runtime_config["MAX_LEN"]
        self.overlap_stride = runtime_config["OVERLAP_STRIDE"]
        self.label2id = runtime_config["LABEL2ID"]
        self.id2label = runtime_config["ID2LABEL"]

        print(f"Model loaded successfully on {self.device}")

    # -------------------------------------
    # TEXT CHUNKING
    # -------------------------------------
    def _chunk_text(self, tokens: List[str]):

        if len(tokens) <= self.max_len - 2:
            return [(tokens, 0, len(tokens))]

        chunks = []
        start = 0
        effective_max = self.max_len - 2

        while start < len(tokens):
            end = min(start + effective_max, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append((chunk_tokens, start, end))

            if end >= len(tokens):
                break

            start += effective_max - self.overlap_stride

        return chunks

    # -------------------------------------
    # PREDICT CHUNK
    # -------------------------------------
    def _predict_chunk(self, chunk_tokens):

        encoding = self.tokenizer(
            chunk_tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits[0]
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        word_ids = encoding.word_ids(batch_index=0)

        word_predictions = []
        previous_word_id = None

        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous_word_id:
                word_predictions.append(predictions[idx])
                previous_word_id = word_id

        return word_predictions

    # -------------------------------------
    # MERGE OVERLAPPING CHUNKS
    # -------------------------------------
    def _merge_predictions(self, chunks_info, predictions):

        if len(chunks_info) == 1:
            return predictions[0]

        total_words = chunks_info[-1][2]
        final_predictions = [0] * total_words

        for idx, (_, start, end) in enumerate(chunks_info):
            chunk_preds = predictions[idx]

            if idx == 0:
                for i, pred in enumerate(chunk_preds):
                    if start + i < total_words:
                        final_predictions[start + i] = pred
            else:
                for i, pred in enumerate(chunk_preds):
                    if i >= self.overlap_stride and start + i < total_words:
                        final_predictions[start + i] = pred

        return final_predictions

    # -------------------------------------
    # RECONSTRUCT TEXT
    # -------------------------------------
    def _reconstruct_text(self, words, predictions):

        result = []
        capitalize_next = True

        for word, pred_id in zip(words, predictions):

            label = self.id2label[str(pred_id)]

            if capitalize_next:
                word = word.capitalize()
                capitalize_next = False

            result.append(word)

            if label == "COMMA":
                result.append(",")
            elif label == "PERIOD":
                result.append(".")
            elif label == "PERIOD+CAPS":
                result.append(".")
                capitalize_next = True
            elif label == "QM":
                result.append("?")
            elif label == "QM+CAPS":
                result.append("?")
                capitalize_next = True
            elif label == "EXCLAM":
                result.append("!")
            elif label == "EXCLAM+CAPS":
                result.append("!")
                capitalize_next = True

        text = " ".join(result)
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" ?", "?")
        text = text.replace(" !", "!")

        return text

    # -------------------------------------
    # PUBLIC API
    # -------------------------------------
    def restore(self, text: str):

        if not text.strip():
            return text

        words = text.lower().split()

        chunks_info = self._chunk_text(words)

        all_predictions = []
        for chunk_tokens, _, _ in chunks_info:
            preds = self._predict_chunk(chunk_tokens)
            all_predictions.append(preds)

        final_predictions = self._merge_predictions(chunks_info, all_predictions)

        return self._reconstruct_text(words, final_predictions)