from typing import Optional, Tuple, Union, List, Dict
from collections import defaultdict
import unicodedata
from pprint import pprint as pp

import numpy as np
import tantivy
import torch
from transformers import AutoTokenizer
from model import ELModel
import spacy

def get_span_index(x: int, y: int, length):
    idx = ((length - x) * (length - x + 1)) // 2
    idx = -(idx - (y - x))
    return idx 

def get_candidates(index, lemmatizer, query):
    searcher = index.searcher()
    query = unicodedata.normalize("NFC", query)
    query = [token.lemma_.lower() for token in lemmatizer(query) if not token.is_stop and not token.is_punct]

    query = " ".join([f"+{token}" for token in  query])
    query = index.parse_query(f'{query}', ["alias"])
    docs = searcher.search(query, 100000).hits

    qids = []
    indices = []
    scores = []
    names = []
    for score, addr in docs:
        doc = searcher.doc(addr)
        qid = doc["qid"][0]
        idx = doc["idx"][0]
        name = doc["name"][0]
        qids.append(qid)
        indices.append(idx)
        scores.append(score)
        names.append(name)

    return qids, indices, scores, names

def disambiguate(text: str, spans: List[Tuple[int, int]], top_k: int, embeddings, index, tokenizer, model, lemmatizer) -> List[List[Tuple[int, str, float, float]]]:
    stride = 256
    inputs = tokenizer(text, truncation=True, stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True, return_tensors="pt", padding='longest')

    offset_mapping = inputs.pop("offset_mapping")
    _ = inputs.pop("overflow_to_sample_mapping")

    document_token_spans = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        
        spans = []

        for span_start, span_end in mentions:
            # Start token index of the current span in the text.
            token_start_index = 0
            
            while offsets[token_start_index][0] == 0 and offsets[token_start_index][1] == 0:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            
            while offsets[token_end_index][0] == 0 and offsets[token_end_index][1] == 0:
                token_end_index -= 1
            
            # Detect if the span is out of the sequence length.
            if (offsets[token_start_index][0] <= span_start and offsets[token_end_index][1] >= span_end):
                # Move the token_start_index and token_end_index to the two ends of the span.
                # Note: we could go after the last offset if the span is the last word (edge case).
                try:
                    while offsets[token_start_index][0] < span_start:
                        token_start_index += 1

                    while offsets[token_end_index][1] > span_end:
                        token_end_index -= 1
                except Exception:
                    continue

                spans.append((token_start_index, token_end_index))
        
        if len(spans) > 0:
            spans = list(sorted(spans))

        document_token_spans.append(spans)
                    
    max_length = max(len(spans) for spans in document_token_spans)
    padded_document_token_spans = [spans + [(0, 0)] * (max_length - len(spans)) for spans in document_token_spans]

    span_indices = torch.tensor(padded_document_token_spans)

    document_predicted_embeddings = model(**inputs, span_indices=span_indices)["embeddings"]

    predictions = defaultdict(lambda: [])
    order = []
    for i, (token_spans, predicted_embeddings, input_ids) in enumerate(zip(document_token_spans, document_predicted_embeddings, inputs["input_ids"])):
        for j, ((x, y), predicted_embedding) in enumerate(zip(token_spans, predicted_embeddings)):
            if (x == 0 and y == 0):
                break

            mention = tokenizer.decode(input_ids[x:y + 1])
            candidate_qids, candidate_indices, fts_scores, preferred_names = get_candidates(index, lemmatizer, mention)

            if len(candidate_qids) == 0:
                continue

            scores = np.dot(embeddings[candidate_indices], predicted_embedding.detach().numpy())
            top_k_indices = np.argpartition(scores, -min(top_k, len(scores)))
            top_k_indices = top_k_indices[-min(top_k, len(scores)):]

            prediction = [(candidate_qids[k], preferred_names[k], fts_scores[k], scores[k]) for k in reversed(top_k_indices)]
            key = (x + stride * i, y + stride * i)
            predictions[key].extend(prediction)
            predictions[key].sort(key=lambda x: x[3], reverse=True)
            predictions[key] = predictions[key][:top_k]
            if not key in order:
                order.append(key)

    return [predictions[key] for key in order]

def initialize_disambiguation():
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_integer_field("qid", stored=True, indexed=True)
    schema_builder.add_integer_field("idx", stored=True, indexed=True)
    schema_builder.add_text_field("name", stored=True, index_option='basic')
    schema_builder.add_text_field("alias", stored=True)
    schema = schema_builder.build()

    index = tantivy.Index(schema, path="../../data/processed/en")

    lemmatizer = spacy.load("en_core_web_lg", disable=["senter", "parser", "ner"])

    tokenizer = AutoTokenizer.from_pretrained("../../data/models/mapper-pretrained")
    model = ELModel.from_pretrained("../../data/models/mapper-pretrained")

    embeddings = np.memmap("../../data/processed/embeddings.npy", np.float32, "r", shape=(99385029, 128))

    return index, lemmatizer, tokenizer, model, embeddings

if __name__ == "__main__":
    text: str = """Cofinec plunges on H1 results . Emese Bartha BUDAPEST 1996-08-30 Shares of France-registered printed packaging company Cofinec S.A. plunged sharply on the Budapest Stock Exchange ( BSE ) on Friday , despite a mostly reassuring forecast by the group . Cofinec 's Global Depositary Receipts ( GDRs ) opened at 5,200 forints on the BSE , down 600 from Thursday 's close , following the release of its first half results this morning . Cofinec CEO Stephen Frater told reporters in a conference call from Vienna on Friday before the opening of the bourse that he expects a stronger second half , although the group will not be able to achieve its annual profit goal . " We will not achieve the full 37 million French franc ( net ) profit forecast , " Frater said . " Obviously , we cannot make up the unexpected decrease that has been experienced in the first half of the year . " Frater declined to give a forecast for the full year , ahead of a supervisory board meeting next week . Cofinec , the first foreign company to list on the Budapest bourse , released its consolidated first half figures ( IAS ) this morning . In the conference call , Frater said he regarded Cofinec GDRs -- which are trading below their issue price of 6,425 forints -- as a buying opportunity . " Obviously , at some point it represents a buying opportunity , " Frater said . " I think the reality is that we operate in emerging markets , emerging markets tend to be more volatile . " " My message is that the fundamental strategy of the company , its fundamental market position has not changed . " The group , which operates in Hungary , Poland and the Czech Republic , reported an operating profit before interest of 21.8 million French francs compared to 34.1 million in the same six months of 1995 . Net profit for the January-June 1996 period was 2.1 million French francs , down from 10.3 million in the first six months of 1995 , with the bulk of this decline attributable to the performance of Petofi , one of its Hungarian units . Cofinec said Petofi general manager Laszlo Sebesvari had submitted his resignation and will be leaving Petofi but will remain on Petofi 's board of directors . " Until a new general manager of Petofi is appointed ... I will in fact move to Kecskemet ( site of Petofi printing house ) for the interim and will serve as acting chief executive officer of Petofi , " Frater said . -- Budapest newsroom ( 36 1 ) 327 4040"""
    mentions: List[Tuple[int, int]] = [(45, 53), (155, 178), (181, 184), (329, 332), (500, 506), (705, 711), (1031, 1039), (1605, 1612), (1615, 1621), (1630, 1644), (1708, 1714), (1840, 1846), (1998, 2007), (2256, 2265), (2396, 2404)]
    qids: List[int] = [1781, 851259, 851259, 851259, 1741, 142, 1781, 28, 36, 213, 142, 142, 28, 171357, 1781]

    # text = "The first decade of the 20th century saw increasing diplomatic tension between the European great powers. This reached breaking point on 28 June 1914, when a Bosnian Serb named Gavrilo Princip assassinated Archduke Franz Ferdinand, heir to the Austro-Hungarian throne. Austria-Hungary held Serbia responsible, and declared war on 28 July. Russia came to Serbia's defence, and by 4 August, defensive alliances had drawn in Germany, France, and Britain, with the Ottoman Empire joining the war in November."
    # mentions = [(244, 260), (339, 345), (422, 429)]
    # qids: List[int] = [28513, 34266, 43287]

    for (x, y), qid in zip(mentions, qids):
        print(text[x:y], qid)

    index, lemmatizer, tokenizer, model, embeddings = initialize_disambiguation()

    predictions: List[List[Tuple[int, str, float, float]]] = disambiguate(text, mentions, 5, embeddings, index, tokenizer, model, lemmatizer)

    for mention, qid, prediction in zip(mentions, qids, predictions):
        print(mention, qid)
        print(prediction)