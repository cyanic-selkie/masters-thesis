from typing import Optional, Tuple, Union, List, Dict

# import numpy as np
# import tantivy
# import torch
# from transformers import AutoTokenizer
# from model import ELModel
# import spacy
import unicodedata
from pprint import pprint as pp

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
    scores = []
    names = []
    for score, addr in docs:
        doc = searcher.doc(addr)
        qid = doc["qid"][0]
        name = doc["name"][0]
        qids.append(qid)
        scores.append(score)
        names.append(name)

    return qids, scores, names

def disambiguate(text: str, spans: List[Tuple[int, int]], top_k: int, embeddings, index, tokenizer, model, lemmatizer) -> List[List[Tuple[int, str, float, float]]]:
    inputs = tokenizer(text, truncation=True, max_length=512, stride=256, return_overflowing_tokens=True, return_offsets_mapping=True, return_tensors="pt", padding=True)

    token_spans = []
    for span in spans:
        x, y = span

        start = 0
        end = 0
        for i, (s, e) in enumerate(inputs["offset_mapping"][0]):
            if s == 0 and e == 0:
                continue

            if s >= 512:
                break

            # TODO make this more robust
            if s == x:
                start = i
            if e == y:
                end = i  

        if start != 0 and end != 0:
            token_spans.append((start, end))
            #token_spans.append(get_span_index(start, end, inputs["input_ids"].shape[1]))

    del inputs["offset_mapping"]
 
    #span_indices = torch.linspace(0, inputs["input_ids"].shape[1] - 1, inputs["input_ids"].shape[1], dtype=torch.int32)
    #span_indices = torch.combinations(span_indices, 2, with_replacement=True)
    span_indices = torch.tensor(token_spans)
    span_indices = span_indices.unsqueeze(0)

    output = model(**inputs, span_indices=span_indices)["embeddings"]

    predictions= []
    for i, (x, y) in enumerate(token_spans):
        mention = tokenizer.decode(inputs["input_ids"][0][x:y + 1])
        embedding = output[0][i].detach().numpy()
        candidates, candidate_indices, fts_scores, names = get_candidates(index, lemmatizer, mention)

        prediction= []
        if len(candidates) > 0:
            scores = np.dot(embeddings[candidate_indices], embedding)
            indices = np.argpartition(scores, -min(top_k, len(scores)))
            indices = indices[-min(top_k, len(scores)):]

            prediction = [(candidates[i], names[i], fts_scores[i], scores[i]) for i in indices]

        prediction.sort(key=lambda x: x[3])

        predictions.append(prediction)

    return predictions

def initialize_disambiguation():
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_integer_field("qid", stored=True, indexed=True)
    schema_builder.add_integer_field("idx", stored=True, indexed=True)
    schema_builder.add_text_field("name", stored=True, index_option='basic')
    schema_builder.add_text_field("alias", stored=True)
    schema = schema_builder.build()

    index = tantivy.Index(schema, path="data/en")

    lemmatizer = spacy.load("en_core_web_lg", disable=["senter", "parser", "ner"])

    tokenizer = AutoTokenizer.from_pretrained("models/mapper")
    model = ELModel.from_pretrained("models/mapper")

    embeddings = np.memmap("data/embeddings.npy", np.float32, "r", shape=(99385029, 128))

    return index, lemmatizer, tokenizer, model, embeddings

if __name__ == "__main__":
    text: str = """Cofinec plunges on H1 results . Emese Bartha BUDAPEST 1996-08-30 Shares of France-registered printed packaging company Cofinec S.A. plunged sharply on the Budapest Stock Exchange ( BSE ) on Friday , despite a mostly reassuring forecast by the group . Cofinec 's Global Depositary Receipts ( GDRs ) opened at 5,200 forints on the BSE , down 600 from Thursday 's close , following the release of its first half results this morning . Cofinec CEO Stephen Frater told reporters in a conference call from Vienna on Friday before the opening of the bourse that he expects a stronger second half , although the group will not be able to achieve its annual profit goal . " We will not achieve the full 37 million French franc ( net ) profit forecast , " Frater said . " Obviously , we cannot make up the unexpected decrease that has been experienced in the first half of the year . " Frater declined to give a forecast for the full year , ahead of a supervisory board meeting next week . Cofinec , the first foreign company to list on the Budapest bourse , released its consolidated first half figures ( IAS ) this morning . In the conference call , Frater said he regarded Cofinec GDRs -- which are trading below their issue price of 6,425 forints -- as a buying opportunity . " Obviously , at some point it represents a buying opportunity , " Frater said . " I think the reality is that we operate in emerging markets , emerging markets tend to be more volatile . " " My message is that the fundamental strategy of the company , its fundamental market position has not changed . " The group , which operates in Hungary , Poland and the Czech Republic , reported an operating profit before interest of 21.8 million French francs compared to 34.1 million in the same six months of 1995 . Net profit for the January-June 1996 period was 2.1 million French francs , down from 10.3 million in the first six months of 1995 , with the bulk of this decline attributable to the performance of Petofi , one of its Hungarian units . Cofinec said Petofi general manager Laszlo Sebesvari had submitted his resignation and will be leaving Petofi but will remain on Petofi 's board of directors . " Until a new general manager of Petofi is appointed ... I will in fact move to Kecskemet ( site of Petofi printing house ) for the interim and will serve as acting chief executive officer of Petofi , " Frater said . -- Budapest newsroom ( 36 1 ) 327 4040"""
    mentions: List[Tuple[int, int]] = [(45, 53), (155, 178), (181, 184), (329, 332), (500, 506), (705, 711), (1031, 1039), (1605, 1612), (1615, 1621), (1630, 1644), (1708, 1714), (1840, 1846), (1998, 2007), (2256, 2265), (2396, 2404)]
    qids: List[int] = [1781, 851259, 851259, 851259, 1741, 142, 1781, 28, 36, 213, 142, 142, 28, 171357, 1781]

    for (x, y), qid in zip(mentions, qids):
        print(text[x:y], qid)

    exit()

    index, lemmatizer, tokenizer, model, embeddings = initialize_disambiguation()

    predictions: List[List[Tuple[int, str, float, float]]] = disambiguate(text, mentions, 5, embeddings, index, tokenizer, model, lemmatizer)

    for mention, qid, prediction in zip(mentions, qids, predictions):
        print(mention, qid, prediction[-1] if len(prediction) > 0 else None)