from typing import Optional, Tuple, Union, List, Dict
import tantivy

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel, AutoModelForTokenClassification, pipeline
from transformers.utils import ModelOutput
from model import NELModel
import spacy
import unicodedata
from pprint import pprint as pp

def get_span_index(x: int, y: int, length):
    idx = ((length - x) * (length - x + 1)) // 2
    idx = -(idx - (y - x))
    return idx 

def get_candidates(index, lemmatizer, query, nodes):
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
        if qid in nodes:
            qids.append(qid)
            scores.append(score)
            names.append(name)

    return qids, scores, names

def disambiguate(text: str, spans: List[Tuple[int, int]], top_k: int, nodes: Dict[int, int], embeddings, index, tokenizer, model, lemmatizer):
    inputs = tokenizer(text, return_offsets_mapping=True, truncation=True, return_tensors="pt")

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

    result = []
    for i, (x, y) in enumerate(token_spans):
        mention = tokenizer.decode(inputs["input_ids"][0][x:y + 1])
        embedding = output[0][i].detach().numpy()
        candidates, fts_scores, names = get_candidates(index, lemmatizer, mention, nodes)

        qids = []
        if len(candidates) > 0:
            scores = np.dot(embeddings[[nodes[qid] for qid in candidates]], embedding)
            indices = np.argpartition(scores, -min(top_k, len(scores)))
            indices = indices[-min(top_k, len(scores)):]

            qids = [(candidates[i], names[i], fts_scores[i], scores[i]) for i in indices]

        qids.sort(key=lambda x: x[3])

        result.append(qids)

    return result

def initialize_disambiguation():
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_integer_field("qid", stored=True, indexed=True)
    schema_builder.add_text_field("name", stored=True, index_option='basic')
    schema_builder.add_text_field("alias", stored=True)
    schema = schema_builder.build()

    index = tantivy.Index(schema, path="data/en")

    lemmatizer = spacy.load("en_core_web_lg", disable=["senter", "parser", "ner"])

    tokenizer = AutoTokenizer.from_pretrained("models/mapper-3-3")
    model = NELModel.from_pretrained("models/mapper-3-3")

    nodes = {qid: i for i, qid in enumerate(pq.read_table("data/nodes.parquet")["qid"].to_pylist())}
    embeddings = np.memmap("data/embeddings.npy", np.float32, "r", shape=(len(nodes), 128))

    return index, lemmatizer, tokenizer, model, nodes, embeddings

def initialize_detection():
    ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
    ner = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="first") 

    return ner

if __name__ == "__main__":
    # text = "Michael Jordan (born February 25, 1956) is an American scientist, professor at the University of California, Berkeley and researcher in machine learning, statistics, and artificial intelligence." 
    # text = "In November 1939, the United States was taking measures to assist China and the Western Allies and amended the Neutrality Act to allow \"cash and carry\" purchases by the Allies. In 1940, following the German capture of Paris, the size of the United States Navy was significantly increased. In September the United States further agreed to a trade of American destroyers for British bases. Still, a large majority of the American public continued to oppose any direct military intervention in the conflict well into 1941. In December 1940, Roosevelt accused Hitler of planning world conquest and ruled out any negotiations as useless, calling for the United States to become an \"arsenal of democracy\" and promoting Lend-Lease programmes of military and humanitarian aid to support the British war effort, which was later extended to the other Allies, including the Soviet Union. The United States started strategic planning to prepare for a full-scale offensive against Germany."
    text = "The first decade of the 20th century saw increasing diplomatic tension between the European great powers. This reached breaking point on 28 June 1914, when a Bosnian Serb named Gavrilo Princip assassinated Archduke Franz Ferdinand, heir to the Austro-Hungarian throne. Austria-Hungary held Serbia responsible, and declared war on 28 July. Russia came to Serbia's defence, and by 4 August, defensive alliances had drawn in Germany, France, and Britain, with the Ottoman Empire joining the war in November."
    #text = "Michael Jordan is an American scientist, professor at the University of California, Berkeley and researcher in machine learning, statistics, and artificial intelligence. However, that is not who we are refering to. We are, in fact, talking about Michael Jordan who played 15 seasons for the Chicago Bulls and is considered to be one of the best basketball players of all time."
    #text = "Michael Jeffrey Jordan, also known by his initials MJ, is an American former professional basketball player and businessman."
    #text = "In the war, American patriot forces were supported by the Kingdom of France and the Kingdom of Spain. The British, in turn, were supported by Hessian soldiers from present-day Germany, most Native Americans, Loyalists, and freedmen. The conflict was fought in North America, the Caribbean, and the Atlantic Ocean."
    #text = "This irony will not be lost on Queen Camilla. The former Lady Elizabeth Bowes-Lyon transformed the fortunes of the monarchy: first as a steadfast supporter to her stumbling, stammering husband King George VI during the dark days of World War II and then, through the long years of widowhood, to her daughter, Queen Elizabeth II."
    #text = "Russian state television showed Putin looking bloated as he engaged in discussions with Economic Development Minister Maxim Reshetnikov at his official residence, less than 48 hours after fireballs erupted above the Kremlin's roof."
    #text = "Hunter daughter Finnegan going to King's coronation with Jill Biden."
    ner = initialize_detection()
    mentions = ner(text)

    surface_forms = []
    spans = []
    for mention in mentions:
        surface_forms.append(mention["word"])
        spans.append((mention["start"], mention["end"]))

    index, lemmatizer, tokenizer, model, nodes, embeddings = initialize_disambiguation()

    results = disambiguate(text, spans, 5, nodes, embeddings, index, tokenizer, model, lemmatizer)

    for surface_form, result in zip(surface_forms, results):
        print(surface_form)
        pp(result)
