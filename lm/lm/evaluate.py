from inference import  disambiguate, initialize_disambiguation
from dataset import get_dataset_evaluate

if __name__ == "__main__":
    index, lemmatizer, tokenizer, model, nodes, embeddings = initialize_disambiguation()

    dataset = get_dataset_evaluate()

    for example in iter(dataset["validation"]):
        text = example["text"]
        spans = []
        qids = []
        for entity in example["entities"]:
            spans.append((entity["start"], entity["end"]))
            qids.append(entity["qid"])

        if text.strip() == "" or len(spans) == 0:
            continue

        results = disambiguate(text, spans, 5, nodes, embeddings, index, tokenizer, model, lemmatizer)

        result_qids = []
        result_names = []
        for result in results:
            if len(result) > 0:
                result_qids.append([qid for qid, _, _, _ in result])
                result_names.append([name for _, name, _, _ in result])
            else:
                result_qids.append([])
                result_names.append([])

        breakpoint()
