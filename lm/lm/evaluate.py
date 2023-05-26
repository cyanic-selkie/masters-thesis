from inference import disambiguate, initialize_disambiguation
from dataset import get_dataset_evaluate
from tqdm import tqdm
import argparse
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",
                        choices=["validation", "test"],
                        type=str,
                        required=True)
    parser.add_argument("--errors", type=str, required=True)
    args = parser.parse_args()

    resources = initialize_disambiguation()

    index, lemmatizer, tokenizer, model, embeddings, filter = resources

    dataset = get_dataset_evaluate()

    micro_top5_tp = 0
    micro_top1_tp = 0
    micro_all = 0

    macro_top5_all = []
    macro_top1_all = []

    errors = []
    for i, document in tqdm(enumerate(dataset[args.split]),
                            total=len(dataset[args.split])):
        text = document["text"]
        spans = []
        qids = []
        for entity in document["entities"]:
            if entity["qid"] is not None and entity["tag"] != 4:
                spans.append((entity["start"], entity["end"]))
                qids.append(entity["qid"])

        if len(spans) == 0:
            continue

        predictions = disambiguate(text, spans, 5, embeddings, index,
                                   tokenizer, model, lemmatizer, filter)
        prediction_qids = []
        prediction_names = []
        for prediction in predictions:
            prediction_qids.append([qid for qid, _, _, _ in prediction])
            prediction_names.append([name for _, name, _, _ in prediction])

        document_top5_tp = 0
        document_top1_tp = 0
        document_all = 0
        for qid, prediction_qid, prediction_name, (x, y) in zip(
                qids, prediction_qids, prediction_names, spans):
            if qid in prediction_qid:
                micro_top5_tp += 1
                document_top5_tp += 1

            if len(prediction_qid) > 0 and qid == prediction_qid[0]:
                micro_top1_tp += 1
                document_top1_tp += 1
            elif len(prediction_qid) > 0 and qid != prediction_qid[0]:
                errors.append((text, x, y, qid, text[x:y], prediction_qid,
                               prediction_name))

            micro_all += 1
            document_all += 1

        macro_top5_all.append(document_top5_tp / document_all)
        macro_top1_all.append(document_top1_tp / document_all)

        with open(args.errors, "w") as out:
            writer = csv.writer(out)
            writer.writerow([
                "text", "start", "end", "qid", "surface_form", "predicted_qid",
                "predicted_name"
            ])
            writer.writerows(errors)

    print(f"Macro Top 5 accuracy: {sum(macro_top5_all) / len(macro_top5_all)}")
    print(f"Macro Top 1 accuracy: {sum(macro_top1_all) / len(macro_top1_all)}")
    print(f"Micro Top 5 accuracy: {micro_top5_tp / micro_all}")
    print(f"Micro Top 1 accuracy: {micro_top1_tp / micro_all}")
