import avro
import json
from avro.datafile import DataFileReader
from avro.io import DatumReader
import pickle

schema = {
    "type":
    "record",
    "name":
    "wiki2qid",
    "fields": [{
        "name": "title",
        "type": "string"
    }, {
        "name": "pageid",
        "type": "int"
    }, {
        "name": "qid",
        "type": ["null", "int"]
    }]
}

schema = avro.schema.Parse(json.dumps(schema))

with open('../../data/processed/en/wiki2qid.avro', 'rb') as f:
    qids = set()
    reader = DataFileReader(f, DatumReader())
    for record in reader:
        pageid = record["pageid"]
        if pageid is not None:
            qid = record["qid"]
            qids.add(qid)
    reader.close()

with open('qid-filter.pickle', 'wb') as handle:
    pickle.dump(qids, handle, protocol=pickle.HIGHEST_PROTOCOL)
