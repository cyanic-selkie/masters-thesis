def get_torchbiggraph_config():

    config = dict(
        # I/O data
        entity_path="data/entities",
        edge_paths=[
            "data/edges",
        ],
        checkpoint_path="model/wikidata-cos-512",
        # Graph structure
        entities={"all": {"num_partitions": 10}},
        relations=[
            {
                "name": "all_edges",
                "lhs": "all",
                "rhs": "all",
                "operator": "none",
            }
        ],
        # Scoring model
        dimension=512,
        global_emb=False,
        comparator="cos",
        bias=False,
        # Training
        num_epochs=4,
        batch_size=10000,
        num_edge_chunks=10,
        num_batch_negs=500,
        num_uniform_negs=500,
        loss_fn="softmax",
        lr=0.1,
        # Evaluation during training
        eval_fraction=0,
        # GPU
        num_gpus=2,
    )

    return config
