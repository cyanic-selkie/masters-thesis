def get_torchbiggraph_config():

    config = dict(
        # I/O data
        entity_path="entities",
        edge_paths=[
            "edges",
        ],
        checkpoint_path="model/wikidata",
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
        dimension=128,
        global_emb=False,
        comparator="dot",
        bias=False,
        # Training
        num_epochs=4,
        batch_size=50000,
        num_edge_chunks=10,
        num_batch_negs=5000,
        num_uniform_negs=5000,
        loss_fn="softmax",
        lr=0.1,
        # Evaluation during training
        eval_fraction=0,
        # GPU
        num_gpus=2,
    )

    return config
