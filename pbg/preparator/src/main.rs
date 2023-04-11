use arrow2::io::parquet::read;
use clap::Parser;
use hdf5;
use itertools::Itertools;
use ndarray::arr1;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the Apache Parquet file containing a list of nodes.
    #[arg(long)]
    input_nodes: String,
    /// Path to the Apache Parquet file containing a list of edges.
    #[arg(long)]
    input_graph: String,
    /// Path to the output directory for entity counts.
    #[arg(long)]
    output_entities_dir: String,
    /// Path to the output directory for edges.
    #[arg(long)]
    output_edges_dir: String,
    /// Number of partitions to split the graph into.
    #[arg(long)]
    partitions: usize,
}

fn read_nodes(path: &Path) -> BTreeMap<i64, usize> {
    let mut reader = File::open(path).unwrap();

    let metadata = read::read_metadata(&mut reader).unwrap();
    let schema = read::infer_schema(&metadata).unwrap();

    // we can then read the row groups into chunks
    let nodes = read::FileReader::new(reader, metadata.row_groups, schema, None, None, None)
        .next()
        .unwrap()
        .ok()
        .unwrap()
        .into_arrays()[0]
        .as_any()
        .downcast_ref::<arrow2::array::UInt32Array>()
        .unwrap()
        .to_owned();

    nodes
        .into_iter()
        .enumerate()
        .map(|(i, node)| (node.unwrap() as i64, i))
        .collect()
}

fn read_edges(path: &Path) -> (Vec<i64>, Vec<i64>) {
    let mut reader = File::open(path).unwrap();

    let metadata = read::read_metadata(&mut reader).unwrap();
    let schema = read::infer_schema(&metadata).unwrap();
    let schema = schema.filter(|index, _| index == 0 || index == 2);

    let arrays = read::FileReader::new(reader, metadata.row_groups, schema, None, None, None)
        .next()
        .unwrap()
        .ok()
        .unwrap()
        .into_arrays();

    let lhs = arrays[0]
        .to_owned()
        .as_any()
        .downcast_ref::<arrow2::array::UInt32Array>()
        .unwrap()
        .into_iter()
        .map(|x| *x.unwrap() as i64)
        .collect::<Vec<_>>();
    let rhs = arrays[1]
        .to_owned()
        .as_any()
        .downcast_ref::<arrow2::array::UInt32Array>()
        .unwrap()
        .into_iter()
        .map(|x| *x.unwrap() as i64)
        .collect::<Vec<_>>();

    (lhs, rhs)
}

fn write_edges(path: &Path, lhs: Vec<i64>, rhs: Vec<i64>, rel: Vec<i64>) {
    let file = hdf5::File::create(path).unwrap();
    // let chunk = (50 * 2.pow(20) / 8,);
    file.new_dataset_builder()
        .with_data(&arr1(&lhs))
        // .chunk(chunk)
        .create("lhs")
        .unwrap();
    file.new_dataset_builder()
        .with_data(&arr1(&rhs))
        // .chunk(chunk)
        .create("rhs")
        .unwrap();
    file.new_dataset_builder()
        .with_data(&arr1(&rel))
        // .chunk(chunk)
        .create("rel")
        .unwrap();
    let format_version = file.new_attr::<i64>().create("format_version").unwrap();
    format_version.write_scalar(&1).unwrap();
}

fn write_entities(path: &Path, nodes: usize) {
    fs::write(path, nodes.to_string()).unwrap();
}

fn calculate_offset(idx: usize, partition_size: usize) -> usize {
    let partition = idx / partition_size;
    idx - partition * partition_size
}

fn main() {
    let args = Args::parse();

    let nodes = read_nodes(&Path::new(&args.input_nodes));

    let partition_size = (nodes.len() as f32 / args.partitions as f32).ceil() as usize;

    for i in 0..args.partitions {
        write_entities(
            &Path::new(&args.output_entities_dir).join(format!("entity_count_all_{}.txt", i)),
            if i == args.partitions - 1 {
                if nodes.len() % partition_size == 0 {
                    partition_size
                } else {
                    nodes.len() % partition_size
                }
            } else {
                partition_size
            },
        );
    }

    let (lhs, rhs) = read_edges(&Path::new(&args.input_graph));

    let mut edge_partitions: BTreeMap<(usize, usize), (Vec<i64>, Vec<i64>, Vec<i64>)> =
        BTreeMap::new();
    for (i, j) in (0..args.partitions).cartesian_product(0..args.partitions) {
        edge_partitions.insert((i, j), (vec![], vec![], vec![]));
    }

    for (lhs, rhs) in lhs.into_iter().zip(rhs) {
        let lhs = *nodes.get(&lhs).unwrap();
        let rhs = *nodes.get(&rhs).unwrap();
        let (edges_lhs, edges_rel, edges_rhs) = edge_partitions
            .get_mut(&(lhs / partition_size, rhs / partition_size))
            .unwrap();

        edges_lhs.push(calculate_offset(lhs, partition_size) as i64);
        edges_rel.push(0i64);
        edges_rhs.push(calculate_offset(rhs, partition_size) as i64);
    }

    edge_partitions
        .into_par_iter()
        .for_each(|((i, j), (lhs, rel, rhs))| {
            write_edges(
                &Path::new(&args.output_edges_dir).join(format!("edges_{}_{}.h5", i, j)),
                lhs,
                rhs,
                rel,
            );
        });
}
