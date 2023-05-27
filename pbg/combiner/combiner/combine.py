import argparse
import h5py
import numpy as np
from os.path import join


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--embeddings-dir", type=str, required=True)
	parser.add_argument("--nodes", type=int, required=True)
	parser.add_argument("--dimensions", type=int, required=True)
	parser.add_argument("--output", type=str, required=True)
	parser.add_argument("--partitions", type=int, required=True)
	args = parser.parse_args()

	embeddings = np.memmap(args.output,
	                       dtype=np.float32,
	                       mode="w+",
	                       shape=(args.nodes, args.dimensions))
	idx = 0
	for i in range(args.partitions):
		with h5py.File(join(args.embeddings_dir, f"embeddings_all_{i}.v40.h5"),
		               "r") as f:
			partition = f["embeddings"][()]
			embeddings[idx:idx + partition.shape[0]] = partition
			idx += partition.shape[0]


if __name__ == "__main__":
	main()
