# Hierarchical Navigable Small Words (HNSW)

I am trying to implement this algorithm in Rust for learning purposes. My implementation in Python can be found [here](https://github.com/Gumo-A/hnsw).

At the current commit, the package can build the index and query using a vector as input.

Tests against brute-forced NNs of the GloVe dataset show recall@10 levels of ~0.99 for very fast query times (~300 requests/sec).

I'll document how to use the binaries later.

The original paper can be found [here](https://arxiv.org/pdf/1603.09320.pdf).
