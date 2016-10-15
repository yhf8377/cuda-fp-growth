# CUDA based FP-Growth Algorithm Implementation

This project is an implementation of the FP-Growth algorihm [1] for Frequent Itemset Mining (FIM) using CUDA parallelism with the goal to boost its performance and apply it to large data sets.

To facilitate GPU computation this implementation uses a 'top-down' approach outlined in [2]. This implementation also utilizes the parallel radix tree construction method proposed in [3] and the parallel improvement to the top-down approach suggested by [4].

# References
1. Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. *In ACM Sigmod Record* (Vol. 29, No. 2, pp. 1-12). ACM.
2. Wang, K., Tang, L., Han, J., & Liu, J. (2002, May). Top down fp-growth for association rule mining. *In Pacific-Asia Conference on Knowledge Discovery and Data Mining* (pp. 334-340). Springer Berlin Heidelberg.
3. Karras, T. (2012, June). Maximizing parallelism in the construction of BVHs, octrees, and k-d trees. *In Proceedings of the Fourth ACM SIGGRAPH/Eurographics conference on High-Performance Graphics* (pp. 33-37). Eurographics Association.
4. Wang, F., & Yuan, B. (2014, December). Parallel frequent pattern mining without candidate generation on GPUs. *In 2014 IEEE International Conference on Data Mining Workshop* (pp. 1046-1052). IEEE.
