# README 

This is for releasing the source code of the paper "Query-Efficient Decision-Based Sparse Attacks Against Black-Box Machine Learning Models" 

Archived Version: [SparseEvo Attack](https://openreview.net/forum?id=73MEhZ0anV&fbclid=IwAR3Jp9ch0L9T-V5e5K3BHDAG3uVMnjc4_DR3sxKfVVbrWdCbaMqQZB84XQE)

The project is published as part of the following paper and if you re-use our work, please cite the following paper:


```
@inproceedings{vo2022,
title={Query-Efficient Decision-Based Sparse Attacks Against Black-Box Machine Learning Models},
author={Viet Quoc Vo and Ehsan Abbasnejad and Damith C. Ranasinghe},
year = {2022},
journal = {International Conference on Learning Representations (ICLR)},
}
```

The source code is written mostly on *Python 3* and *Pytorch*, so please help to download and install Python3 and Pytorch beforehand.

# Requirements

To install the requirements for this repo, run the following command: 
```
git clone https://github.com/SparseEvoAttack/SparseEvoAttack.github.io.git
cd SparseEvoAttack
pip3 install -r requirements.txt
```

# Run the SparseEvoAttack

There are two ways to run the method:

- The first way is to run step-by-step with the Jupyter Notebook file *SparseEvoAttack.ipynb* in the main folder. 

- The second way is to run the *test.py* file. This is to run the RamBoAttack on the whole test set for that task.: 

```python
# For example, to run SparseEvoAttack on CIFAR10
cd main
python3 test.py
```
  
## TODO 
- [ ] add the testing code
