# Mixture of Depths
An unofficial implementation of ["Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"](https://arxiv.org/abs/2404.02258)


## Setup
- First follow instructions for setting up your environment for Llama 2 [here](https://github.com/meta-llama/llama).
- Then:
```bash
pip install einops
```


## Details
- Implementing MoD in Llama 2
- `MoD_training.ipynb` demonstrates training and was uses for the results below.
- `MoD_sampling.ipynb` demonstrates generation which each method.
    - MoD with an auxiliary router doesn't seem to work as well as with just an auxiliary loss (probably a bug).


## Results
- 50 million parameter model
    - C4
        - Baseline after 1 epoch:
            - Loss: 3.73
            - Samples/sec: 6.79
        - MoD w/ Auxiliary Loss after 1 epoch:
            - Loss: 3.81
            - Samples/sec: 8.15
        - MoD w/ Auxiliary Router after 1 epoch:
            - Loss: 4.19
            - Samples/sec: 7.64
    - Tiny Stories
        - Baseline after 5 epochs:
            - Loss: 2.46
            - Samples/sec: 11.38
        - MoD w/ Auxiliary Loss after 5 epochs:
            - Loss: 2.54
            - Samples/sec: 13.08
        - MoD w/ Auxiliary Router after 5 epochs:
            - Loss: 2.53
            - Samples/sec: 12.43

## TODO
- [x] Validate
- [x] Sampling methods
    - [x] Auxiliary loss
    - [x] "Second" router
        - [ ] bug?


## Citations
```bibtex
@misc{raposo2024mixtureofdepths,
    title={Mixture-of-Depths: Dynamically allocating compute in transformer-based language models}, 
    author={David Raposo and Sam Ritter and Blake Richards and Timothy Lillicrap and Peter Conway Humphreys and Adam Santoro},
    year={2024},
    eprint={2404.02258},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
