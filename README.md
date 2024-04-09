# Mixture of Depths (WIP)
An unofficial implementation of ["Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"](https://arxiv.org/abs/2404.02258)


## Details
- Implementing MoD in Llama


## Current Results
- 50 million parameter model
    - Baseline after 1 epoch:
        - Loss: -
        - Samples/sec: -
    - MoD w/ Auxiliary Loss after 1 epoch:
        - Loss: 3.82
        - Samples/sec: 7.84


## TODO
- [ ] Validate
- [ ] Sampling methods
    - [x] Auxiliary loss
    - [ ] "Second" router


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
