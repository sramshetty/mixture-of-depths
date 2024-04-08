# Mixture of Depths (WIP)
An unofficial implementation of ["Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"](https://arxiv.org/abs/2404.02258)


## Details
- Implementing MoD in Llama


## Current Results
- 50 million parameter model
    - Baseline after 1 epoch:
        - Loss: 4.04
        - Samples/sec: 5.83
    - MoD after 1 epoch:
        - Loss: 4.06
        - Samples/sec: 6.77


## TODO
- [ ] Validate
- [ ] Sampling methods
    - [ ] Auxiliary loss
    - [ ] "Second" router