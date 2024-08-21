# PIPPIN

**Particles Into Particles with Permutation Invariant Network**

This repository contains the code to reproduce the results described in the paper:

> G. Quétant, J. A. Raine, M. Leigh, D. Sengupta and T. Golling  
> **_PIPPIN: Generating variable length full events from partons_**  
> [arXiv:2406.13074 [hep-ph]](https://arxiv.org/abs/2406.13074)

## Setup

Clone the repository and initialise the necessary submodule<sup>1</sup>:

```
git clone git@github.com:rodem-hep/pippin.git
git submodule update --init
```

Make sure to also install the required dependencies (See `requirements.txt`). For instance:
```
conda create -n pippin python=3.12
conda activate pippin
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

---

1. The [MLTools](https://gitlab.cern.ch/mleigh/mltools) submodule should be called `mattstools` and be pointing at the commit `1cc1d7a80351232804bd4fa79ffa46f21bfc9af2`.


## Training

Train and test the model with the following command:

```
python run.py
```

A debug config is available with less epochs and data:

```
python run.py +debug=debug
```

Note that data should be located at `./data`. See [Zenodo](https://zenodo.org/records/12117432) to download it.

## Evaluation

Evaluate the model with the following command:

```
python eval.py --name <network_name> --metrics --plots --inclusive
```

The arguments are:
- `--name, -n`: the network _name_
- `--metrics, -m`: compute the _metrics_
- `--plots, -p`: make the _plots_
- `--inclusive, -i`: consider the _inclusive_ dataset (i.e. PIPPIN)

Other arguments can be added:
- `--compare, -c`: make the _comparison_ plots (i.e. restrained PIPPIN vs retrained Turbo-Sim). Needs both `-t` and `-T`
- `--leading, -l`: consider the _leading_ particles (i.e. PIPPIN)
- `--turbolike, -t`: consider the _Turbo-like_ outputs (i.e. restrained PIPPIN)
- `--turbosim, -T`: consider the _Turbo-Sim_ outputs (i.e. retrained Turbo-Sim)

Note that several runs of the same model can be combined. See `eval.py` line `95` and change the list with the desired `job_id`. The runs outputs should be located at `./outputs/PIPPIN/ttbar/<network_name>/hdf5/test/<job_id>/outputs.h5`.
