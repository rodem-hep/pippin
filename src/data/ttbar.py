import logging
from typing import Optional, Tuple, List
from pathlib import Path
import h5py

import lightning.pytorch as pl
import torch as T
from torch.utils.data import Dataset, DataLoader

from src.data.utils import structured_array_to_tensor
from src.data.utils import log_squash, inv_log_squash
from src.data.utils import pad_with_zeros, cat_with_options
from src.data.utils import matchability_to_presence
from src.physics import to_cartesian, to_polar, Part2RecoMatching

log = logging.getLogger(__name__)


class TopQuarksDataset(Dataset):
    """A collection of parton level and jet level objects from top quark pairs
    production
    """

    def __init__(
        self,
        path: Path,
        filename: str,
        decay_channel: str = "inclusive",
        n_min: int = 0,
        n_max: int | None = None,
        do_preprocessing: bool = True,
    ) -> None:
        """
        args:
            path: Absolute path to the dataset
            filename: Name of the HDF file
        kwargs:
            decay_channel: The decay channel to use
            n_min: The starting point in the HDF file to read in data
            n_max: The ending point in the HDF file to read in data
            do_preprocessing: Whether to pre-process the data
        """

        self.n_min = n_min
        self.n_max = n_max

        if decay_channel == "inclusive":
            with h5py.File(path/filename, "r") as f:
                self.part = self._get_part(f)
                self.reco = self._get_reco(f)
                self.chan = self._get_chan(f)
                self.pres = self._get_pres(f)
        else:
            raise ValueError(f"Unknown decay channel '{decay_channel}'")

        # Sanity check
        assert len(self.part) == len(self.reco)
        assert len(self.part) == len(self.chan)
        assert len(self.part) == len(self.pres)

        # Pre-process the data
        if do_preprocessing:
            self._preprocess()

    def __len__(self) -> int:
        return len(self.part)

    def __getitem__(self, idx: int) -> Tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
        """Returns partons, reco. objects, decay channel and presence mask"""
        part = self.part[idx]
        reco = self.reco[idx]
        chan = self.chan[idx]
        pres = self.pres[idx]
        mask_part = self._get_mask(part)
        mask_reco = self._get_mask(reco)
        return part, mask_part, reco, mask_reco, chan, pres

    def _get_mask(self, data: T.Tensor) -> T.Tensor:
        """Returns a mask for the zero padded data"""
        return data.sum(dim=-1) != 0

    def _preprocess(self):
        """
        Pre-process the data:
            - log-squash pT, mass and energy
        """

        # Log-squash pT, mass and energy
        self.part[..., 0] = log_squash(self.part[..., 0])
        self.part[..., 3] = log_squash(self.part[..., 3])
        self.reco[..., 0] = log_squash(self.reco[..., 0])
        self.reco[..., 3] = log_squash(self.reco[..., 3])

    def _get_part(self, f: h5py.File):
        """Returns the partons for the 'inclusive' decay channel"""

        # Get the data
        leptons = structured_array_to_tensor(
            f["delphes"]["truth_leptons"]["pt", "eta", "phi", "mass"][self.n_min:self.n_max]
        )
        neutrinos = structured_array_to_tensor(
            f["delphes"]["truth_neutrinos"]["pt", "eta", "phi", "mass"][self.n_min:self.n_max]
        )
        quarks = structured_array_to_tensor(
            f["delphes"]["truth_quarks"]["pt", "eta", "phi", "mass"][self.n_min:self.n_max]
        )

        # Get the decay channel to order leptons and neutrinos in data
        chan = T.tensor(f["delphes"]["decay_channel"][self.n_min:self.n_max], dtype=T.uint8)

        # Depending on the decay channel, fill the zeros padded quarks with
        # the corresponding leptons and neutrinos
        # Fully hadronic and b/s-quarks
        part = quarks

        # Semi leptonic (top)
        part[chan == 0b01, 1] = leptons[chan == 0b01, 0]
        part[chan == 0b01, 2] = neutrinos[chan == 0b01, 0]

        # Semi leptonic (anti-top)
        part[chan == 0b10, 4] = leptons[chan == 0b10, 0]
        part[chan == 0b10, 5] = neutrinos[chan == 0b10, 0]

        # Fully leptonic
        part[chan == 0b11, 1] = leptons[chan == 0b11, 0]
        part[chan == 0b11, 2] = neutrinos[chan == 0b11, 0]
        part[chan == 0b11, 4] = leptons[chan == 0b11, 1]
        part[chan == 0b11, 5] = neutrinos[chan == 0b11, 1]

        return part

    def _get_reco(self, f: h5py.File):
        """Returns the reconstructed objects for the 'inclusive' decay channel"""

        # Get the leptons, MET and jets
        leptons = structured_array_to_tensor(
            f["delphes"]["leptons"]["pt", "eta", "phi", "energy"][self.n_min:self.n_max]
        )
        met = structured_array_to_tensor(
            f["delphes"]["MET"]["MET", "phi"][self.n_min:self.n_max]
        )
        jets = structured_array_to_tensor(
            f["delphes"]["jets"]["pt", "eta", "phi", "energy"][self.n_min:self.n_max]
        )

        # Compute the total neutrino four-momentum (i.e. the true MET)
        neutrinos = structured_array_to_tensor(
            f["delphes"]["truth_neutrinos"]["pt", "eta", "phi", "mass"][self.n_min:self.n_max]
        )
        neutrinos = to_cartesian(neutrinos, has_mass=True)
        met_true = to_polar(neutrinos[:, 0] + neutrinos[:, 1])

        # Generate a fake eta when there is no neutrino in the event
        idx = met_true[..., 1] == 0
        met_true[idx, 1] = T.randn_like(met_true[idx, 1])

        # Use parton-level eta for MET eta and corresponding energy
        met_fake = T.empty(size=(len(met), 4))
        met_fake[..., 0] = met[..., 0]  # pT
        met_fake[..., 1] = met_true[..., 1]  # parton-level eta
        met_fake[..., 2] = met[..., 1]  # phi
        met_fake[..., 3] = met_fake[..., 0] * T.cosh(met_fake[..., 1])  # = |p| ~= energy (no mass)
        met_fake = met_fake.unsqueeze(1)

        # Concatenate the leptons, MET and jets
        reco = T.cat([leptons, met_fake, jets], dim=1)

        return reco

    def _get_chan(self, f: h5py.File):
        """Returns the decay channel for the 'inclusive' decay channel"""

        chan = T.tensor(f["delphes"]["decay_channel"][self.n_min:self.n_max], dtype=T.uint8)

        return chan

    def _get_pres(self, f: h5py.File):
        """Returns the partons presence in output for the 'inclusive' decay channel"""

        matchability = T.tensor(f["delphes"]["matchability"][self.n_min:self.n_max], dtype=T.uint8)
        chan = T.tensor(f["delphes"]["decay_channel"][self.n_min:self.n_max], dtype=T.uint8)

        truth_leptons = structured_array_to_tensor(
            f["delphes"]["truth_leptons"]["pt", "eta", "phi", "mass"][self.n_min:self.n_max]
        )
        reco_leptons = structured_array_to_tensor(
            f["delphes"]["leptons"]["pt", "eta", "phi", "energy"][self.n_min:self.n_max]
        )

        part2reco = Part2RecoMatching(
            truth_leptons,
            reco_leptons,
            mask_part=self._get_mask(truth_leptons),
            mask_reco=self._get_mask(reco_leptons),
            R=0.2,
        )
        is_matched = part2reco.is_matched

        pres = matchability_to_presence(matchability)
        pres[chan == 0b01, 1] = is_matched[chan == 0b01, 0]
        pres[chan == 0b10, 4] = is_matched[chan == 0b10, 0]
        pres[chan == 0b11, 1] = is_matched[chan == 0b11, 0]
        pres[chan == 0b11, 4] = is_matched[chan == 0b11, 1]

        return pres


class TopQuarksDataModule(pl.LightningDataModule):
    """
    A pytorch lightning datamodule to simplify the data preparation steps
    for the top quark pairs datasets

    Currently loads all datasets at the beginning so we have access to
    parameters such as dataset size. This may not work with distributed
    training!
    """

    def __init__(
        self,
        path: str = "/set/me/please/",
        decay_channel: str = "allhad",
        n_train: Optional[int] = None,
        n_valid: Optional[int] = None,
        n_test: Optional[int] = None,
        loader_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        kwargs:
            path: Location(s) of the stored data
            decay_channel: The decay channel to use
            n_train: The number of samples to load from the training set
            n_valid: The number of samples to load from the validation set
            n_test: The number of samples to load from the test set
            loader_kwargs: Dict for setting up the pytorch dataloaders
        """
        super().__init__()
        self.path = Path(path)
        self.decay_channel = decay_channel
        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test
        self.loader_kwargs = loader_kwargs or {}

    def setup(self, stage: Optional[str] = None) -> None:
        """
        This will be called manually as we need to know the size and shape
        of our dataset to setup the network and schedulers
        """

        if stage == "fit":
            self.train_ds = TopQuarksDataset(
                path=self.path,
                filename="ttbar_train.h5",
                decay_channel=self.decay_channel,
                n_max=self.n_train,
            )

            self.valid_ds = TopQuarksDataset(
                path=self.path,
                filename="ttbar_val.h5",
                decay_channel=self.decay_channel,
                n_max=self.n_valid,
            )

        elif stage == "test":
            self.test_ds = TopQuarksDataset(
                path=self.path,
                filename="ttbar_test.h5",
                decay_channel=self.decay_channel,
                n_max=self.n_test,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_ds, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, shuffle=False, **self.loader_kwargs)
