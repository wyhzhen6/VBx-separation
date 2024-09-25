#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on Libri2/3Mix datasets.
The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer-libri2mix.yaml
> python train.py hparams/sepformer-libri3mix.yaml


The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both libri2mix and
libri3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import torch
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from torch.cuda.amp import autocast
from hyperpyyaml import load_hyperpyyaml


# Define training procedure
class Separation(sb.Brain):

    def infer(self, wavpath):
        
        with torch.no_grad():
            mix = sb.dataio.dataio.read_audio(wavpath).reshape(1,-1)
            mix = mix.to(self.device)

            mix_w = self.hparams.Encoder(mix)
            est_mask = self.hparams.MaskNet(mix_w)

            mix_w = torch.stack([mix_w] * (
                    self.hparams.num_spks + 1)) if self.hparams.use_noise_ground and not self.hparams.use_noise_ground_only else torch.stack(
                [mix_w] * self.hparams.num_spks)

            sep_h = mix_w * est_mask

            # Decoding
            est_source = torch.cat(
                [
                    self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                    for i in range(
                    self.hparams.num_spks + 1 if self.hparams.use_noise_ground and not self.hparams.use_noise_ground_only else self.hparams.num_spks)
                ],
                dim=-1,
            )

            # T changed after conv1d in encoder, fix it here
            T_origin = mix.size(1)
            T_est = est_source.size(1)
            if T_origin > T_est:
                est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
            else:
                est_source = est_source[:, :T_origin, :]
        
            return est_source



def nass_separator(wavpath):

    # please download the pretrained model from the following link
    # https://github.com/TzuchengChang/NASS/releases/download/Pretrained_Model/results.zip
    # It is worth noting that the speechbrain version used by nass is different from that of pyannote
    hparams_file, run_opts, overrides = sb.parse_arguments(["VBx/pretrained/NASS/hyperparams.yaml", 
                                                   ])# "--device", "cpu"])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.utils.distributed.ddp_init_group(run_opts)


    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )


    # Infer
    return  separator.infer(wavpath)

