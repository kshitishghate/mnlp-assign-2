#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple
import sys
sys.path.append('/content/fairseq/examples/')

import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
)
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm


log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

SPLITS = ["train", "dev", "test"]

## DOC: Used torchaudio to load all wav files and get sampling rate
## Created dataset class for CommonVoice
def load_df_from_tsv(path) -> pd.DataFrame:
    _path = path if isinstance(path, str) else path.as_posix()
    return pd.read_csv(
        _path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )

class CommonVoice(Dataset):
    

    def __init__(
        self,
        root: str,
        split: str
    ) -> None:
        
        self.root: Path = Path(root)

        split_path = f"{self.root}/{split}_raw.tsv"
        df = load_df_from_tsv(split_path)
            
        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        for e in data:
            try:
                path = self.root / "clips" / e["path"]
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError:
                pass

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, sentence, translation, speaker_id,
            sample_id)``
        """
        data = self.data[n]
        path = self.root / "clips" / data["path"]
        waveform, sample_rate = torchaudio.load(path)
        sentence = data["sentence"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        return waveform, sample_rate, sentence, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)

def process(args):
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)
    # Extract features
    feature_root = out_root / "fbank80"
    feature_root.mkdir(exist_ok=True)
    for split in SPLITS:
        print(f"Fetching split {split}...")
        dataset = CommonVoice(out_root.as_posix(), split=split)
        print("Extracting log mel filter bank features...")
        for wav, sample_rate, sent, spk_id, utt_id in tqdm(dataset):
            sample_id = f"{spk_id}-{utt_id}"
            extract_fbank_features(
                wav, sample_rate, feature_root / f"{sample_id}.npy"
            )
    # Pack features into ZIP
    zip_path = out_root / "fbank80.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = CommonVoice(out_root.as_posix(), split=split)
        for wav, sample_rate, sent, spk_id, utt_id in tqdm(dataset):
            sample_id = f"{spk_id}-{utt_id}"
            manifest["id"].append(sample_id)
            manifest["audio"].append(audio_paths[sample_id])
            manifest["n_frames"].append(audio_lengths[sample_id])
            manifest["tgt_text"].append(sent.lower())
            manifest["speaker"].append(spk_id)
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), out_root / f"{split}.tsv"
        )
        if split.startswith("train"):
            train_text.extend(manifest["tgt_text"])
    # Generate vocab
    print("---Debugging---")
    print(len(train_text))
    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            out_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
        )
    # Generate config YAML
    gen_config_yaml(
        out_root,
        spm_filename=spm_filename_prefix + ".model",
        specaugment_policy="ld"
    )
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=10000, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
