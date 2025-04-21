#!/usr/bin/env python
# coding: utf-8
""" Pre-processing steps for speech corpora.

This module includes functions that are helpful for pre-processing the
Middle East Technical University Turkish Microphone Speech v1.0
(https://catalog.ldc.upenn.edu/LDC2006S33) which was the data source
for our 2024 experiments.

There is a function to segment speech files according to provided
alignments, and a function to compute spectrograms. (the latter
uses parameters specific to the TTS code that we used previously
and will likely change or go away)

TODO: update this to process CommonVoice (and potentially other
speech corpora.
"""
# stdlib imports
from collections import defaultdict, Counter
from glob import glob
from itertools import product
import json
import logging
import os
from pathlib import Path
from random import sample
import sys
from time import time, sleep
# 3rd-party imports
import ipdb
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import zeyrek
# shut zeyrek's logger up
import logging.config
logging.config.dictConfig({
    'version': 1,
    # Other configs ...
    'disable_existing_loggers': True
})
# local imports
from melspec import mel_spectrogram


logger = logging.getLogger(__name__)
#FIXME @fmailhot: These constants are for the LDC METU
# corpus and assume METUbet transcription
VOWELS = {'A', 'AA', 'E', 'EE', 'I', 'IY', 'O', 'OE', 'U', 'UE'}
# !!! METU corpus caveat !!!
# The following speakers have a mismatched set of speech and transcript files:
# s1004, s1023, s1043, s1044, s1045, s1049,
# s1051, s1054, s1062, s1071, s1073, s1076,
# s1080, s1082, s1089, s1101, s1114, s1120,
# s1128, s1132, s1134, s1135, s1141
# In our 2024 experiments we eliminated these speakers from our corpus,
# shifting the (binarized) gender balance from (m=60, f=60) to (m=49,f=48)
OMITTED_METU_SPKRS = {'s1004', 's1023', 's1043', 's1044', 's1045', 's1049',
                      's1051', 's1054', 's1062', 's1071', 's1073', 's1076',
                      's1080', 's1082', 's1089', 's1101', 's1114', 's1120',
                      's1128', 's1132', 's1134', 's1135', 's1141'}
# METUbet metadata files are latin-5 encoded
TK_ENC = 'iso-8859-9'
# make these part of init using config file?
DATA_ROOT_DIR = './turkish/data'
SEGMENTED_WRD_WAVS_DIR = './wrd_wavs'
WRD_MSPECS_DIR = './wrd_mspecs'
# gonna use zeyrek to create lexeme-like mapping
analyzer = zeyrek.MorphAnalyzer()


def orthographize(wd):
    TK_ORTHO_MAP = {'C': 'ç', 'S': 'ş', 'I': 'ı', 'O': 'ö', 'U': 'ü', 'G': 'ğ'}
    phon_orth_map = str.maketrans(TK_ORTHO_MAP)
    return wd.translate(phon_orth_map)


def phonify(wd):
    TK_PHONE_MAP = {'ç':'C', 'ş':'S', 'ı':'I', 'ö':'O', 'ü':'U', 'ğ':'G'}
    orth2phon = str.maketrans(TK_PHONE_MAP)
    return wd.translate(orth2phon)


def is_nominal(lex_ent):
    return ('Noun' in lex_ent.morph_feats and
            set(lex_ent.morph_feats) & {'Adj', 'Verb'} == set())


def is_plural(lex_ent):
    zeyrek_plurals = {'A1pl', 'A2pl', 'A3pl',
                      'P1pl', 'P2pl', 'P3pl'}
    return set(lex_ent.morph_feats) & zeyrek_plurals != set()


def is_possessive(lex_ent):
    zeyrek_possessives = {'P1sg', 'P2sg', 'P3sg',
                      'P1pl', 'P2pl', 'P3pl'}
    return set(lex_ent.morph_feats) & zeyrek_possessives != set()


def is_genitive(lex_ent):
    return 'Gen' in lex_ent.morph_feats


def is_probably_nominative(lex_ent):
    zeyrek_non_nominative = {'Acc', 'Dat', 'Loc',
                             'Ins', 'Abl', 'Gen'}
    # surely there's a better way to do this?
    return set(lex_ent.morph_feats) & zeyrek_non_nominative == set()


def is_proper_noun(lex_ent):
    # surely there's a better way to do this?
    return lex_ent.orth_form.istitle()


# Mnemorphon uses this to create morphs2toks mapping
# code from: https://stackoverflow.com/a/64320524
def powerset(iterable):
    for sl in product(*[[[], [i]] for i in iterable]):
        yield {j for i in sl for j in i}


#TODO @fmailhot: set this up to handle CommonVoice corpus structure
def segment_wavs(align_dir: str, wav_dir: str, wavs_dir: str,
                 align_typ: str = 'wrd', verbose: bool = False) -> None:
    """ Split speech files into words or phones, given alignments.

    align_dir: path to top-level directory of aligments
        must contain speaker subdirectories containing
        .wrd/.phon files with one alignment per line
    wav_dir: path to top-level directory of speech files
        must contain speaker subdirectories (with same IDs
        as align_dir) containing .wav files with parallel
        naming to the alignment files (spkrID-uttID)
    wavs_dir: path to output directory; upon completion, will
        contain wav files named like
        {word/phone}_{tok-cnt}_{spkr-ID}_{gender-marker}
        N.B.1 this will be created if it does not exist
        N.B.2 if this exists its contents will be clobbered!!
    align_typ: str in ('wrd', 'phn') indicating whether to segment
        contents of wav_dir into words or phones

    - uses speaker metadata for file naming
    - creates token <---> word/vowel mapping files for easy re-use
    
    **N.B.** This code makes strong assumptions about data organization.
    In particular it assumes your speech and alignment files are in parallel
    directories each containing speaker-specific subdirectories containing
    WAVs and metadata files, and word-level and phone-level alignments, i.e.:

    - speech-files/
        - speaker_ID1/
            - spkr_ID1.txt         # metadata; see METU corpus for details
            - spkr_ID1-utt_ID.wav
            - spkr_ID1-utt_ID.txt  # METUbet transcript
        - speaker_ID2/
        - ...
    - alignment-files/
        - speaker_ID1/
            - spkr_ID1-utt_ID.wrd
            - spkr_ID1-utt_ID.phn
        - speaker_ID2/
            - ...
    """
    assert align_typ in ('wrd', 'phn'), f'align_typ {align_typ} not supported'
    os.makedirs(wavs_dir, exist_ok=True)
    tick = time()
    vocab = Counter()
    tokens = set()
    align_dir = os.path.abspath(align_dir)
    wav_dir = os.path.abspath(wav_dir)
    spkr_dirs = sorted([os.path.basename(d) for d in glob(f'{align_dir}/*')])
    for spkr_dir in tqdm(spkr_dirs):
        # see above
        if spkr_dir in OMITTED_METU_SPKRS:
            continue
        # get metadata
        with open(f'{wav_dir}/{spkr_dir}/{spkr_dir}.txt', encoding=TK_ENC) as f_spkr:
            for line in f_spkr:
                if line.startswith('GENDER'):
                    g_marker = line.strip().split()[-1].lower()
                    assert g_marker in ('m', 'f'), f'Invalid gender marker in {f_spkr}'
                    break
        align_files = sorted(glob(f'{align_dir}/{spkr_dir}/*.{align_typ}'))
        wav_files = sorted(glob(f'{wav_dir}/{spkr_dir}/*.wav'))
        for align_file, wav_file in zip(align_files, wav_files):
            with open(align_file, 'r', encoding='utf8') as f_align, \
                    open(wav_file, 'rb') as f_wav:
                # N.B. y is a 2d Tensor here, shape == (1, n_samples)
                y, sr = torchaudio.load(f_wav, normalize=False)
                words_seen = defaultdict(int)
                spkr_sentnum, _ = os.path.splitext(os.path.basename(align_file))
                for line in f_align:
                    start, end, phon_form = line.strip().split()
                    if phon_form == 'SIL':
                        continue
                    if align_typ == 'phn' and phon_form not in VOWELS:
                        continue
                    vocab.update([phon_form])
                    words_seen[phon_form] += 1
                    tok_name = f'{phon_form}_{words_seen[phon_form]}_{spkr_sentnum}_{g_marker}'
                    # tokens2phonforms[tok_name] = phon_form
                    tokens.add(tok_name)
                    wav_name = f'{wavs_dir}/{tok_name}.wav'
                    with open(wav_name, 'wb') as f_out:
                        torchaudio.save(f_out,
                                        src=y[:,int(start):int(end)],
                                        sample_rate=sr,
                                        format='wav',
                                        encoding="PCM_S",
                                        bits_per_sample=16)
    with open(f'{align_typ}_freqs.json', 'w') as f_out:
        f_out.write(json.dumps(dict(vocab.most_common()), indent=2))
        logger.info('vocab counts written to %s', f'{align_typ}_freqs.json')

    tock = time()
    logger.info('segmented %d tokens in %d seconds', len(tokens), int(tock-tick))


# pre-compute token mel spectrograms and save token-to-lemma mapping for later
def compute_token_spectrograms(wavs_dir: str = None, spec_dir: str = None,
                               verbose: bool = False) -> None:
    """ Compute Mel spectrograms for all wave files in user spec'd dir."""
    os.makedirs(spec_dir, exist_ok=True)
    wav_files = sorted(glob(f'{wavs_dir}/*.wav'))
    skipped = []
    succeeded = []
    form2meanings = defaultdict(list)
    tick = time()
    for f_wav in tqdm(wav_files, total=len(wav_files)):
        try:
            # ipdb.set_trace()
            # load audio
            y, _ = torchaudio.load(f_wav, normalize=True)
            # compute & save spectrogram
            y_melspec = mel_spectrogram(y)
            tok_name, _ = os.path.splitext(os.path.basename(f_wav))
            phon_form = tok_name.split('_')[0]
            orth_form = orthographize(phon_form)
            analyses = analyzer.analyze(orth_form)[0]
            spec_name = f'{spec_dir}/{tok_name}.pt'
            for analysis in analyses:
                # skip proper nouns and deficient analyses
                # N.B. this is a hacky/bad way to check for names
                if analysis.word.isupper() or len(analysis.morphemes) == 1:
                    continue
                # ipdb.set_trace()
                form2meanings[tok_name].append([f'{spec_name}', # path to actual token
                                                analysis.morphemes,
                                                analysis.word,
                                                analysis.lemma])
            torch.save(y_melspec, spec_name)
        except Exception as e:
            ipdb.set_trace()
            skipped.append(f'{wavs_dir}/{f_wav}')
            continue
        succeeded.append(f'{wavs_dir}/{f_wav}')
    assert len(succeeded + skipped) == len(wav_files), f'succeeded: {len(succeeded)}\nskipped:{(skipped)}\ntotal:{len(wav_files)}'
    tock = time()
    logger.info('%d melspecs computed (%d skipped) in %d seconds',
                    len(succeeded), len(skipped), int(tock-tick))

    with open('forms2meanings.json', 'w', encoding='utf8') as f_tokmap:
        f_tokmap.write(json.dumps(form2meanings, indent=2))
    logger.info('%d forms:meanings written to forms2meanings.json', len(form2meanings))


def preprocess_metu_data(data_root_dir: str = DATA_ROOT_DIR,
                         segmented_wavs_dir: str = SEGMENTED_WRD_WAVS_DIR,
                         mspec_dir: str = WRD_MSPECS_DIR,
                         align_typ: str = 'wrd', verbose: bool = True) -> None:
    assert align_typ in ('wrd', 'phn'), 'align_typ must be one of {"wrd", "phn"}'
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    wavs_dir = Path(data_root_dir) / 'speech-text'
    align_dir = Path(data_root_dir) / 'alignments'
    logger.info('running: segment_wavs(%s, %s, %s, %s, %s)',
                 align_dir, wavs_dir, segmented_wavs_dir, align_typ, str(verbose))
    sleep(1)
    segment_wavs(align_dir, wavs_dir, segmented_wavs_dir, align_typ, verbose)
    logger.info('running: compute_token_spectrograms(%s, %s)',
                 segmented_wavs_dir, mspec_dir)
    sleep(1)
    compute_token_spectrograms(segmented_wavs_dir, mspec_dir)
    with open('./config.yml', 'w', encoding='utf8') as f_conf:
        f_conf.write(f'mspec_path: {mspec_dir}\n')
        f_conf.write(f'wavs_path: {segmented_wavs_dir}\n')
        f_conf.write(f'forms2meanings_map: forms2meanings.json\n')
        f_conf.write(f'verbose: {verbose}\n')
    logger.info('Done pre-processing METU data; see config.yml')


if __name__ == '__main__':
    try:
        (data_root_dir,
         segmented_wavs_dir,
         mspec_dir,
         align_typ) = sys.argv[1:]
        preprocess_metu_data(data_root_dir, segmented_wavs_dir, mspec_dir, align_typ)
    except (ValueError, IndexError) as e:
        sys.exit(f'\nUsage: {sys.argv[0]} data_root_dir out_wavs_dir, out_specs_dir, [wrd|phn]')
