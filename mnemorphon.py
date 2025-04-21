#!/usr/bin/env python
# coding: utf-8
#
# Mnemorphon  v2.0
# 
# Module & class defining importable package for exemplar-based 
# modeling in Python from dynamic/variable speech signals (e.g. spectrograms)
# stdlib
from ast import literal_eval
from collections import defaultdict, Counter, namedtuple
from glob import glob
from itertools import permutations, chain
import json
import logging
from pathlib import Path
from random import sample
import sys
from time import time
# 3rd-party
import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging as DBA_mm
# from tslearn.barycenters import dtw_barycenter_averaging_subgradient as DBA_sg
from tslearn.metrics import dtw, dtw_path, dtw_path_from_metric
import torch
import torchaudio.functional as F
import torchaudio.transforms as T
from tqdm import tqdm
import yaml
import zeyrek
# local imports
from utils import orthographize, powerset


NO_CONF_ERR_MSG = ('!!! No config found; adjust conf path or '
                   'run utils.pre_process_metu_data() '
                   'before initializing Mnemorphon instance')
PREPROC_ERR_MSG = ('!!! No forms to meanings; '
                   'run utils.pre_process_metu_data() '
                   'or adjust config["forms2meanings_map"] '
                   'before initializing Mnemorphon instance')

#TODO:@phoneme consider whether this could/should be a DataClass
# notes from zeyrek code about singular/plural morph feats...
###
# Number-Person agreement.
# "FirstPersonSingular", "A1sg"
# "SecondPersonSingular", "A2sg"
# "ThirdPersonSingular", "A3sg"
# "FirstPersonPlural", "A1pl"
# "SecondPersonPlural", "A2pl"
# "ThirdPersonPlural", "A3pl"
###
# Possessive
## First person singular possession suffix.  "elma-m = my apple"
# p1sg = "FirstPersonSingularPossessive", "P1sg"
# p2sg = "SecondPersonSingularPossessive", "P2sg"
## Third person singular possession suffix. "elma-sı = his/her apple"
# p3sg = "ThirdPersonSingularPossessive", "P3sg"
## First person plural possession suffix.
# p1pl = "FirstPersonPluralPossessive", "P1pl"
# p2pl = "SecondPersonPluralPossessive", "P2pl"
# p3pl = "ThirdPersonPluralPossessive", "P3pl"
LexicalEntry = namedtuple('LexicalEntry', ['token_id', 'morph_feats', 'orth_form', 'lemma'])


class UnknownWordException(BaseException):
    pass


class Mnemorphon:
    """ Basic class for speech-based exemplar production model.
    
    TODO:@phoneme update/edit the below...

    Core structure is a mapping of FORMS (exemplar tokens) to MEANINGS:
    - Forms are variable-length acoustic representations (Mel spectrograms)
      - in lexicon these are *paths* to on-disk spectrograms
    - Meanings are lemmas (representing broad lexical semantics) and morphosyntactic features
    """
    def __init__(self, config_path: str = './config.yaml'):
        try:
            with open(config_path) as f_conf:
                self.config = yaml.load(f_conf, yaml.Loader)
                self.verbose = self.config['verbose']
                self.mspec_path = self.config['mspec_path']
                if self.verbose:
                    logging.info(json.dumps(self.config, indent=2))
        except FileNotFoundError:
            logging.error(NO_CONF_ERR_MSG)
            sys.exit(1)
        self.morph_analyzer = zeyrek.MorphAnalyzer()

        # load lexicon
        try:
            with open(self.config['forms2meanings_map'], encoding='utf8') as f_map:
                self.forms2meanings = json.load(f_map)
        except FileNotFoundError as e:
            logging.error(PREPROC_ERR_MSG)
            sys.exit(1)
        if self.config['verbose']:
            logging.info('Mnemorphon --> loading lexicon')
        self.lemmas2tokens = defaultdict(set)   # this is like a lexeme?
        self.morphs2tokens = defaultdict(set)
        self.lexicon = []
        for tok_idx in tqdm(self.forms2meanings,
                            disable=False if self.verbose else True):
            for tok_analysis in self.forms2meanings[tok_idx]:
                tok, morphs, orth, lemma = tok_analysis
                self.lexicon.append(LexicalEntry(tok, morphs, orth, lemma))
                self.lemmas2tokens[lemma].add(tok)
                for morphset in powerset(morphs):
                    self.morphs2tokens[tuple(sorted(morphset))].add(tok)
                # N.B. skipping index over orthographic form for now
        logging.info('%d entries in self.lexicon', len(self.lexicon))
        # disable until we're interested in looking at word-level
        # abstraction
        #self.word_freqs = Counter(dict([(x, len(self.phonforms2tokens[x])) for
        #                                x in self.phonforms2tokens]))

    def precompute_intracloud_distances(self, phonform: str, save: bool = False,
                                        numpy: bool = True):
        toks = self.get_tokens(phonform)
        tok_dists = {tok: {} for tok in toks}
        cloud = self.get_cloud(phonform, numpy)
        # N.B. the dtw() computation is WAY slower with torch.Tensor vs ndarray
        for (t0, sp0), (t1, sp1) in tqdm(permutations(zip(toks, cloud), 2),
                                         total=(len(toks)**2 - len(toks))):
            d = dtw(sp0, sp1)
            # call .item() to get dist as float if working with torch.Tensor
            tok_dists[t0][t1] = d if numpy else d.item()

        if save:
            with open(f'{phonform}_dists.json', 'w', encoding='utf8') as f_out:
                f_out.write(json.dumps(tok_dists, indent=2) + '\n')
        return tok_dists

    def make_weights(self, dists, power=1, normed=True):
        assert np.greater(dists, 0.0).all(), 'Input dists must all be > 0.0'
        if power in (1, 2):
            wts = 1.0/(np.power(dists, power))
        elif power=='exp':
            wts = np.exp(-1.0*np.array(dists))
        return wts/np.sum(wts) if normed else wts

    def compute_DBA(self, seed_tok: str, cloud, weights=None, verbose=False):
        seed = self.get_token_spec(seed_tok)
        dba_mm = DBA_mm(cloud,
                        init_barycenter=seed,
                        max_iter=50,
                        tol=1e-05,
                        weights=weights,
                        metric_params=None,
                        verbose=verbose)
        return dba_mm

    def get_tokens(self, lemma, morph_feats, strict, context: dict = {}):
        #TODO(@phoneme) at some future date
        # if 'gender' in context:
        #     related_toks |= {le for le in self.lexicon if le.gender==context['gender']}
        #TODO(@phoneme) hacky approaches to leveraging context
        # - check token name directly for m/f
        # - sample according to most recently sampled tokens ("activation")
        # - maintain dict (palimpsest?) of IDs2tokens
        # - (stretch) maintain a dict of letters2tokens
        # - (stretch) maintain dict of IDs2meanf0
        # - (stretch) mapping by acoustic distance to some threshold
        lemma_related_toks = self.lemmas2tokens[lemma]
        morph_related_toks = self.morphs2tokens[morph_feats]
        if strict:
            # known word: take only tokens with same lemma and same morph_feats
            related_toks = lemma_related_toks & morph_related_toks
        else:
            # take everything with same lemma and subset of morph_feats (upweight these!)
            submorph_toks = set()
            for morphset in sorted(powerset(morph_feats), key=lambda x:len(x), reverse=True):
                if morphset:
                    # skip empty subset
                    submorph_toks |= lemma_related_toks & self.morphs2tokens[morphset]
            # AND everything with same morph_feats
            related_toks = submorph_toks | morph_related_toks
        return sorted(related_toks)

    def get_token_spec(self, token: str, numpy=False):
        # squeeze() to get rid of batch dimension, transpose for tslearn compatibility
        mspec = torch.load(f'{self.mspec_path}/{token}.pt').squeeze().t()
        return mspec.numpy() if numpy else mspec
    
    def lex_ent_is_known(self, lemma, morph_feats):
        """ Do I have tokens matching the lemma and exact morphosyntactic features? """
        return (self.lemmas2tokens[lemma] & self.morphs2tokens[morph_feats] != set())

    def produce(self, lemma: str, morph_feats: tuple, context: dict = {}, N: int = 0, numpy: bool = True):
        """ Produce output spectrogram for given input lexical item.

        Given a lexical entry i.e. (lemma, case, pl) tuple, an output form is generated by:

        1. if the word is known (i.e. have token stored with same lemma & morph)
          a. sample from the stored forms associated with the meaning for the seed exemplar
          b. output as the DBA of the cloud, with the seed as initial "guess".
        2. if the word is _not_ known BUT associated lemma is known[*]:
          a. find all previously known words with same lemma and subset of morph tags
          b. find all previously known words with same morph tags
          c. output DBA across everything
        [*] if lemma is not known, there is nothing to be done

        N.B. Applied naively, this algorithm produces noisy outputs,
        but our SCiL and SMPh experiments suggest that being smart about cloud
        construction/composition, along with careful weighting, can mitigate that.
        """
        if lemma not in self.lemmas2tokens:
            raise UnknownWordException(f'Unknown lemma: {lemma}.')

        # build (possibly downsampled) cloud of token IDs and melspecs
        known_lex_ent = self.lex_ent_is_known(lemma, morph_feats)
        tokens = self.get_tokens(lemma, morph_feats, strict=known_lex_ent, context=context)
        tokens = sample(tokens, N) if N > 0 else tokens
        cloud = [self.get_token_spec(tok, numpy) for tok in tokens]
        # for unknown word, this could sample from maximal lemma+morphs match
        seed_token = sample(tokens, 1)[0]
        dba_mm = self.compute_DBA(seed_token, cloud)
        return dba_mm

    def perceive(self, form, meaning=None):
        # A form (and optional meaning) is perceived as follows:
        # Given a form (and optional meaning)...
        # 1a. if a known meaning is given, append the form to the collection of forms
        #    already associated with that meaning
        # 1b. if an unknown meaning is given, create a new meaning category and initialize
        #    it with the given form
        # 2. if no meaning is given, find the meaning whose forms are closest to the
        #    given one and associate the given form with that meaning
        pass
