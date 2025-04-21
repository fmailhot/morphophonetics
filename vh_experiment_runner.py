#!/usr/bin/env python
# coding: utf-8
#
# Mnemorphon v1.0
# 
# Each acoustic *form* (mel-spectrogram) is stored along with a *meaning*, which
# comprises a few pieces of information:
# - "lexical" category (*viz.* a string representation of the lemma)
# - "case" (a string representation from `{NOM, GEN}`)
# - "plurality" (or lack thereof, a string from `{SG, PL}`)
# 
# So e.g. given a form like `erkeklerin`, a GEN-PL form (meaning "men's" or "of men"),
# we would have a full spectrogram associated with the following set of tags or metadata:
# {ERKEK, GEN, PL}
# 
# ### Experiment 2: Generalization of harmony in production of novel forms
# stdlib
from collections import defaultdict, namedtuple
from copy import deepcopy
from glob import glob
from itertools import permutations, chain
import logging
from pathlib import Path
from random import sample
import sys
from time import sleep
# 3rd-party
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tslearn.barycenters import dtw_barycenter_averaging as DBA
from tslearn.metrics import dtw
import torch
import torchaudio
import zeyrek
# local imports
from mnemorphon import Mnemorphon, LexicalEntry
from utils import (is_genitive,
                   is_nominal,
                   is_plural,
                   is_possessive,
                   is_probably_nominative,
                   is_proper_noun)


logging.basicConfig(level=logging.INFO, stream=sys.stdout)

logging.info('='*59)
logging.info('= MNEMORPHON EXPERIMENT 2 (INFL-INDUCTION) RUNNER SCRIPT =')
logging.info('='*59 + '\n')
sleep(1)

logging.info('--> INITIALIZE BASE MODEL')
mnem_model = Mnemorphon(config_path='config.yml')
sleep(1)

logging.info('--> PREPARE DATA FOR EXP 2')
sleep(1)

logging.info('--> Find NOM/GEN SG/PL (non-possessive, non-proper, noun) TARGETS in lexicon')

target_words = {
    'NOM_SG': [],
    'NOM_PL': [],
    'GEN_SG': [],
    'GEN_PL': []
}

for lex_ent in tqdm(mnem_model.lexicon):
    # just non-possessive non-proper nouns for now
    if is_possessive(lex_ent) or is_proper_noun(lex_ent):
        continue
    if is_nominal(lex_ent):
        # we're looking for nominative and genitive plurals
        if is_plural(lex_ent):
            if is_genitive(lex_ent):
                target_words['GEN_PL'].append(lex_ent)
            else:
                if is_probably_nominative(lex_ent):
                    target_words['NOM_PL'].append(lex_ent)
        else:
            if is_genitive(lex_ent):
                target_words['GEN_SG'].append(lex_ent)
            else:
                if is_probably_nominative(lex_ent):
                    target_words['NOM_SG'].append(lex_ent)

def dedupe_lex_ents(lex_ent_list):
    tmp_lex_ent_list = []
    for parse in lex_ent_list:
        if lex_ent not in tmp_lex_ent_list:
            tmp_lex_ent_list.append(parse)
    return tmp_lex_ent_list

logging.info('--> DEDUPE & WRITE TARGET PARSES')
for morph_tag in target_words:
    target_words[morph_tag] = dedupe_lex_ents(target_words[morph_tag])
    logging.info(f'--> {len(target_words[morph_tag])} {morph_tag} wordforms extracted')

# with open('extracted_targets.txt', 'w', encoding='utf8') as f_out:
#     for morph_tag in target_words:
#         for word in target_words[morph_tag]:
#             f_out.write(f'{word}\n')
# sleep(1)


logging.info('--> TEST PLURAL GENERATION')
### Hacky test of plural generalization
case = sample(['NOM', 'GEN'], 1)[0]
# number = sample(['SG', 'PL'], 1)[0]
number = 'SG'
# make a copy so I'm not messing directly with the model's lexicon
samp_candidate = sample(target_words[f'{case}_{number}'], 1)[0]
logging.info(f'--> TEST 1: sampled {str(samp_candidate)} from target_words[{case}_{number}]')
test_candidate = deepcopy(samp_candidate)
for idx, feat in enumerate(test_candidate.morph_feats):
    if feat.endswith('sg'):
        test_candidate.morph_feats[idx] = test_candidate.morph_feats[idx].replace('sg', 'pl')
logging.info(f'--> TEST: Producing output for new candidate {test_candidate.lemma, str(test_candidate.morph_feats)}')

ipdb.set_trace()

lex_ent_spec = mnem_model.produce(test_candidate.lemma, test_candidate.morph_feats)


word = sample(target_words[f'NOM_SG'], 1)[0]
logging.info(f'--> TEST 2: sampled {word} from target_words[SG]')
tweaked_entry = LexicalEntry(word,
                             sample_analysis.morphemes+['A3pl'],
                             sample_analysis.word,
                             sample_analysis.lemma)
test_out = mnem_model.produce(tweaked_entry)
logging.info(test_out.shape)
sys.exit()

### Hacky test of genitive generalization
wd_nom_sg = sample(target_words['NOM_SG'], 1)[0]
lexeme = utils.orthographize(wd_nom_sg)
print(f'SG wd sample: {lexeme}')
meaning = (lexeme, 'GEN', 'SG')
foo = mnem_model.produce(meaning)
print(foo.shape)

