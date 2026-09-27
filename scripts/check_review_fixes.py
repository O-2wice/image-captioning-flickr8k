"""Tiny CPU regression fixtures extracted from the notebook; no downloads/training."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
nb = json.loads((root / 'notebooks/image-captioning.ipynb').read_text(encoding='utf-8'))
tree = ast.parse(''.join(nb['cells'][37]['source']))
trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef))
methods = {node.name: node for node in trainer.body if isinstance(node, ast.FunctionDef)}
scope = {'copy': copy, 'torch': torch, 'plt': plt}
for name in ['checkpoint_metadata', 'check_checkpoint_compatibility', 'plot_training_history']:
    exec(compile(ast.Module(body=[methods[name]], type_ignores=[]), name, 'exec'), scope)

class Fixture:
    checkpoint_metadata = scope['checkpoint_metadata']
    check_checkpoint_compatibility = scope['check_checkpoint_compatibility']

fixture = Fixture()
fixture.vocab = SimpleNamespace(word2idx={'<pad>': 0, 'dog': 1, 'cat': 2})
fixture.run_config = {'embed_size': 8}
fixture.validation_decoding = 'greedy'
fixture.model = torch.nn.Linear(2, 2)
checkpoint = {'metadata': fixture.checkpoint_metadata()}
fixture.check_checkpoint_compatibility(checkpoint)
swapped = copy.deepcopy(checkpoint)
swapped['metadata']['word2idx'].update(dog=2, cat=1)
changed_config = copy.deepcopy(checkpoint)
changed_config['metadata']['config']['embed_size'] = 16
for bad in [{}, swapped, changed_config]:
    try:
        fixture.check_checkpoint_compatibility(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Incompatible checkpoint accepted')
print('PASS: compatible metadata accepted; missing metadata, swapped word IDs and changed config rejected.')

history = SimpleNamespace(train_loss_history=[3, 2], val_loss_history=[3, 2.5],
                          bleu_history={'bleu1': [.1, .2]}, learning_rate_history=[.001, .0009])
with patch.object(plt, 'show') as show:
    scope['plot_training_history'](history)
    assert show.call_count == 1
print('PASS: training history displays locally.')

# Use the actual plotting function with deterministic fake predictions. PAD is
# generated between words; attention maps must retain their original step IDs.
plot_tree = ast.parse(''.join(nb['cells'][46]['source']))
function = next(node for node in plot_tree.body if isinstance(node, ast.FunctionDef))
scope.update(np=np, transforms=transforms)
exec(compile(ast.Module(body=[function], type_ignores=[]), 'attention_plot', 'exec'), scope)

class Decoder:
    def __init__(self):
        self.step = 0
    def init_hidden_state(self, features):
        return torch.zeros(1, 2), torch.zeros(1, 2)
    def embedding(self, inputs):
        return torch.zeros(1, 1, 2)
    def attention(self, features, h):
        return torch.zeros(1, 2), torch.full((1, 1, 1, 4), float(self.step + 1))
    def lstm(self, inputs, state):
        return state
    def fc(self, h):
        token = [4, 0, 5, 2][self.step]  # dog, PAD, runs, EOS
        self.step += 1
        logits = torch.zeros(1, 6)
        logits[0, token] = 1
        return logits

model = SimpleNamespace(eval=lambda: None, encoder=lambda image: torch.zeros(1, 4, 2), decoder=Decoder())
vocab = SimpleNamespace(idx2word={0: '<pad>', 1: '<sos>', 2: '<eos>', 4: 'dog', 5: 'runs'})
# Deliberately shorter reference: generated words must not be cut to its length.
samples = [{'image': torch.zeros(3, 8, 8), 'caption_tokens': torch.tensor([1, 2])}]
with patch.object(plt, 'show'), patch.object(plt, 'savefig'):
    scope['plot_attention_for_samples'](samples, model, vocab, 'cpu', max_len=6)
axes = plt.gcf().axes
assert 'Generated: dog runs' in axes[0].get_title()
assert [ax.get_title() for ax in axes[1:]] == ["'dog'", "'runs'"]
assert [float(ax.images[1].get_array().mean()) for ax in axes[1:]] == [1., 3.]
assert model.decoder.step == 4
plt.close('all')
print('PASS: generated words retain matching maps across PAD and stop at EOS independently of reference length.')
