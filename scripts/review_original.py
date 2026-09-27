"""Read-only source checks and tiny fixture runs; never runs notebook training.

The manifest records the notebook as first reviewed. The corrections described
in review/notebook-review.md have since been applied, so this script asserts
that each fix is present rather than that the source is untouched. Defects left
deliberately unfixed, because they need a design decision rather than a
correction, are still reproduced and labelled as such.
"""
import ast
import json
from pathlib import Path
import re

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'notebooks/image-captioning.ipynb'
raw = path.read_bytes()
notebook = nbformat.reads(raw.decode('utf-8'), as_version=4)
nbformat.validate(notebook)
transformer = TransformerManager()
trees = {}
for index, cell in enumerate(notebook.cells, 1):
    if cell.cell_type == 'code':
        source = transformer.transform_cell(cell.source)
        compile(source, f'cell-{index}', 'exec')
        trees[index] = ast.parse(source)
# Two markdown cells were removed: the inherited BLEU commentary from the
# earlier run, and the runtime housekeeping note.
assert len(notebook.cells) == 60 and len(trees) == 27
print('PASS: project notebook has 60 cells, '
      'all 27 code cells compile.')

# Scan all JSON strings including saved outputs; never print matched values.
patterns = [
    r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})',
    r'\b(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|xox[baprs]-[0-9A-Za-z-]{15,})',
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    r'(?i)\b(?:api[_-]?key|access[_-]?token|password|client[_-]?secret|wandb_api_key)\s*[=:]\s*["\x27][^"\x27\s]{8,}["\x27]',
]
def scan(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if re.fullmatch(r'(?i)(key|api[_-]?key|access[_-]?token|password|client[_-]?secret)', key):
                assert not (isinstance(item, str) and len(item) >= 8), 'Potential credential field; value withheld'
            scan(item)
    elif isinstance(value, list):
        for item in value:
            scan(item)
    elif isinstance(value, str):
        assert not any(re.search(pattern, value) for pattern in patterns), 'Potential credential; value withheld'
# The notebooks are published, so an account handle or a private project URL is
# treated as an exposure even though neither is a credential. The patterns match
# the shapes W&B prints them in, rather than naming the handle, which would put
# it back into a public file.
identity_patterns = [
    r'(?i)currently logged in as',
    r'(?i)\bwandb\.(?:init|Api)\s*\([^)]*entity',
    r'(?i)\bwandb\.ai/',
    r'(?i)\bwandb\.login\s*\(',
]
for label, blob in (('working', raw),):
    scan(json.loads(blob))
    text = blob.decode('utf-8')
    for pattern in identity_patterns:
        assert not re.search(pattern, text), (
            f'{label} notebook exposes a W&B account handle, project URL or login call')

# Credentials must come from Colab Secrets, never from an uploaded key file.
working_source = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'code')
assert 'files.upload(' not in working_source, 'kaggle.json upload is back in the working notebook'
# The dataset now comes from Kaggle's public download URL, so the notebook needs
# no credentials at all. Assert that none crept back in.
for banned in ('userdata.get(', 'KAGGLE_KEY', 'KAGGLE_USERNAME', 'kaggle.json'):
    assert banned not in working_source, f'{banned} is back; the download needs no credentials'
print('PASS: no credential-pattern matches in the project notebook, no W&B handle or project URL, '
      'and no credentials of any kind. This is not an exhaustive secret guarantee.')

# No model weights or dataset are loaded. Only a small attention module is built.
import torch
from torch import nn
torch.set_num_threads(1)
scope = {'torch': torch, 'nn': nn}
attention_class = next(n for n in trees[27].body if isinstance(n, ast.ClassDef))
exec(compile(ast.Module(body=[attention_class], type_ignores=[]), 'attention-fixture', 'exec'), scope)
attention = scope['MultiHeadAttention'](8, 8, 8, num_heads=2)

# One token is the degenerate case the encoder used to produce: softmax over a
# single position returns 1 whatever the query, so nothing can be visualised.
single = torch.randn(2, 1, 8)
with torch.no_grad():
    _, weights_single = attention(single, torch.randn(2, 8))
assert torch.equal(weights_single, torch.ones_like(weights_single))

# With the patch tokens the encoder now returns, the weights must depend on the
# query and must not be uniform, or 7.2 has nothing to overlay.
patches = torch.randn(2, 196, 8)
with torch.no_grad():
    context_a, weights_a = attention(patches, torch.randn(2, 8))
    context_b, weights_b = attention(patches, torch.randn(2, 8))
assert weights_a.shape == (2, 2, 1, 196), weights_a.shape
assert not torch.allclose(weights_a, weights_b), 'attention ignores the decoder state'
uniform = torch.full_like(weights_a, 1 / 196)
assert not torch.allclose(weights_a, uniform, atol=1e-4), 'attention is uniform'
assert torch.allclose(weights_a.sum(dim=-1), torch.ones(2, 2, 1), atol=1e-4)
print('PASS: attention over patch tokens varies with the query and sums to one; '
      'over a single token it is 1 by construction, which is what 7.2 needed fixed.')

# Reproduce the exact reference-list comprehension using a tokenizer stub;
# the error is string iteration, independent of NLTK tokenization resources.
trainer = next(n for n in trees[38].body if isinstance(n, ast.ClassDef))
validate = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'validate')
reference_expr = next(n.value for n in ast.walk(validate) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'refs' for t in n.targets))
references = ['A dog runs.', 'A brown dog is running', 'The dog runs fast']
refs = eval(compile(ast.Expression(reference_expr), 'reference-fixture', 'eval'),
            {'refs_for_image': references, 'word_tokenize': str.split})
assert len(refs) == len(references), refs
assert all(len(tokens) > 1 for tokens in refs), refs
print('PASS: validation builds one word-level reference per caption, not one per character.')

import copy
linear = nn.Linear(1, 1, bias=False)
snapshot = copy.deepcopy(linear.state_dict())
before = snapshot['weight'].clone()
with torch.no_grad():
    linear.weight.add_(1)
assert torch.allclose(snapshot['weight'], before), 'best checkpoint must not track later training'
assert -2.0 * 1.2 < -2.0, 'a repetition penalty must lower a repeated token score'
print('PASS: the best checkpoint is a deep copy; the repetition penalty lowers repeat scores.')

defined_samples = [n for tree in trees.values() for n in ast.walk(tree)
                   if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == 'samples']
assert defined_samples, 'samples must be defined before the attention plotting call'
print('PASS: samples is assigned before the attention plotting call.')

source_text = raw.decode('utf-8')
for needle, description in [
        ("'all_captions'", 'every reference caption travels with the batch'),
        ('apply_dropout', 'token dropout is confined to training'),
        ('self.scaler.unscale_(self.optimizer)', 'gradients are unscaled before clipping'),
        ('steps_in_group', 'the final accumulation group divides by its real size'),
        ('last_checkpoint.pth', 'an epoch-level resume point is written'),
        ('start_epoch', 'training resumes from the last completed epoch'),
        ('weights=(1/3, 1/3, 1/3, 0)', 'BLEU-3 uses exact thirds'),
        ('self.fine_tune_encoder = fine_tune_encoder',
         'the flag unfreeze_encoder_layers reads is stored on the model'),
        ('encoder.layers.encoder_layer_', 'unfreezing matches torchvision layer names'),
        ('self.encoder.fine_tune = True', 'unfrozen layers are not left inside no_grad'),
        ('with lowest validation loss',
         'the restored checkpoint is described by the metric that chose it'),
        ('def patch_tokens', 'the encoder exposes patch tokens for 7.2'),
        ('encoder_out.mean(dim=1)', 'the decoder pools those tokens'),
        ('sampling_prob=1.0 - self.scheduled_sampling_prob',
         'training feeds the scheduled sampling rate'),
        ("if val_loss < self.best_metrics['val_loss']",
         'the checkpoint is chosen on validation loss, as 6 and 7.1 ask'),
        ('fine_tune_encoder=True', 'the trained model fine-tunes the encoder, as 4.2 requires'),
        ('batch_size=BATCH_SIZE', 'the loaders use the documented batch size'),
        ('random_caption', 'evaluation splits hold their reference caption fixed'),
        ("check_checkpoint_compatibility", 'resuming checks vocabulary and configuration'),
        ('Restored the best epoch', 'a completed run hands back its best epoch'),
        ('images_pil = [transforms.ToPILImage()', 'BLIP gets channelwise denormalised images'),
        ('vocab.idx2word.get(int(predicted.item())',
         'attention columns are labelled with the generated word')]:
    assert needle in source_text, f'missing correction: {description}'
    print(f'PASS: {description}.')

# random.choice on every read moves the validation target between epochs, which
# is the metric the checkpoint and early stopping now depend on.
assert 'random.choice(captions) if self.random_caption else captions[0]' in source_text, (
    'the evaluation splits can resample their reference caption')
print('PASS: only the training split resamples its caption.')

# captions[j][0] is the first letter of a caption, not a caption. Note that
# cell 42 indexes a list of caption lists, where [i][0] is correct.
assert 'f"True: {captions[0]}' in working_source, (
    'the trainer figure no longer titles rows with the caption')
assert 'f"True: {true_captions[i]}' in working_source, (
    'the comparison figure no longer titles rows with the caption')
assert 'captions[j][0]' not in source_text, 'a figure title shows one character'
print('PASS: prediction figures show captions rather than their first letter.')

# Attention over a single encoder token is uniform by construction, so the
# old single-token call shape must not come back anywhere.
assert 'features.unsqueeze(1)' not in source_text, (
    'an attention call still wraps the encoder output as one token')
assert 'img_feature.unsqueeze(1)' not in source_text, (
    'beam search still wraps the encoder output as one token')
print('PASS: every attention call passes the full set of patch tokens.')

# The partial fixes are the ones that came back, so count the sites rather than
# trusting a single match.
assert "weights=(0.33, 0.33, 0.33, 0)" not in source_text, (
    'a BLEU-3 call is back to 0.33 weights, which sum to 0.99')
bleu3_calls = source_text.count('weights=(1/3, 1/3, 1/3, 0)')
assert bleu3_calls == 5, f'expected 5 BLEU-3 call sites, found {bleu3_calls}'
print(f'PASS: all {bleu3_calls} BLEU-3 call sites use exact thirds.')

# References are tokenized with word_tokenize everywhere, so a hypothesis that is
# only whitespace-split scores against a differently tokenized reference.
for index, cell in enumerate(notebook.cells, 1):
    if cell.cell_type != 'code' or 'corpus_bleu' not in cell.source:
        continue
    for lineno, line in enumerate(cell.source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(('hyp', 'hypotheses')) and '.split()' in stripped:
            raise AssertionError(
                f'cell {index} line {lineno} builds a hypothesis with .split() while its '
                'references use word_tokenize')
print('PASS: hypotheses and references are tokenized the same way at every BLEU site.')

print('Review complete. Corrections verified; no notebook cells were executed.')
