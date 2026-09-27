# Image Captioning with Flickr8k

[![Read the write-up](https://img.shields.io/badge/read-the%20write--up-a8431c)](https://o-2wice.github.io/image-captioning-flickr8k/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/O-2wice/image-captioning-flickr8k/blob/main/notebooks/image-captioning.ipynb)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)
![Encoder](https://img.shields.io/badge/encoder-ViT--B%2F16-3f477f)
![Decoder](https://img.shields.io/badge/decoder-LSTM%20%2B%20attention-3f477f)
![Dataset](https://img.shields.io/badge/data-Flickr8k-blue)
![Reference](https://img.shields.io/badge/reference-BLIP-purple)
![BLEU-4](https://img.shields.io/badge/BLEU--4-0.2036-a8431c)

**Topics:** `image-captioning` · `vision-transformer` · `attention` · `lstm` ·
`bleu` · `beam-search` · `scheduled-sampling` · `pytorch` · `flickr8k` · `blip` ·
`deep-learning` · `computer-vision` · `nlp`


An image captioning experiment using a custom ViT-B/16 encoder, LSTM decoder and multi-head attention module, with Salesforce BLIP as a pretrained reference.

The notebook runs end to end in Colab. A completed twenty-epoch run reaches BLEU-4 of 0.2036 on the held-out test split, and the [write-up](https://o-2wice.github.io/image-captioning-flickr8k/) explains the method and reads the results.

## Repository layout

| Path | Purpose |
| --- | --- |
| [notebooks/image-captioning.ipynb](notebooks/image-captioning.ipynb) | Working notebook for the project and subsequent fixes |
| [review/notebook-review.md](review/notebook-review.md) | Initial correctness findings, referenced by original cell number |
| [review/original-notebook.json](review/original-notebook.json) | Source provenance and SHA-256 fingerprint |
| [scripts/review_original.py](scripts/review_original.py) | Notebook validation and lightweight correctness checks |

`notebooks/image-captioning.ipynb` is the sole project notebook. The provenance manifest retains historical source fingerprints; the duplicate original notebook has been removed. Original attribution remains in the project notebook.

## Corrections applied to the working notebook

The review's defects are fixed in `notebooks/image-captioning.ipynb`, and
`scripts/review_original.py` asserts each one so it cannot silently regress.

- **BLEU references were characters, not captions.** The dataset returned one
  caption string and evaluation iterated it as if it were a list of five, so
  every reference was a single letter and every BLEU score, including the one
  selecting the best checkpoint, was meaningless. All five captions now travel
  with the batch, through the trainer, the decoding comparison, the BLIP
  evaluation and the model comparison.
- **The repetition penalty rewarded repetition.** Log-probabilities are
  negative, so dividing a repeated token's score by 1.2 raised it from -0.70 to
  -0.58. It multiplies now, which lowers it.
- **The best checkpoint was not a snapshot.** It held live tensor references
  that kept changing during later training; it is a deep copy.
- **Token dropout corrupted evaluation targets.** It applied to every split, so
  validation loss moved even with frozen weights. It is training-only, and the
  validation loader no longer shuffles.
- **Gradient accumulation mishandled its final group**, clipping still-scaled
  gradients and dividing a partial group by the full step count.
- **The BLIP comparison was scored on a different tokenizer.** Its references
  went through `word_tokenize` while its hypothesis was whitespace-split and
  left capitalised, so `grass.` never matched `grass` and every sentence-final
  n-gram was lost. On a worked example the clipped unigram precision moves from
  0.571 to 0.875; the reference model was being marked down by the scoring, not
  by its captions.
- **Layer unfreezing unfroze nothing.** `unfreeze_encoder_layers` matched
  `encoder.layers.8`, but torchvision names the blocks
  `encoder.layers.encoder_layer_8`, so every parameter was set to
  `requires_grad = False` and the cell reported the unchanged count as a
  result. The encoder's `forward` also runs under `no_grad` while its
  `fine_tune` flag is false, so even a correct match produced no gradients, and
  `self.fine_tune_encoder` was read but never assigned, which raised
  `AttributeError` on a fresh model. With the three fixed, unfreezing the last
  four blocks moves trainable parameters from 5,223,012 to 33,574,500 and
  gradients reach block 11 while block 0 stays frozen.
- **The restored checkpoint was described by the wrong metric.** Selection is on
  BLEU-4, but the restore message, the docstring, the loss-curve annotation and
  the final report all said lowest validation loss, and the green marker sat at
  the BLEU-selected epoch rather than the loss minimum. The wording now names
  the metric that does the choosing.
- Also: an undefined `samples` that raised `NameError`, a BLIP comparison run on
  the validation split while described as held-out, BLEU-3 weights of 0.33
  rather than exact thirds at all five call sites, and a missing `word_tokenize`
  import.

Emoji were removed from code cells, 362 clusters across 23 cells, including 160
inside `print()` calls that appeared in the training logs. Markdown keeps its
voice.

## Credentials

The dataset downloads from Kaggle's public URL, so the notebook needs no credentials at all, and its checksum is verified against a pinned archive hash before extraction. Metrics and plots are displayed locally, and checkpoints retain training history; no external experiment-tracking account is needed. `scripts/review_original.py` scans the project notebook for credential patterns and exposed account identifiers.


## Colab runs

Training mounts Drive when it is available and writes a resume checkpoint every
epoch carrying model, optimizer, scheduler, scaler and history, so re-running
the training cell after a disconnect continues from the last completed epoch
rather than starting again. DataLoaders use workers and pinned memory on Colab.

## Matching the notebook to its brief

Four sections asked for behaviour the code did not have. Each is now
implemented, and `scripts/review_original.py` asserts it.

- **7.2 asks for attention weights to overlay on the image.** The encoder
  returned ViT's class token alone, one vector, so the attention softmax ran
  over a single position and was uniform by construction: every weight was 1,
  whatever the query. It now returns the 196 patch embeddings, which reshape to
  the 14x14 grid the plotting cells already expected. The attention module
  needed no change; it was always written for a sequence.
- **4.2 requires fine-tuning.** Training unfreezes the last four ViT blocks.
  Gradients reach block 11 while block 0 stays frozen, confirmed by running it.
- **6 and 7.1 ask for the best epoch by validation loss.** Selection was on
  BLEU-4. It now tracks the lowest validation loss and still reports BLEU.
- **Scheduled sampling was decayed, logged and checkpointed but never used.**
  The decoder now substitutes its own previous prediction at that rate, in
  training only.

The batch size follows the hyperparameter table and the config dictionary,
which both say 32; the loaders had 64 hard-coded.

## Results

Held-out test split, 1,214 images, beam search with width 5, scored against all
five references with matched tokenisation on both sides.

| | Custom (ViT + LSTM) | BLIP, zero-shot |
| --- | ---: | ---: |
| BLEU-1 | **0.6557** | 0.6162 |
| BLEU-2 | **0.4504** | 0.4427 |
| BLEU-3 | **0.3063** | 0.3057 |
| BLEU-4 | 0.2036 | **0.2080** |
| Average | **0.4040** | 0.3931 |

Twenty epochs, best validation loss 3.8039 at epoch 20, 45,727,184 trainable
parameters. Validation loss was still improving when the schedule ended, so the
model is undertrained and these numbers are a floor. The
[write-up](https://o-2wice.github.io/image-captioning-flickr8k/) explains the
method and reads the results.

## Trained weights

The checkpoints are too large for the repository and are not tracked:

| File | Size | Contents |
| --- | ---: | --- |
| `best_model.pth` | 394 MB | weights of the selected epoch |
| `last_checkpoint.pth` | 1.14 GB | weights plus optimizer, scheduler, scaler and full history |

Download: <!-- DOWNLOAD_LINK --> *(add the shared link here)*

Place them in `checkpoints/` beside the notebook. Both carry the vocabulary and
the run configuration, and resuming refuses a checkpoint whose vocabulary or
configuration does not match the current one, rather than silently assigning
weights to different words.

The dataset is not tracked either; the notebook downloads Flickr8k from Kaggle's
public URL and verifies it against a pinned archive checksum.

## Current status

The correctness defects listed above are fixed, the notebook implements what
sections 4.2, 6, 7.1, 7.2 and the scheduled-sampling note ask for, and a full
twenty-epoch run has been completed and written up. Validation loss was still
improving at the final epoch, so the reported scores are a floor rather than a
converged result.

No dataset, model weights or credentials are committed. Generated data,
outputs, checkpoints and local environments are ignored.

## Validation

The original review was checked with Python 3.11 using `nbformat`, IPython and PyTorch from the existing local detection environment. Run `python scripts/review_original.py` in an environment containing those packages. It performs no training, downloads or authentication. Successful execution validates the project notebook, checks corrections and scans for exposed credentials or account identifiers. It does not certify a training run.

Next, correct the working notebook against the documented findings, verify it, and complete the user-run Colab experiment. The project write-up follows validated results.