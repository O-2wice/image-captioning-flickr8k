# Image Captioning on Flickr8k

[![Read the write-up](https://img.shields.io/badge/read-the%20write--up-a8431c)](https://o-2wice.github.io/image-captioning-flickr8k/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/O-2wice/image-captioning-flickr8k/blob/main/notebooks/image-captioning.ipynb)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)
![Encoder](https://img.shields.io/badge/encoder-ViT--B%2F16-3f477f)
![Decoder](https://img.shields.io/badge/decoder-LSTM%20%2B%20attention-3f477f)
![Dataset](https://img.shields.io/badge/data-Flickr8k-blue)
![Reference](https://img.shields.io/badge/reference-BLIP-purple)
![BLEU-4](https://img.shields.io/badge/BLEU--4-0.2036-a8431c)

A captioning model that writes a sentence about a photograph and can show you
which part of the image it was looking at for every word. A pretrained ViT-B/16
encoder exposes 196 positioned patch tokens, an 8-head scaled dot-product
attention layer picks among them at each step, and an LSTM decoder produces the
caption one token at a time. Salesforce BLIP serves as a zero-shot reference.

**[Read the write-up](https://o-2wice.github.io/image-captioning-flickr8k/)** for
the method and the results. It includes an attention explorer driven by real
per-token weights from the trained checkpoint, and a live BLEU calculator.

## Results

Held-out test split, 1,214 images, beam search with width 5, scored against all
five references with the same tokenisation on both sides.

| | Custom (ViT + LSTM) | BLIP, zero-shot |
| --- | ---: | ---: |
| BLEU-1 | **0.6557** | 0.6162 |
| BLEU-2 | **0.4504** | 0.4427 |
| BLEU-3 | **0.3063** | 0.3057 |
| BLEU-4 | 0.2036 | **0.2080** |
| Average | **0.4040** | 0.3931 |

Twenty epochs, 45,727,184 trainable parameters, best validation loss 3.8039 at
epoch 20. Validation loss was still falling when the schedule ended and early
stopping never triggered, so the model is undertrained and these scores are a
floor rather than a converged result.

Leading on three of four is not beating BLIP. The custom model learned
Flickr8k's register and BLEU rewards overlap with Flickr8k references; BLIP
writes in its own register and loses overlap for phrasing that describes the
image perfectly well. The write-up shows captions where BLIP is clearly better
English and scores lower.

## How it works

| Stage | Detail |
| --- | --- |
| Encoder | ViT-B/16, ImageNet weights, last four of twelve blocks fine-tuned |
| Features | 196 patch tokens on a 14 × 14 lattice, projected 768 → 512 |
| Attention | 8 heads, scaled dot-product, `d_k` = 64, re-queried every step |
| Decoder | LSTM cell, hidden 1024, input = word embedding ‖ context |
| Training | AdamW 3e-4, label smoothing 0.1, cosine warm restarts, AMP, batch 32 accumulated to 128 |
| Decoding | Beam search, width 5, repetition penalty 1.2 |
| Selection | Lowest validation loss, with BLEU-1..4 tracked alongside |

Because the encoder keeps the patch tokens positioned, the attention weights
reshape to 14 × 14 and can be laid over the photograph. That is what makes the
explorer in the write-up possible.

## Data

Flickr8k: 8,091 photographs, 40,455 captions, exactly five per image.

| Split | Images | Captions |
| --- | ---: | ---: |
| Train | 5,663 | 28,315 |
| Validation | 1,214 | 6,070 |
| Test | 1,214 | 6,070 |

A 70/15/15 partition seeded at 42, taken over image identities so no photograph
appears on both sides of the boundary. The vocabulary keeps types occurring at
least five times in the training split: 2,512 entries including four reserved
symbols.

The notebook downloads the dataset from Kaggle's public URL, so no credentials
or API keys are needed, and verifies it against a pinned archive checksum before
extracting. `reproducibility/flickr8k-v1.json` records a SHA-256 for every file.

## Running it

Open the notebook in Colab with the badge above and run it top to bottom. It
mounts Drive when available and writes a resume checkpoint every epoch carrying
model, optimizer, scheduler, scaler and history, so re-running the training cell
after a disconnect continues from the last completed epoch.

Locally, `scripts/review_original.py` and `scripts/check_review_fixes.py` run
read-only checks on the notebook source plus small CPU fixtures. They perform no
training, downloads or authentication:

```bash
pip install -r requirements.txt
python scripts/review_original.py
python scripts/check_review_fixes.py
```

They assert that attention varies with the query and sums to one, that only the
training split resamples its reference caption, that checkpoint selection
follows validation loss, that BLEU uses all five references with matched
tokenisation, and that no credentials or account identifiers appear anywhere in
the notebook.

## Trained weights

Too large for the repository and not tracked:

| File | Size | Contents |
| --- | ---: | --- |
| `best_model.pth` | 394 MB | weights of the selected epoch |
| `last_checkpoint.pth` | 1.14 GB | weights plus optimizer, scheduler, scaler and full history |

Download: *(shared link to be added)*

Place them in `checkpoints/`. Both carry the vocabulary and run configuration,
and resuming refuses a checkpoint whose vocabulary or configuration does not
match the current run rather than silently assigning weights to different words.

## Repository layout

| Path | Purpose |
| --- | --- |
| [notebooks/image-captioning.ipynb](notebooks/image-captioning.ipynb) | The project notebook, executed end to end |
| [index.qmd](index.qmd) | Source of the write-up |
| [assets/](assets/) | Figures, themes, and the exported attention data behind the explorer |
| [reproducibility/](reproducibility/) | Per-file SHA-256 manifest for the dataset |
| [review/](review/) | Correctness review of the original notebook and its provenance record |
| [scripts/](scripts/) | Read-only checks over the notebook source |
| [requirements.txt](requirements.txt) | Dependencies for those checks |

Model weights, the dataset, generated outputs and local environments are not
tracked.

## Notes

The model names subjects, actions and settings correctly on straightforward
images and grounds those words in the right pixels. It breaks syntax on longer
sentences, emits `<unk>` for rare words, and needs a beam to stay coherent:
greedy decoding scores 0.0427 BLEU-4 against beam search's 0.3409 on the same
samples. Training longer is the clearest next step, since validation loss never
flattened.

Built on a course assignment notebook by Tamás Takács and Imre Molnár,
Department of Artificial Intelligence, Eötvös Loránd University.
