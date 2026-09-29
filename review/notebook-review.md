# Original notebook correctness review

Cell numbers below are one-based, counting Markdown and code cells.

This records the defects found in the notebook this project started from, before any of them were fixed. The source notebook itself is not in the repository; `notebooks/image-captioning.ipynb` is the corrected version, and every finding listed here is asserted by `scripts/review_original.py` so it cannot silently return. Kept as the provenance record of what changed and why.

## Findings requiring correction or an explicit methodological decision

1. **BLEU references are malformed (cells 19, 38, 42, 56, 59).** The dataset returns one randomly chosen caption string as `original_caption`. Evaluation iterates that string as if it were a list of five captions, producing character-sized references. Figure labels using `caption[0]` similarly display only the first character. Cell 49 wraps a complete string correctly but still uses only one random reference. All five reference captions must survive batching, and the evaluators must agree on tokenization and reference structure. Existing BLEU-based conclusions and checkpoint selection are unreliable. NLTK documents the required nested reference structure in its [BLEU API](https://www.nltk.org/api/nltk.translate.bleu_score.html).

2. **The attention maps are not spatial attention (cells 25, 27, 38, 47).** The encoder projects ViT's single class-token output and returns `[batch, 1, embed_size]`. Softmax over one key is always one, independently of the decoder query. Eight attention heads are not eight spatial image patches. Reshaping head weights or enlarging a one-pixel map cannot reveal image regions. The [Torchvision implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html) returns the class token from its standard forward method. Whether to retain a global-feature baseline or expose spatial patch tokens needs discussion; this review does not redesign the model.

3. **The active training run does not fine-tune ViT (cells 25, 31, 38).** Training constructs a fresh model with `fine_tune_encoder=False` by default. Its ViT parameters are frozen and its forward pass uses `no_grad`. The optional unfreeze helper also checks a parent attribute never initialized by the constructor, matches `encoder.layers.8` rather than Torchvision's `encoder.layers.encoder_layer_8`, and does not update `encoder.fine_tune`. Setting only `requires_grad` would not remove the forward-pass gradient guard. The earlier demonstration modifies a different model from the one trained.

4. **The best in-memory checkpoint is not a snapshot (cells 38, 44).** Assigning `self.model.state_dict()` to `best_metrics['state_dict']` retains tensor references that can change during later training. The disk save is separate, but final evaluation reloads the mutable in-memory mapping. A deep copy or explicit reload of the selected disk checkpoint is needed. The selection criterion is BLEU-4, while cell 44 claims lowest validation loss.

5. **Attention visualization has an undefined input (cell 47).** `plot_attention_for_samples(samples, ...)` is called without a notebook-level definition of `samples`. A fresh ordered run reaches a `NameError` there if earlier cells complete. Its token-label alignment also needs checking against the actual generated sequence.

6. **Scheduled sampling is advertised but not applied (cell 38 and following prose).** The teacher-forcing probability is updated and logged. Training always supplies `captions[:, :-1]`, and the decoder always uses the ground-truth embeddings. There is no input-selection step using this probability. Validation monitors beam search, not the per-epoch greedy BLEU described in the opening requirements.

7. **The repetition penalty rewards repetition (cell 38).** Log-probabilities are nonpositive. Dividing a repeated token's log-probability by 1.2 makes its score larger, so it becomes more attractive when beams are sorted descending. This is the opposite of a penalty.

8. **The final gradient-accumulation group is handled differently (cell 38).** The remainder branch clips scaled gradients without first calling `scaler.unscale_`. It also continues dividing loss by four even when fewer than four batches remain, reducing that update. Full groups use the correct unscale-before-clip order. This affects runs whose batch count is not divisible by four.

9. **Evaluation data are stochastic (cell 19).** Validation/test datasets randomly choose a caption, and vocabulary token dropout also applies to their loss targets. Validation shuffles examples. Loss-based monitoring is therefore noisy even when model weights are unchanged. Image-level splitting before vocabulary construction is a sound starting point, and the vocabulary is built only from training captions. Exact split files, dataset checksums and vocabulary serialization are still missing.

10. **BLIP preprocessing and comparison are inconsistent (cells 52, 56, 59).** Cell 56 min-max scales the entire normalized batch instead of reversing ImageNet normalization, then passes those float images to the processor without an explicit rescaling policy. Cell 59 correctly reverses ImageNet normalization and converts to PIL, but evaluates the comparison on `val_loader` despite the stated test comparison. Use the same held-out image IDs, references and metric protocol; investigate the processor configuration before fixing preprocessing. The [BLIP model card](https://huggingface.co/Salesforce/blip-image-captioning-large) demonstrates the intended processor workflow.

11. **Checkpoint persistence and recovery are incomplete (cell 38).** Only best model weights are saved to relative `best_model.pth`. There is no last-epoch resume checkpoint, optimizer/scheduler/scaler restoration, vocabulary/configuration snapshot, or Drive backup. A Colab reset can lose progress. These are engineering additions to discuss after correctness decisions, not permission to replace the model or workflow.

## Presentation, setup and resource issues

- The configuration and prose say batch size 32, but the actual dataloaders use 64 (cells 19, 38, 40). Several configuration fields are descriptive rather than wired into consumers.
- Cell 42 loads and concatenates the whole test image tensor collection merely to display three samples. This increases peak CPU memory unnecessarily. Beam search loops over individual images and beams, so full validation each epoch may be slow.
- Some trainer plotting methods close figures without showing or saving them when W&B is disabled. The prediction helper reuses a subplot for multiple images and indexes reference strings as characters. These are separate from the useful existing figure layouts, which should be retained.
- `warnings.filterwarnings('ignore')` hides relevant compatibility warnings. Graphviz is imported before dependency setup. Transformers and other imports are not covered by a reproducible environment specification. Colab-specific upload and shell commands preclude a portable local run without setup changes.
- W&B is imported, installed and enabled by default. Preserve the source now, but disable/remove this integration before the intended no-W&B run. Do not commit uploaded Kaggle credentials.
- Training augmentation can remove captioned objects or change caption-relevant color/direction. This is a modeling tradeoff to review, not proof that every augmentation is wrong.
- The standalone label-smoothing loss averages over padding after masking it to zero. The active trainer filters padding first, so this particular denominator issue does not affect its default training/validation calls.
- BLEU-3 weights use `(0.33, 0.33, 0.33)` rather than exact thirds. This is a minor metric-definition discrepancy compared with the reference-caption bug.
- Final prose attributes scores to overfitting, caption style and BLIP pretraining without valid comparative evidence. It also includes assignment instructions, blank answers, a deadline timer, an object-detection diagram name, and prose inside Python fences. Preserve attribution and technical content when polishing later; do not present these historical conclusions as validated findings.

## Review limits and next steps

The accompanying script checks the original fingerprint, notebook format, cell syntax and credential patterns, then reproduces selected issues with tiny CPU-only fixtures. It does not train a model, download weights, access Kaggle, authenticate to W&B or execute notebook setup. A passing review script confirms the checks ran and the defects were reproduced; it does not certify the original notebook as correct.

First agree on the reference-caption contract and the intended spatial attention/fine-tuning behavior. Then make narrowly scoped corrections with the existing sequence and figures retained. Reproducible data, durable checkpoints, Colab execution and the final project narrative follow. The notebook remains untouched until those changes are discussed.
