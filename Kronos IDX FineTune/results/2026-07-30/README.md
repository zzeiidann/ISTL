# Kronos IDX experiment results — 30 July 2026

This directory consolidates two completed experiment archives containing three
distinct model checkpoints. The original ZIP files remain in `~/Downloads` as
backups and were not modified.

## Runs

### `validated-no-refit-e15`

- Profile: `80gb_dynamic_no_refit`
- Selected epoch: 15
- Best July validation loss: `2.5651533603668213`
- Final model: `kronos_base_idx_all/best_model/model.safetensors`
- Forecast top five: SSMS, MGNA, TAMA, RSCH, BAIK
- Purpose: preferred independently validated forecast model

### `refit-run-e4`

- Profile: `80gb_dynamic_refit`
- Selected epoch: 4
- Best July validation loss: `2.594270774296352`
- Validated model: `kronos_base_idx_all/best_model/model.safetensors`
- Unvalidated refit model: `production_model/model.safetensors`
- Forecast top five: HOPE, NANO, MGNA, MCAS, WGSH
- Purpose: retained comparison run; its production refit has no independent
  post-refit validation

## SHA-256 model checksums

```text
4a03ea7e813528a4e1e49e8a8bf340b9d46bc7512b5493e582ce6cdd28ccae1f  validated-no-refit-e15/kronos_base_idx_all/best_model/model.safetensors
d5d20980c22fde34e5026eca4c6114e9f4979c0fe2c12de898488004ce773c3f  refit-run-e4/kronos_base_idx_all/best_model/model.safetensors
944d1cac75359d6db7f3ccabde6731247de12f3fe41e81340bb26b95026c64c0  refit-run-e4/production_model/model.safetensors
```

## Source archive checksums

```text
8e39998cbe1e3d6be27ef9b32b283e6414bed589657cc9c315ab9e9094cf0207  modelnorefit25615.zip
9e79a2f583c8b5753fb0111418b3d69c0308c0cccedc57d92e54e601d54846cb  modelrefit1284.zip
```

Both source archives passed `unzip -t` integrity checks before extraction.
