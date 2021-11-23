# Commands for Reproducing Paper Results

In all the following commands, replace the text within `<>` with the appropriate argument. The following commands denote the choice of arguments for running the training and evaluation code for the experiments in the paper. The only difference between the commands for the baseline and proposed models is in the `--model` argument, along with the necessary `--component` argument for the Social Process and Variational Encoder-Decoder models:

- baselines: `--model NEURAL_PROCESS` or `--model VAE_SEQ2SEQ --comonent <MLP/RNN>`
- proposed: `--model SOCIAL_PROCESS --comonent <MLP/RNN>`

For simplicity, these are denoted as `[model arguments]` in the commands below. Replace the entire text with one of the two options above. Additionally, the optional arguments are denoted within `[]`.

## Synthetic Glancing Behavior

Training:

```
python -m run.run_synthetic_glancing --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir <output directory> [model arguments] --skip_normalize_rot --data_dim 1 --dropout 0.25 --max_epochs 1000 --r_dim 32 --z_dim 32 --override_hid_dims --hid_dim 32
```

Example complete command
```
python -m run.run_synthetic_glancing --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir synthetic-glancing/phased-sine-all/sp/rnn --model SOCIAL_PROCESS --component RNN --skip_normalize_rot --data_dim 1 --dropout 0.25 --max_epochs 1000 --r_dim 32 --z_dim 32 --override_hid_dims --hid_dim 32
```

Evaluation:

```
python -m run.run_toy_sine --gpus "0" --outdir <dir containing checkpoint> --future_len 10 [model arguments] --waves_file phased-sin-with-stops.npy --data_dim 1 --dropout 0.25 --skip_normalize_rot --ckpt_fname <checkpoint file name> --test
```

## Real-world Behavior

Training:

For the simplest *-latent* variants of the meta-learning models, the template commands for the datasets are as follows -

```
# MatchNMingle
python -m run.train_dataset --dataset_root mnm --data_file mnm-hbp.h5 --feature_set HBP --gpus 1 --observed_len 4 --future_len 4 --max_future_offset 80 --fix_future_len --time_stride 20 --nposes 2 --data_dim 14 [model arguments] --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 [--pooler_nout 64] --override_hid_dims --hid_dim 64 --out_dir <output directory>
```

```
# Haggling
python -m run.train_dataset --dataset_root panoptic-haggling --data_file haggling-hbps.h5 --feature_set HBPS --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --nposes 2 --data_dim 15 [model arguments] --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4 --r_dim 64 --z_dim 64 [--pooler_nout 64] --override_hid_dims --hid_dim <dim for hidden layers> --out_dir <output directory>
```

For the other variants, the following additional options are needed :


 *-uniform* : `--use_deterministic_path`
 *-dot* : `--use_deterministic_path --attention_type DOT`
 *-mh* : `--use_deterministic_path --attention_type MULTIHEAD --attention_rep RNN`

For the MLP backbone multihead attention models, use the arguments `--attention_rep MLP --attention_qk_dim <r_dim>`.


Evaluation:

The template evaluation commands are as follows:

```
# MatchNMingle
python -m run.test_dataset --gpus 0 --dataset_root mnm --data_file mnm-hbp.h5 --feature_set HBP \
    --observed_len 4 --future_len 4 --fix_future_len --max_future_offset 80 --time_stride 20 --batch_size 128 \
    [model arguments] --ckpt_relpath <path to the checkpoint to load> \
    --project_rot --ndata_workers 4 --results_dir <output directory>  [--context_regime FIXED]
```

```
# Haggling
python -m run.test_dataset --gpus 1 --dataset_root panoptic-haggling --data_file haggling-hbps.h5 --feature_set HBPS \
    --observed_len 60 --future_len 60 --fix_future_len --max_future_offset 150 --time_stride 1 --batch_size 128 \
    [model arguments] --ckpt_relpath <path to the checkpoint to load> \
    --project_rot --ndata_workers 4 --results_dir <output directory>  [--context_regime FIXED]
```

The command writes the test metrics as a pickle file and a summary of the metrics as a csv.