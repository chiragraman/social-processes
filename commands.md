# Commands for Reproducing Paper Results

In all the following commands, replace the text within `<>` with the appropriate argument. The following commands denote the choice of arguments for running the training and evaluation code for the experiments in the paper. The only difference between the commands for the baseline and proposed models is in the `--model` argument, along with the necessary `--component` argument for Social Process models:

- baselines: `--model NEURAL_PROCESS`
- proposed: `--model SOCIAL_PROCESS --comonent <MLP/RNN>`

For simplicity, these are denoted as `[model arguments]` in the commands below. Replace the entire text with one of the two options above.

## Forecasting Synthetic Glancing Behavior

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

## Haggling

Training:

- *-latent*

```
python -m run.run_haggling --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --nposes 2 --data_dim 15 [model arguments] --lr 3e-5  --max_epochs 500 --fix_future_len --train_all --dropout 0.25 --reg 1e-6 --r_dim 64 --z_dim 64 --pooler_nout 64 --override_hid_dims --hid_dim 64 --out_dir <output directory>
```

 - *-latent+det*

```
python -m run.run_haggling --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --nposes 2 --data_dim 15 [model arguments] --lr 3e-5 --schedule_lr --max_epochs 500 --fix_future_len --train_all --dropout 0.25 --reg 1e-6 --r_dim 64 --z_dim 64 --pooler_nout 64 --override_hid_dims --hid_dim 64 --out_dir <output directory> --use_deterministic_path
```

 - *-dot*

```
python -m run.run_haggling --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --nposes 2 --data_dim 15 [model arguments] --lr 3e-5 --max_epochs 500 --fix_future_len --train_all --dropout 0.25 --reg 1e-6 --r_dim 64 --z_dim 64 --pooler_nout 64 --override_hid_dims --hid_dim 64 --out_dir <output directory> --use_deterministic_path --attention_type DOT
```

 - *-multihead*

```
python -m run.run_haggling --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --nposes 2 --data_dim 15 [model arguments] --lr 3e-5 --max_epochs 500 --fix_future_len --train_all --dropout 0.25 --reg 1e-6 --r_dim 64 --z_dim 64 --pooler_nout 64 --override_hid_dims --hid_dim 64 --out_dir <output directory> --use_deterministic_path --attention_type MULTIHEAD --attention_rep RNN
```

For the MLP backbone models, use the arguments `--attention_rep MLP --attention_qk_dim <r_dim>`.

Evaluation:

```
python -m run.run_haggling_test --gpus 1 --observed_len 60 --future_len 60 --fix_future_len --max_future_offset 150 --time_stride 1 --batch_size 128 [model arguments] --load_proc CKPT --ckpt_path <path to checkpoint> --project_rot --ndata_workers 4
```

The command writes the test metrics as a pickle file. As a convenience, the
following command can be used to convert the metrics to a csv.

```
python -m run.summarize_metrics --metrics_path <path to the metrics file>
```


##### SLURM

./slurm/train_panoptic.sh -n sp1024 -o $VAULT/social-processes/exp/sp/rnn/stoch-1024 -m run.run_panoptic -a " --component RNN --lr 1e-5 --schedule_lr --lr_steps 16 18 20 --max_epochs 500 --train_all --nlayers 1 --nz_layers 3 --r_dim 64 --z_dim 64 --pooler_nout 64 --override_hid_dim --hid_dim 1024 --weight_decay 1e-3 --batch_size 32"
