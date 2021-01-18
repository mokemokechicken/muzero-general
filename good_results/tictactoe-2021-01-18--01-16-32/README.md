# Game: TicTacToe

- Date: 2021-01-18
- Environment:
  - GPU: GeForce GTX 1080 (8GB Memory)
  - CPU: 8 CPU (Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz)
  - Memory: 64GB
  - OS: CentOS 7.5
- SourceCode Version: [651109a](https://github.com/mokemokechicken/muzero-general/tree/651109a)
- Trained Model: download [model.checkpoint](./model.checkpoint)

## TensorBoard
![Image1](./TensorBoard-1.png)
![Image2](./TensorBoard-2.png)
![Image3](./TensorBoard-3.png)

## Parameters

| Parameter | Value |
|-----|-----|
| seed | 0 |
| max_num_gpus | None |
| observation_shape | (3, 3, 3) |
| action_space | [0, 1, 2, 3, 4, 5, 6, 7, 8] |
| players | [0, 1] |
| stacked_observations | 0 |
| muzero_player | 0 |
| opponent | expert |
| num_workers | 4 |
| selfplay_on_gpu | False |
| max_moves | 9 |
| num_simulations | 25 |
| discount | 1 |
| temperature_threshold | None |
| root_dirichlet_alpha | 0.2 |
| root_exploration_fraction | 0.25 |
| pb_c_base | 19652 |
| pb_c_init | 1.25 |
| network | resnet |
| support_size | 1 |
| downsample | False |
| blocks | 1 |
| channels | 16 |
| reduced_channels_reward | 16 |
| reduced_channels_value | 16 |
| reduced_channels_policy | 16 |
| resnet_fc_reward_layers | [8] |
| resnet_fc_value_layers | [8] |
| resnet_fc_policy_layers | [8] |
| encoding_size | 32 |
| fc_representation_layers | [] |
| fc_dynamics_layers | [16, 16] |
| fc_reward_layers | [8] |
| fc_value_layers | [8] |
| fc_policy_layers | [16] |
| results_path | /home/ken/muzero-general/games/../results/tictactoe/2021-01-18--01-16-32 |
| save_model | True |
| training_steps | 500000 |
| batch_size | 256 |
| checkpoint_interval | 10 |
| value_loss_weight | 0.25 |
| train_on_gpu | True |
| optimizer | Adam |
| weight_decay | 0.0001 |
| momentum | 0.9 |
| lr_init | 0.003 |
| lr_decay_rate | 1 |
| lr_decay_steps | 10000 |
| replay_buffer_size | 3000 |
| num_unroll_steps | 3 |
| td_steps | 9 |
| PER | False |
| PER_alpha | 0.5 |
| use_last_model_value | True |
| reanalyse_on_gpu | False |
| self_play_delay | 0 |
| training_delay | 0 |
| ratio | None |
