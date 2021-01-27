![supported platforms](https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20%7C%20Windows%20(soon)-929292)
![supported python versions](https://img.shields.io/badge/python-%3E%3D%203.6-306998)
![dependencies status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)
[![style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![license MIT](https://img.shields.io/badge/licence-MIT-green)
[![discord badge](https://img.shields.io/badge/discord-join-6E60EF)](https://discord.gg/GB2vwsF)

# MuZero General

A commented and [documented](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation) implementation of MuZero based on the Google DeepMind [paper](https://arxiv.org/abs/1911.08265) (Nov 2019) and the associated [pseudocode](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py).
It is designed to be easily adaptable for every games or reinforcement learning environments (like [gym](https://github.com/openai/gym)). You only need to add a [game file](https://github.com/werner-duvaud/muzero-general/tree/master/games) with the hyperparameters and the game class. Please refer to the [documentation](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation) and the [example](https://github.com/werner-duvaud/muzero-general/blob/master/games/cartpole.py).

MuZero is a state of the art RL algorithm for board games (Chess, Go, ...) and Atari games.
It is the successor to [AlphaZero](https://arxiv.org/abs/1712.01815) but without any knowledge of the environment underlying dynamics. MuZero learns a model of the environment and uses an internal representation that contains only the useful information for predicting the reward, value, policy and transitions. MuZero is also close to [Value prediction networks](https://arxiv.org/abs/1707.03497). See [How it works](https://github.com/werner-duvaud/muzero-general/wiki/How-MuZero-works).

...[ORIGINAL README](./ORIGINAL_README.md)

# Mokemokechicken's Experiments Results

## Note
This original repository is very nice.
However, it has some problems in two player games and needs to be patched.
Check [all_patch_for_master](https://github.com/mokemokechicken/muzero-general/tree/all_patch_for_master) branch if you try muzero-general.

## Results

- [TicTacToe](./good_results/tictactoe-2021-01-18--01-16-32/): Almost Perfect Trained.
- [connect4](./good_results/connect4-2021-01-18--07-07-21/): Good Trained(Always win to expert).
- [animal_shogi](./good_results/animal_shogi-2021-01-24--19-21-10/): Good Trained(Somewhat strong).
