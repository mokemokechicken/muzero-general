import datetime
import os
import copy
from dataclasses import dataclass, astuple
from typing import Optional

import numpy
import torch


try:
    from abstract_game import AbstractGame
except ImportError:
    from .abstract_game import AbstractGame

try:
    from models import MuZeroResidualNetwork
except ImportError:
    from ..models import MuZeroResidualNetwork


BOARD_SIZE_X = 3
BOARD_SIZE_Y = 4
UNIT_KIND_NUM = 5  # Lion, Elephant, Giraph, Piyo, Chicken(Piyo Promoted)
CAPTURABLE_KIND_NUM = 3  # Elephant, Giraph, Piyo

ACTION_SPACE_SIZE = (
        (BOARD_SIZE_X * BOARD_SIZE_Y + CAPTURABLE_KIND_NUM) *  # FROM
        (BOARD_SIZE_X * BOARD_SIZE_Y) *  # TO
        2  # Promote
)


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available


        ### Game
        self.observation_shape = ((UNIT_KIND_NUM+CAPTURABLE_KIND_NUM)*2 + 1, BOARD_SIZE_Y, BOARD_SIZE_X)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(ACTION_SPACE_SIZE))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class


        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 100  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "animal_shogi"  # "resnet" / "fullyconnected"
        self.support_size = 2  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network and animal_shogi Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 4  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 256  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = self.max_moves  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < self.training_steps * 0.5:
            return 1
        elif trained_steps < self.training_steps * 0.75:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = AnimalShogi()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.
    
        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        return self.env.human_to_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        
        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return self.env.action_to_string(action_number)


@dataclass
class Move:
    from_board: Optional[int]  # (y*3 + x) or None
    from_stock: Optional[int]  # (E=0, G=1, P=2) or None
    to_board: int   # (y*3 + x)
    promotion: int  # 0 or 1(promote)

    @classmethod
    def decode_from_action_index(cls, action: int):
        """

        :param action:
        ActionSpace: combination of below
          From     H*W + 3(E G P stock) (15)
          To       H*W                  (12)
          Promote  2                    (2)
        """
        board_size = BOARD_SIZE_Y * BOARD_SIZE_X
        assert 0 <= action < (board_size+3) * board_size * 2
        promote = action % 2
        action //= 2
        to_board = action % board_size
        action //= board_size
        if action < board_size:
            from_board = action
            from_stock = None
        else:
            from_board = None
            from_stock = action - board_size  # (E=0, G=1, P=2)
        return cls(from_board, from_stock, to_board, promote)

    def encode_to_action_index(self) -> int:
        board_size = BOARD_SIZE_Y * BOARD_SIZE_X
        if self.from_stock is None:
            action = self.from_board
        else:
            action = board_size + self.from_stock
        action *= board_size * 2
        action += self.to_board * 2
        action += self.promotion
        assert 0 <= action < (board_size+3) * board_size * 2
        return action

    def from_pos(self):
        assert self.from_board is not None
        return self.from_board // BOARD_SIZE_X, self.from_board % BOARD_SIZE_X

    def to_pos(self):
        assert self.to_board is not None
        return self.to_board // BOARD_SIZE_X, self.to_board % BOARD_SIZE_X

    def clone(self):
        return Move(*astuple(self))


class AnimalShogi:
    board = None
    stocks = None
    player = 0
    _legal_actions = None

    def __init__(self):
        self.init_game()

    def init_game(self):
        # Board(H=4, W=3)
        #   player-0: L=1, E=2, G=3, P=4, C=5
        #   player-1: L=6, E=7, G=8, P=9, C=10
        # stocks for p0 = (E, G, P)
        # stocks for p1 = (E, G, P)
        self.board = numpy.array([
            [G2, L2, E2],
            [0 , P2,  0],
            [0 , P1,  0],
            [E1, L1, G1],
        ], dtype="int32")
        self.stocks = numpy.zeros((2, CAPTURABLE_KIND_NUM), dtype="int32")
        self.player = 0
        self._legal_actions = None

    def reset(self):
        self.init_game()
        return self.get_observation()

    def to_play(self):
        return self.player

    def step(self, action):
        move = Move.decode_from_action_index(action)
        if not self.is_legal(move):
            return self.get_observation(), -1, True
        win, lose, done = self.do_move(move)
        self.player = 1 - self.player
        reward = 0
        if win:
            reward = 1
        elif lose:
            reward = -1
        return self.get_observation(), reward, done

    def do_move(self, move: Move):
        self._legal_actions = None
        player = self.to_play()
        win = False
        lose = False
        done = False
        if move.from_stock is not None:  # drop
            self.stocks[player][move.from_stock] -= 1
            unit_kind = move.from_stock + 2 + player * 5  # (2,3,4 or 7,8,9)
            self.board[move.to_pos()] = unit_kind
        else:
            unit_kind = self.board[move.from_pos()]
            self.board[move.from_pos()] = 0
            if self.board[move.to_pos()] > 0:  # capture
                captured_unit_kind = self.board[move.to_pos()] % 5
                if captured_unit_kind == 1:  # Lion
                    done = win = True
                else:
                    stock_kind = [2, None, 0, 1, 2][captured_unit_kind]  # board:E, G, P, C -> stock:E, G, P, P
                    self.stocks[player][stock_kind] += 1
            self.board[move.to_pos()] = unit_kind + move.promotion
        if player == 0 and numpy.any(self.board[BOARD_SIZE_Y-1] == L2):  # Player1 Lion Try!
            lose = done = True
        elif player == 1 and numpy.any(self.board[0] == L1):  # Player0 Lion Try!
            lose = done = True
        return win, lose, done

    @staticmethod
    def is_legal_move_direction(unit_kind, from_pos, to_pos):
        diff = (to_pos[0]-from_pos[0], to_pos[1]-from_pos[1])
        return diff in ALLOWED_MOVES[unit_kind]

    def is_legal(self, move: Move):
        if move.from_stock is not None:
            remain_num = self.stocks[self.to_play()][move.from_stock]
            if remain_num < 1:
                return False
            if move.promotion == 1:
                return False
        else:
            unit_kind = self.board[move.from_pos()]
            if unit_kind == 0:  # no unit there
                return False
            elif unit_kind < 6 and self.to_play() == 1:  # opponent unit
                return False
            elif unit_kind > 5 and self.to_play() == 0:  # opponent unit
                return False
            if move.promotion == 1 and unit_kind not in (P1, P2):  # only P can promote.
                return False
            if not self.is_legal_move_direction(unit_kind, move.from_pos(), move.to_pos()):
                return False

        captured = self.board[move.to_pos()]
        if captured:
            if move.from_stock is not None:
                return False  # drop on the unit directly
            if captured < 6 and self.to_play() == 0:  # capture my team0
                return False
            if captured > 5 and self.to_play() == 1:  # capture my team1
                return False
        return True

    def get_observation(self):
        channels = []
        # board
        for kind in range(1, 11):
            ch = numpy.where(self.board == kind, 1, 0)
            channels.append(ch)
        # stock
        for player in [0, 1]:
            for kind in range(CAPTURABLE_KIND_NUM):
                ch = numpy.full_like(channels[0], self.stocks[player][kind] / 2.)
                channels.append(ch)
        # to_play
        ch = numpy.full_like(channels[0], 1 - self.to_play() * 2)
        channels.append(ch)
        return numpy.array(channels, dtype="int32")

    def legal_actions(self):
        if self._legal_actions is None:
            ret = []
            for action in range(ACTION_SPACE_SIZE):
                if self.is_legal(Move.decode_from_action_index(action)):
                    ret.append(action)
            self._legal_actions = ret
        return copy.copy(self._legal_actions)

    def human_to_action(self):
        stock_kinds = {"E": 0, "G": 1, "C": 2}

        def convert_position_string_to_pos_index(pos_str):
            try:
                pos_str = pos_str.lower()
                col = int(pos_str[0]) - 1
                row = "abcd".index(pos_str[1])
                return row * BOARD_SIZE_X + col
            except:
                return None

        # input from
        from_stock = None
        from_board = None
        to_board = None
        promotion = 0
        player = self.to_play()
        while True:
            while True:
                try:
                    from_str = input(f"From(ex: '1a', '2d', or 'E' 'G' 'C' from stock): ").strip()
                    if from_str.upper() in stock_kinds:
                        from_stock = stock_kinds[from_str.upper()]
                        if self.stocks[player][from_stock] > 0:
                            break
                        else:
                            print(f"You do not have {from_str}")
                    elif len(from_str) == 2:
                        from_board = convert_position_string_to_pos_index(from_str)
                        if from_board is None:
                            print(f"illegal position {from_str}")
                except:
                    pass
                print("Wrong input, try again")

            while True:
                try:
                    to_str = input(f"To(ex: '1a', '2d'): ").strip()
                    if len(to_str) == 2:
                        to_board = convert_position_string_to_pos_index(to_str)
                        if to_str is None:
                            print(f"illegal position {from_str}")
                        else:
                            break
                except:
                    pass
                print("Wrong input, try again")

            move = Move(from_board, from_stock, to_board, 0)
            if self.is_legal(move) and move.from_board is not None:
                m2 = move.clone()
                m2.promotion = 1
                if self.is_legal(m2):
                    pr_str = input("Promotion? [Y]/[n]: ").lower()
                    if pr_str != "n":
                        move.promotion = 1
            if self.is_legal(move):
                break
            else:
                print("Illegal Move, try again")
        return move.encode_to_action_index()

    def render(self):
        from colorama import Back

        chars = {
            0: Back.CYAN + "  " + Back.RESET,
            L1: Back.BLUE + "🐯" + Back.RESET,
            E1: Back.BLUE + "🐘" + Back.RESET,
            G1: Back.BLUE + "🐴" + Back.RESET,
            P1: Back.BLUE + "🐥" + Back.RESET,
            C1: Back.BLUE + "🐔" + Back.RESET,
            L2: Back.GREEN + "🐯" + Back.RESET,
            E2: Back.GREEN + "🐘" + Back.RESET,
            G2: Back.GREEN + "🐴" + Back.RESET,
            P2: Back.GREEN + "🐥" + Back.RESET,
            C2: Back.GREEN + "🐔" + Back.RESET,
        }
        lines = []
        for line in self.board:
            line_ch_list = []
            for kind in line:
                line_ch_list.append(chars[kind])
            lines.append("".join(line_ch_list))

        stock_lines = []
        for stocks in self.stocks:
            stock = ""
            for i, num in enumerate(stocks):
                stock += "🐘🐴🐥"[i] * num
            stock_lines.append(stock)

        print(Back.GREEN + f"stock: {stock_lines[1]}" + Back.RESET)
        print(" | 1 2 3")
        print("-+------")
        print("\n".join([f"{m}|{line}" for m, line in zip("abcd", lines)]))
        print("-+------")
        print(Back.BLUE + f"stock: {stock_lines[0]}" + Back.RESET)

    def action_to_string(self, action_number):
        move = Move.decode_from_action_index(action_number)
        if move.from_board is not None:
            from_pos, to_pos = move.from_pos(), move.to_pos()
            kind = self.board[from_pos]
            if kind == 0:
                ch = " "
            else:
                ch = "🐯🐘🐴🐥🐔"[(kind-1) % 5]
            pos_from = "123"[from_pos[1]] + "abcd"[from_pos[0]]
            pos_to = "123"[to_pos[1]] + "abcd"[to_pos[0]]
            return f"{pos_from}{pos_to}{ch}"
        else:
            to_pos = move.to_pos()
            pos_to = "123"[to_pos[1]] + "abcd"[to_pos[0]]
            ch = "🐘🐴🐥"[move.from_stock]
            return f"->{pos_to}{ch}"


# first player
L1 = 1  # Lion
E1 = 2  # Elephant
G1 = 3  # Giraph
P1 = 4  # Chick  (Piyo Piyo! or Pawn)
C1 = 5  # Chicken

# second player
L2 = 6
E2 = 7
G2 = 8
P2 = 9
C2 = 10

# move direction
UL = (-1, -1)  # Y, X
UU = (-1,  0)
UR = (-1,  1)
ML = ( 0, -1)
MR = ( 0,  1)
DL = ( 1, -1)
DD = ( 1,  0)
DR = ( 1,  1)

ALLOWED_MOVES = {
    L1: [UL, UU, UR, ML, MR, DL, DD, DR],
    L2: [UL, UU, UR, ML, MR, DL, DD, DR],
    E1: [UL, UR, DL, DR],
    E2: [UL, UR, DL, DR],
    G1: [UU, ML, MR, DD],
    G2: [UU, ML, MR, DD],
    P1: [UU],
    P2: [DD],
    C1: [UL, UU, UR, ML, MR, DD],
    C2: [DL, DD, DR, ML, MR, UU],
}


class AnimalShogiNetwork(MuZeroResidualNetwork):
    def encode_hidden_and_action(self, encoded_state, action):
        super().encode_hidden_and_action(encoded_state, action)
