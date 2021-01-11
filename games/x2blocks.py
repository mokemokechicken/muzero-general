import datetime
import os

import numpy
import torch

try:
    from abstract_game import AbstractGame
except ImportError:
    from .abstract_game import AbstractGame


NUM_BLOCK_KINDS = 13
NUM_NEW_BLOCK_KINDS = 6
BOARD_SIZE_X = 5
BOARD_SIZE_Y = 7


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available


        ### Game
        self.observation_shape = (NUM_BLOCK_KINDS+NUM_NEW_BLOCK_KINDS+1, BOARD_SIZE_Y, BOARD_SIZE_X)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(BOARD_SIZE_X))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 10000  # Maximum number of moves if game is not finished before
        self.num_simulations = 20  # Number of future moves self-simulated
        self.discount = 0.95  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 2  # Number of blocks in the ResNet
        self.channels = 32  # Number of channels in the ResNet
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
        self.batch_size = 64  # Number of parts of games to train on at each training step
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
        self.num_unroll_steps = 1  # Number of game moves to keep for every batch element
        self.td_steps = 3  # Number of steps in the future to take into account for calculating the target value
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
        self.env = X2Blocks()

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
        return 0

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


class X2Blocks:
    # bottom = (BOARD_SIZE_Y - 1), top = 0. blocks stack from bottom to top.
    board = None
    upgraded_count = None
    next_number = None

    def __init__(self):
        self.init_game()

    def init_game(self):
        self.board = numpy.zeros((BOARD_SIZE_Y + 1, BOARD_SIZE_X), dtype="int32")  # row=0 is extra line.
        self.upgraded_count = 0
        self.next_number = self.gen_next_number()

    def reset(self):
        self.init_game()
        return self.get_observation()

    def gen_next_number(self):
        return numpy.random.randint(1, NUM_NEW_BLOCK_KINDS + 1)

    def step(self, action):
        score = self.do_action(action)
        max_number = numpy.max(self.board)
        if NUM_BLOCK_KINDS < max_number:
            up_diff = max_number - NUM_BLOCK_KINDS
            self.board[numpy.where(self.board > 0)] -= up_diff
            self.board = numpy.clip(self.board, 0, NUM_BLOCK_KINDS)
            self.upgraded_count += up_diff

        done = any(self.board[0] > 0)
        reward = 0
        if score > 1:
            reward = numpy.log2(score)
        self.next_number = self.gen_next_number()
        return self.get_observation(), reward, done

    def do_action(self, action):
        """

        :param int action:
        """
        bottom = numpy.where(self.board[:, action] == 0)[0].max()
        self.board[bottom, action] = self.next_number
        rate = 1
        total_score = 0
        while True:
            score = self.process_connected(self.board, action)
            if not score:
                break
            self.drop_numbers(self.board)
            total_score += score * rate
            rate += 0.5
        return total_score

    def drop_numbers(self, board):
        """

        :param numpy.array board:
        """
        sy, sx = board.shape
        for col in range(sx):
            for row in range(sy-2, -1, -1):
                if board[row, col] == 0 or board[row+1, col] > 0:
                    continue
                for under in range(row+1, sy):
                    if board[under, col] > 0:
                        board[under-1, col] = board[row, col]
                        board[row, col] = 0
                        # print(f"drop({2**board[under-1, col]}): ({row}, {col}) -> ({under-1, col})")
                        break
                    elif under == sy - 1:
                        board[under, col] = board[row, col]
                        board[row, col] = 0
                        # print(f"drop({2**board[under, col]}): ({row}, {col}) -> ({under, col})")
                        break

    def process_connected(self, board, action):
        """

        :param numpy.ndarray board:
        :param int action:
        :return:
        """
        connected_list = self.get_connected_list(board)
        score = 0
        if not connected_list:
            return score
        # print(f"connect: {connected_list}")
        for pos_list in connected_list:
            number = board[pos_list[0]]
            new_number = number + len(pos_list) - 1
            for pos in pos_list:
                board[pos] = 0
            new_pos = pos_list[int(numpy.argmin(([abs(p[1]-action) for p in pos_list])))]
            board[new_pos] = new_number
            # print(f"{new_pos} = {2**new_number}")
            score += 2 ** (new_number + self.upgraded_count) * (len(pos_list)-1)
        return score

    def get_connected_list(self, board):
        checked = numpy.zeros_like(board)
        connected_list = []
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                number = board[row, col]
                if not number:
                    continue
                pos_list = self.check_connected(board, row, col, number, checked)
                if len(pos_list) >= 2:
                    connected_list.append(pos_list)
        return connected_list

    def check_connected(self, board, row, col, number, checked):
        """

        :param numpy.ndarray board:
        :param int row:
        :param int col:
        :param int number:
        :param numpy.ndarray checked:
        :return:

        >>> env = X2Blocks()
        >>> b = numpy.zeros((4, 4), dtype="int32")
        >>> b[1, 2] = b[1, 3] = 1
        >>> env.check_connected(b, 1, 2, 1, numpy.zeros_like(b))
        [(1, 2), (1, 3)]
        >>> b[2, 2] = b[3, 3] = 1
        >>> checked = numpy.zeros_like(b)
        >>> env.check_connected(b, 1, 2, 1, checked)
        [(1, 2), (2, 2), (1, 3)]
        >>> env.check_connected(b, 1, 3, 1, checked)  # already checked
        []
        >>> env.check_connected(b, 3, 3, 1, checked)
        [(3, 3)]
        >>> b = numpy.zeros((4, 4), dtype="int32")
        >>> b[2, 2] = b[3, 2] = 3
        >>> env.check_connected(b, 2, 2, 3, numpy.zeros_like(b))
        [(2, 2), (3, 2)]
        """
        if checked[row, col] != 0 or board[row, col] != number:
            return []
        checked[row, col] = 1
        pos_list = [(row, col)]
        sy, sx = board.shape
        if row > 0 and board[row - 1, col] == number:
            pos_list += self.check_connected(board, row - 1, col, number, checked)
        if row < sy - 1 and board[row + 1, col] == number:
            pos_list += self.check_connected(board, row + 1, col, number, checked)
        if col > 0 and board[row, col - 1] == number:
            pos_list += self.check_connected(board, row, col - 1, number, checked)
        if col < sx - 1 and board[row, col + 1] == number:
            pos_list += self.check_connected(board, row, col + 1, number, checked)
        return pos_list

    def get_observation(self):
        channels = []
        for k in range(1, NUM_BLOCK_KINDS+1):
            ch = numpy.where(self.board[1:, :] == k, 1, 0)
            channels.append(ch)
        for k in range(1, NUM_NEW_BLOCK_KINDS+1):
            if self.next_number == k:
                ch = numpy.ones_like(channels[0])
            else:
                ch = numpy.zeros_like(channels[0])
            channels.append(ch)
        up_ch = numpy.ones_like(channels[0]) * (self.upgraded_count * 0.1)
        channels.append(up_ch)
        return numpy.array(channels, dtype="int32")

    def legal_actions(self):
        return list(range(BOARD_SIZE_X))

    def human_to_action(self):
        while True:
            try:
                action = int(input(f"Input(0 ~ {BOARD_SIZE_X-1}) "
                                   f"NextNumber={2**(self.next_number+self.upgraded_count)}: ").strip())
                if action in self.legal_actions():
                    return action
            except:
                pass
            print("Wrong input, try again")

    def render(self):
        print(numpy.array((2 ** (self.board+self.upgraded_count)) * numpy.where(self.board > 0, 1, 0), dtype="int32"))
        print(f"upgraded count: {self.upgraded_count}")

    def action_to_string(self, action_number):
        return str(action_number)
