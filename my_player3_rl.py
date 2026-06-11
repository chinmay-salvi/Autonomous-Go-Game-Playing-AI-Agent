import random
import numpy as np
import torch
from collections import deque
from typing import NamedTuple, List
from torch import nn


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Data containers
class Transition(NamedTuple):
    """
    A class to represent a single transition in the environment.

    Attributes:
        state (int): The current state.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (int): The next state after the action.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray


class BatchTransition(NamedTuple):
    """
    A class to represent a batch of transitions.

    Attributes:
        states (np.ndarray): Array of states.
        actions (np.ndarray): Array of actions.
        rewards (np.ndarray): Array of rewards.
        next_states (np.ndarray): Array of next states.
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray

    def size(self) -> int:
        """
        Get the size of the batch.

        Returns:
            int: The number of transitions in the batch.
        """
        return len(self.states)

    @staticmethod
    def from_list(transitions: List[Transition]) -> "BatchTransition":
        """
        Create a BatchTransition from a list of Transition objects.

        Args:
            transitions (List[Transition]): List of Transition objects.

        Returns:
            BatchTransition: A BatchTransition object.
        """
        states, actions, rewards, next_states = zip(*transitions)
        return BatchTransition(
            states=np.array(states, dtype=int),
            actions=np.array(actions, dtype=int),
            rewards=np.array(rewards, dtype=float),
            next_states=np.array(next_states, dtype=int),
        )


class ReplayBuffer:
    """
    A class to represent a replay buffer for storing transitions.

    Attributes:
        buffer (deque): A deque to store transitions with a maximum length.
                        When the capacity is reached, we drop earliest items from the buffer
                        (i.e., "Last In Last Out" like a stack).
    """

    def __init__(self, capacity: int = 10000) -> None:
        """
        Initialize the replay buffer.

        Args:
            capacity (int): The maximum number of transitions to store. Defaults to 10000.
        """
        self.buffer = deque(maxlen=capacity)  # DO NOT MODIFY

    @property
    def size(self) -> int:
        """
        Get the current size of the buffer.

        Returns:
            int: The number of transitions in the buffer.
        """
        return len(self.buffer)  # DO NOT MODIFY

    def push(self, transition: Transition) -> None:
        """
        Add a transition to (the end of) buffer.

        Args:
            transition (Transition): The transition to add.
        """
        ### YOUR CODE HERE ###
        self.buffer.append(transition)
        ### END OF YOUR CODE ###

    def sample(self, batch_size: int) -> BatchTransition:
        """
        Sample a batch of transitions from the buffer without replacement.

        Args:
            batch_size (int): The number of transitions to sample from the replay buffer.
                            Must be greater than 0.

        Returns:
            BatchTransition: A batch of sampled transitions. If the number of transitions
                            in the buffer is less than the batch size, return all the transitions
                            in the buffer (as a BatchTransition).

        Notes:
            - Sampling should be done *without replacement*, meaning each transition can only be
              selected once per batch
        """
        ### YOUR CODE HERE ###
        if len(self.buffer) < batch_size:
            return BatchTransition.from_list(list(self.buffer))

        return BatchTransition.from_list(random.sample(self.buffer, batch_size))
        ### END OF YOUR CODE ###

    def clear(self) -> None:
        """
        Clear the buffer.
        """
        self.buffer.clear()  # DO NOT MODIFY


class DQNCNN(nn.Module):
    def __init__(self, board_channels=2, scalar_input_size=3, output_size=26):
        super(DQNCNN, self).__init__()
        self.board_channels = board_channels

        # Convolutional layers for board (2 channels: current + previous board)
        self.conv1 = nn.Conv2d(board_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        # Flattened conv output: 64 channels * 5x5 = 1600
        conv_out_features = 64 * 5 * 5

        # Fully connected layers: combine scalar info and conv features
        self.fc1 = nn.Linear(conv_out_features + scalar_input_size, 512)
        self.fc2 = nn.Linear(512, 512)

        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.out = nn.Linear(512, output_size)

    def forward(self, x):
        # x shape: (batch_size, 53) → split scalar and board input
        scalar_input = x[:, :3]  # (batch_size, 3)
        board_input = x[:, 3:].reshape(-1, 2, 5, 5)  # (batch_size, 2, 5, 5)

        # CNN forward
        x_board = self.relu1(self.conv1(board_input))
        x_board = self.relu2(self.conv2(x_board))
        x_board = self.relu3(self.conv3(x_board))
        x_board = x_board.view(x_board.size(0), -1)  # Flatten

        # Combine CNN and scalar inputs
        x = torch.cat([scalar_input, x_board], dim=1)

        # Fully connected layers
        x = self.relu4(self.fc1(x))
        x = self.relu5(self.fc2(x))
        return self.out(x)

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path))


class DQN(nn.Module):
    def __init__(self, input_size=53, hidden_size=1024, output_size=26):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # x shape: (batch_size, 51)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.out(x)

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path))


class QLearningAgent:
    def __init__(
        self, epsilon=1.0, decay=0.99999, alpha=0.3, gamma=0.9, buffer_capacity=5000
    ):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.decay = decay
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        # self.model = DQN().to(device)
        self.model = DQNCNN().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

    def get_input(
        self, move_count, previous_board, board, piece_type, exploit, only_legal_moves
    ):
        if not exploit and np.random.random() < self.epsilon:
            if only_legal_moves:
                possible_valid_actions = list(
                    r * 5 + c for r, c in np.argwhere(board == 0)
                )
                possible_valid_actions.append(25)
                return possible_valid_actions[
                    np.random.randint(len(possible_valid_actions))
                ]
            else:
                return np.random.randint(26)
        else:
            state = np.concatenate(
                (
                    np.array([piece_type % 2, move_count, 0]),
                    np.array(previous_board).flatten(),
                    np.array(board).flatten(),
                ),
                dtype=np.int8,
            )

            state_tensor = torch.tensor(state.astype(np.float32), device=device)

            return self.model(state_tensor.unsqueeze(0)).argmax().item()

    def learn(self, batch_size):
        state, action, reward, next_state = self.buffer.sample(batch_size)

        state = torch.tensor(state, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.int64).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)

        # Get predicted Q-values for current states
        q_values = self.model(state)
        q_value = q_values.gather(1, action.view(-1, 1)).squeeze(1)

        # Calculate expected Q-values
        with torch.no_grad():
            next_q_values = self.model(next_state)
            next_q_value = next_q_values.max(1)[0]

            done = next_state[:, 2] == 1  # Assuming index 2 indicates terminal state

            # Initialize expected Q-values as rewards
            expected_q_value = torch.where(
                done, reward, reward + self.gamma * next_q_value
            )

        # Compute MSE loss and update the model
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(0.1, self.epsilon * self.decay)

    def save_agent(self, path: str) -> None:
        self.model.save_model(path)

    def load_agent(self, path: str, train: bool) -> None:
        self.model.load_model(path)

        if not train:
            self.model.eval()
