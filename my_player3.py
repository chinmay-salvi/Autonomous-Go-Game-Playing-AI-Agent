import copy
import os
from collections import deque
import numpy as np
import time


def read_input(n, path="input.txt"):
    with open(path, "r") as f:
        lines = f.readlines()
        piece_type = int(lines[0])
        previous_board = np.array(
            [list(map(int, line.strip())) for line in lines[1 : n + 1]], dtype=np.int8
        )
        board = np.array(
            [list(map(int, line.strip())) for line in lines[n + 1 : 2 * n + 1]],
            dtype=np.int8,
        )
        return piece_type, previous_board, board


def write_output(result, path="output.txt"):
    with open(path, "w") as f:
        f.write("PASS" if result == "PASS" else f"{result[0]},{result[1]}")


class GO:
    def __init__(self, n, piece_type, previous_board, board):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        self.n_move = 0
        self.verbose = False
        self.max_move = n * n - 1
        self.komi = 3
        self.neighbor_offsets = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
        self.previous_board = previous_board.copy()
        self.board = board.copy()
        self.died_pieces = np.argwhere(
            (self.previous_board == piece_type) & (self.board != piece_type)
        )

    def detect_neighbors(self, i, j):
        """Returns valid neighboring coordinates of (i, j)."""
        neighbors = self.neighbor_offsets + (i, j)
        return neighbors[
            (0 <= neighbors[:, 0])
            & (neighbors[:, 0] < self.size)
            & (0 <= neighbors[:, 1])
            & (neighbors[:, 1] < self.size)
        ]

    def count_liberty(self, i, j):
        """
        Counting liberties.

        :param i: Row index of the board.
        :param j: Column index of the board.
        :return: the number of unique liberties.
        """
        liberties = set()  # Store unique liberty positions
        visited = set()  # To track visited positions
        stack = [(i, j)]  # Start DFS from the given stone

        while stack:
            ai, aj = stack.pop()

            if (ai, aj) not in visited:
                visited.add((ai, aj))

                for ni, nj in self.detect_neighbors(ai, aj):
                    if self.board[ni, nj] == 0:  # Empty space = liberty
                        liberties.add((ni, nj))  # Unique liberties
                    elif self.board[ni, nj] == self.board[ai, aj]:  # Ally
                        stack.append((ni, nj))  # Add ally to stack for DFS

        return len(liberties)

    def get_all_groups(self, piece_type):
        """Finds all groups of `piece_type` using BFS and returns them."""
        visited = np.zeros((self.size, self.size), dtype=bool)
        groups = []

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == piece_type and not visited[i, j]:
                    # BFS to find entire group
                    queue = deque([(i, j)])
                    group = set()
                    has_liberty = False
                    visited[i, j] = True  # Mark before adding to queue

                    while queue:
                        x, y = queue.popleft()
                        group.add((x, y))

                        for ni, nj in self.detect_neighbors(x, y):
                            if self.board[ni, nj] == 0:
                                has_liberty = True
                            elif (
                                self.board[ni, nj] == piece_type and not visited[ni, nj]
                            ):
                                visited[ni, nj] = True  # Mark before adding to queue
                                queue.append((ni, nj))

                    groups.append((group, has_liberty))

        return groups

    def find_died_pieces(self, piece_type):
        """Finds all stones of `piece_type` that have no liberty."""
        return [
            stone
            for group, has_liberty in self.get_all_groups(piece_type)
            if not has_liberty
            for stone in group
        ]

    def remove_died_pieces(self, piece_type):
        """
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        """

        died_pieces = self.find_died_pieces(piece_type)
        for piece in died_pieces:
            self.board[piece[0]][piece[1]] = 0

        return died_pieces

    def validity_check_with_liberty_count(self, i, j, piece_type):
        """
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        """
        self.board[i, j] = piece_type
        liberty_count = self.count_liberty(i, j)

        if liberty_count:
            self.board[i, j] = 0
            return True, liberty_count

        # If not, remove the died pieces of opponent and check again
        test_go = copy.deepcopy(self)
        test_go.remove_died_pieces(3 - piece_type)
        self.board[i, j] = 0
        liberty_count = test_go.count_liberty(i, j)

        if not liberty_count:
            if self.verbose:
                print("Invalid placement. No liberty found in this position.")
            return False, liberty_count
        # Check special case: repeat placement causing the repeat board state (KO rule)
        elif len(self.died_pieces) and np.array_equal(
            self.previous_board, test_go.board
        ):
            if self.verbose:
                print("Invalid placement. A repeat move not permitted by the KO rule.")
            return False, liberty_count

        return True, liberty_count

    def get_scores(self):
        score1 = np.count_nonzero(self.board == 1)
        score2 = np.count_nonzero(self.board == 2) + self.komi
        return score1, score2

    def evaluate_board(self, piece_type):
        opponent = 3 - piece_type

        my_stones = 0
        opp_stones = 0
        my_liberties = 0
        opp_liberties = 0

        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == piece_type:
                    my_stones += 1
                elif self.board[i][j] == opponent:
                    opp_stones += 1
                elif self.board[i][j] == 0:
                    my_liberty = False
                    opp_liberty = False

                    for dx, dy in directions:
                        nx, ny = i + dx, j + dy
                        if 0 <= nx < self.size and 0 <= ny < self.size:
                            if self.board[nx][ny] == piece_type:
                                my_liberty = True
                            elif self.board[nx][ny] == opponent:
                                opp_liberty = True

                    my_liberties += my_liberty
                    opp_liberties += opp_liberty

        if piece_type == 1:
            opp_stones += self.komi
        else:
            my_stones += self.komi

        # my_score = my_stones + my_liberties - 2 * (opp_stones + opp_liberties)
        my_score = my_stones - opp_stones
        # opp_score = opp_stones + opp_liberties - 2 * (my_score + my_liberties)
        opp_score = opp_stones - my_stones

        return my_score, opp_score


class AlphaBetaMaxMinPlayer:
    def __init__(self):
        self.max_depth = 5
        self.time_limit = 9.6

    def get_possible_placements_with_liberties(self, go, piece_type):
        placements_with_liberties = []

        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == 0:
                    is_valid, liberty_count = go.validity_check_with_liberty_count(
                        i, j, piece_type
                    )
                    if is_valid:
                        placements_with_liberties.append((i, j, liberty_count))

        return placements_with_liberties

    def get_input(self, go, piece_type):
        start_time = time.time()

        cnt_1, cnt_2 = go.get_scores()

        if cnt_1 + cnt_2 - go.komi < 2:
            if os.path.exists("./move_count.txt"):
                os.remove("./move_count.txt")
            move_count = cnt_1 + cnt_2 - go.komi
        else:
            with open("./move_count.txt", "r") as f:
                move_count = int(f.readline())

        if cnt_1 + cnt_2 - 3 < 4 and go.board[go.size // 2][go.size // 2] == 0:
            move = (go.size // 2, go.size // 2)
        else:
            if cnt_1 + cnt_2 > 14:
                self.max_depth = 7

            move, _, _ = self.maximizer(
                go, piece_type, 0, float("-inf"), float("inf"), start_time, move_count
            )

        with open("./move_count.txt", "w") as f:
            f.write(str(move_count + 2))

        end_time = time.time()
        print(f"Time taken for move: {end_time - start_time:.4f} seconds")
        return move

    def capture_potential(self, go, move, piece_type):
        original_value = go.board[move[0], move[1]]
        go.board[move[0], move[1]] = piece_type
        captured = len(go.find_died_pieces(3 - piece_type))
        go.board[move[0], move[1]] = original_value
        return captured

    def maximizer(self, go, piece_type, depth, alpha, beta, start_time, move_count):
        if (
            (time.time() - start_time > self.time_limit)
            or depth >= self.max_depth
            or move_count == go.max_move
        ):
            my_score, opp_score = go.evaluate_board(piece_type)
            return None, my_score, opp_score

        possible_moves = self.get_possible_placements_with_liberties(go, piece_type)
        possible_moves = sorted(
            possible_moves,
            key=lambda placement: (
                -placement[2],
                -self.capture_potential(go, (placement[0], placement[1]), piece_type),
                abs(go.size // 2 - placement[0]) + abs(go.size // 2 - placement[1]),
            ),
        )

        best_move = "PASS"

        if np.array_equal(go.previous_board, go.board):
            max_my_score, min_opp_score = go.evaluate_board(piece_type)
        else:
            new_go = copy.deepcopy(go)
            new_go.previous_board = go.board
            _, min_opp_score, max_my_score = self.minimizer(
                new_go,
                3 - piece_type,
                depth + 1,
                alpha,
                beta,
                start_time,
                move_count + 1,
            )

        for move in possible_moves:
            new_go = copy.deepcopy(go)
            new_go.previous_board = go.board
            new_go.board[move[0]][move[1]] = piece_type
            new_go.died_pieces = new_go.remove_died_pieces(3 - piece_type)

            _, opp_score, my_score = self.minimizer(
                new_go,
                3 - piece_type,
                depth + 1,
                alpha,
                beta,
                start_time,
                move_count + 1,
            )

            if my_score > max_my_score or (
                my_score == max_my_score and opp_score < min_opp_score
            ):
                max_my_score = my_score
                min_opp_score = opp_score
                best_move = move

            if max_my_score >= beta:
                break

            alpha = max(alpha, max_my_score)

        return best_move, max_my_score, min_opp_score

    def minimizer(self, go, piece_type, depth, alpha, beta, start_time, move_count):
        if (
            (time.time() - start_time > self.time_limit)
            or depth >= self.max_depth
            or move_count == go.max_move
        ):
            my_score, opp_score = go.evaluate_board(piece_type)
            return None, my_score, opp_score

        possible_moves = self.get_possible_placements_with_liberties(go, piece_type)
        possible_moves = sorted(
            possible_moves,
            key=lambda placement: (
                -placement[2],
                -self.capture_potential(go, (placement[0], placement[1]), piece_type),
                abs(go.size // 2 - placement[0]) + abs(go.size // 2 - placement[1]),
            ),
        )

        best_move = "PASS"

        if np.array_equal(go.previous_board, go.board):
            max_my_score, min_opp_score = go.evaluate_board(piece_type)
        else:
            new_go = copy.deepcopy(go)
            new_go.previous_board = go.board
            _, min_opp_score, max_my_score = self.maximizer(
                new_go,
                3 - piece_type,
                depth + 1,
                alpha,
                beta,
                start_time,
                move_count + 1,
            )

        for move in possible_moves:
            new_go = copy.deepcopy(go)
            new_go.previous_board = go.board
            new_go.board[move[0]][move[1]] = piece_type
            new_go.died_pieces = new_go.remove_died_pieces(3 - piece_type)

            _, opp_score, my_score = self.maximizer(
                new_go,
                3 - piece_type,
                depth + 1,
                alpha,
                beta,
                start_time,
                move_count + 1,
            )

            if my_score > max_my_score or (
                my_score == max_my_score and opp_score < min_opp_score
            ):
                min_opp_score = opp_score
                max_my_score = my_score
                best_move = move

            if min_opp_score <= alpha:
                break

            beta = min(beta, min_opp_score)

        return best_move, max_my_score, min_opp_score


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = read_input(N)
    go_instance = GO(N, piece_type, previous_board, board)
    player = AlphaBetaMaxMinPlayer()
    action = player.get_input(go_instance, piece_type)
    write_output(action)
