#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
# Originated from the above!

# Sutton's textbook
# Chap 1 study code
# Tic-Tac-Toe
# Sooyung Byeon
# 08/17/2019

import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


# State on the board
# Whose chess pieces are on the board? where are they?
# Who is the winner?
# Is it end of the game?
class State:
    def __init__(self):
        # the board(data) is represented by an n * n array,
        # Chess piece on the board(data)
        # 1 represents a chessman of the player who moves first,
        # -1 represents a chessman of another player
        # 0 represents an empty position
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_val = None
        self.end = None

    # compute the hash value for one state, it's unique (try hand writing)
    # Hash values can be thought of as fingerprints for files.
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):
                self.hash_val = self.hash_val * 3 + i + 1
        return self.hash_val

    # check whether a player has won the game, or it's a tie
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # check row
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        # check columns
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))
        # check diagonals
        trace = 0
        reverse_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reverse_trace)

        # check the results and winner (in case of win or lose)
        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end

    # Displaying
    # @symbol: 1 or -1
    # put chessman symbol in position (i, j)
    def next_state(self, i, j, symbol):
        new_state = State()     # new class
        new_state.data = np.copy(self.data)     # copy the data
        new_state.data[i, j] = symbol
        return new_state

    # print the board
    def print_state(self):
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                elif self.data[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')


# This is crazy... so annoying and confusing. Maybe there is a neat explanation...
def get_all_states_impl(current_state, current_symbol, all_states):
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if current_state.data[i][j] == 0:   # If it is a empty space
                new_state = current_state.next_state(i, j, current_symbol)      # Put symbol at (i, j) position
                # Note that 'current_state' is not updated unless 'not is_end' is true
                # 'new_state' disappears as for loop reinitialized
                new_hash = new_state.hash()
                if new_hash not in all_states:      # all states: all possible board configurations
                    # If something is missing from all_states, adding it
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end)      # tagging the non-existent states using dict()
                    if not is_end:
                        get_all_states_impl(new_state, -current_symbol, all_states)     # After finishing 1, starting -1
                        # This part causes the loop!
                        # 'For loop' is reinitialized


def get_all_states():
     current_symbol = 1
     current_state = State()
     all_states = dict()
     all_states[current_state.hash()] = (current_state, current_state.is_end())
     get_all_states_impl(current_state, current_symbol, all_states)
     return all_states


# all possible board configurations
all_states = get_all_states()

# The first one [0 0 0][0 0 0][0 0 0]
# The second one [1 0 0][0 0 0][0 0 0]
# The third one [1 -1 0][0 0 0][0 0 0]
# The fourth one [1 -1 1][0 0 0][0 0 0]
# The fifth one [1 -1 1][-1 0 0][0 0 0]...

# #8 [1 -1 1][-1 1 -1][1 0 0] --> is_end is True
# #9 [1 -1 1][-1 1 -1][0 1 0]
# #10 [1 -1 1][-1 1 -1][-1 1 0]
# #11 [1 -1 1][-1 1 -1][-1 1 1] --> is_end is True
# #12 [1 -1 1][-1 1 -1][0 1 -1]


class Judger:
    # @player 1: the player who will move first, its chessman will be (1)
    # @player 2: another player with a chessman (-1)
    # Juder receives 'player class' as inputs. So 'self' includes 'judger, player'
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)      # It does not have a '='. How it works?
        self.p2.set_symbol(self.p2_symbol)      # Judger needs 'Player class'
        # symbols of p1 and p2 are determined at this part
        self.current_state = State()       # State() comes in here

    def reset(self):
        self.p1.reset()
        self.p2.reset()
        # How this works?
        # Judge class needs Player class!

    def alternate(self):
        while True:     # why this is necessary?
            yield self.p1       # why 'yield' not 'return'?
            yield self.p2

    # @print_state: if True, print each board during the game
    def play(self, print_state=False):
        alternator = self.alternate()       # What does this mean?
        self.reset()
        current_state = State()     # This not came from the outside... newly defined.
        self.p1.set_state(current_state)       # current_state is a new... this is not in the game!?
        self.p2.set_state(current_state)
        # So, maybe the upper part is initialization?
        if print_state:
            current_state.print_state()
        while True:     # you can get out of here only if 'is_end' is true
            player = next(alternator)       # The next() function returns the next item from the iterator.
            # Why we should 'alternate' the players?
            # --> it means whose turn is it now
            i, j, symbol = player.act()     # action will be chosen based on the value or random try (epsilon)
            next_state_hash = current_state.next_state(i, j, symbol).hash()
            # Is this 'current_state' is accumulating from the initial?
            current_state, is_end = all_states[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if print_state:
                current_state.print_state()
            if is_end:
                return current_state.winner
                # At the end of the play(game), judge announces the winner!


# AI Player
# This part should make me understand the formal 'Judge' class.
# This part is hard to understand. Especially 'act' and 'backup' are confusing. (8/21/2019)
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    # maybe these are learning parameters...
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()   # related to the VALUE
        self.step_size = step_size  # related to estimations
        self.epsilon = epsilon      # related to greedy
        self.states = []    # This is a 'State class'.
        self.greedy = []
        self.symbol = 0

    def reset(self):
        self.states = []
        self.greedy = []
        # only these two are reset

    def set_state(self, state):     # State() (state class) is equivalent with state in Player class
        self.states.append(state)
        self.greedy.append(True)
        # In case of states, the default is 0 (empty) and symbols are 1 or -1

    # ********************************************************************************************* #
    # ********************************************************************************************* #
    # ********************************************************************************************* #
    # This rule is too simple but it works!
    def set_symbol(self, symbol):
        self.symbol = symbol
        for hash_val in all_states:     # all_states is accessible everywhere?
            state, is_end = all_states[hash_val]        # access data using 'state.data'
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0    # if win with current hash value, 1
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5
        # Win: 1
        # lose: 0
        # Tie or keep going: 0.5
    # ********************************************************************************************* #
    # ********************************************************************************************* #
    # ********************************************************************************************* #

    # update value estimation
    # It is a VALUE, one of the most important things in RL
    # ********************************************************************************************* #
    # ********************************************************************************************* #
    # ********************************************************************************************* #
    def backup(self):       # Why is the name is 'backup'? "self.estimations" are updated
        states = [state.hash() for state in self.states]
        # It should be list of the hash values

        for i in reversed(range(len(states) - 1)):
            state = states[i]
            td_error = self.greedy[i] * (self.estimations[states[i + 1]] - self.estimations[state])
            # Is this kind of RL technique? Out of left field...
            # What is greedy at the first time? --> if not greedy, it does not count
            # if estimations of the future time is bigger than current one, td_error > 0
            self.estimations[state] += self.step_size * td_error
            # This part requires some documents... I do not know what is going on
            # note that 'estimations' is a dict()
            # Estimation is recalculate based on the td_error
            # td_error is bigger as estimations value increases sharply
            # as a result, estimation grows up faster!

            # Anyway, this is the key part of the RL, which I do not fully understand
            # I do not know what its name is 'backup' but it is important to adjust estimation values.
    # ********************************************************************************************* #
    # ********************************************************************************************* #
    # ********************************************************************************************* #


    # choose an action based on the state
    # ********************************************************************************************* #
    # ********************************************************************************************* #
    # ********************************************************************************************* #
    def act(self):
        state = self.states[-1]     # the last one
        next_states = []
        next_positions = []
        # These should be determined
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:       # If empty
                    next_positions.append([i, j])
                    next_states.append(state.next_state(i, j, self.symbol).hash())  # The number of possible next move

        # Some random processes... Is it exploration?
        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            # select next move randomly within the list(length) of the possible moves
            action.append(self.symbol)
            self.greedy[-1] = False     # What is the exact meaning of greedy?
            # greedy: choose the best value position
            # not greedy: try exploration based on the random move
            return action

        values = []
        for hash_val, pos in zip(next_states, next_positions):      # zip / tuple need to be clear
            values.append((self.estimations[hash_val], pos))
        # to select one of the actions of equal value at random due to Python's sort is stable
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)       # Sorting 'values' according to the 'values[0]' value?
        # in this case, hash_val, in a reversed order
        action = values[0][1]       # What is selected? and How? --> the highest value action(position). maybe
        action.append(self.symbol)  # takes the symbol at the best value position
        return action
    # ********************************************************************************************* #
    # ********************************************************************************************* #
    # ********************************************************************************************* #


    # Throughout the save & load functions, 'self.estimations' are the results of RL
    # Why use .bin file?
    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)

# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self, **kwargs):       # Keyword(?) argument
        self.symbol = None
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol

    def act(self):
        self.state.print_state()
        key = input("Input your position:")
        # data = self.keys.index(key)     # data should be within the 'keys', which is defined above
        # Can I have an exception handler for this part?
        # data = self.keys.index(key) if key in self.keys else print("Wrong input!")
        while key not in self.keys:
            print('Wrong input: out of input range.')
            self.state.print_state()
            key = input("Input your position:")
        data = self.keys.index(key)

        i = data // BOARD_COLS
        j = data % BOARD_COLS

        # Preventing overlap
        occupied = abs(self.state.data[i][j])
        if occupied:
            print("Overlapped. Try another position.")
            i, j, self.symbol = self.act()

        return i, j, self.symbol


def train(epochs, print_every_n=500):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)
    judger = Judger(player1, player2)
    player1_win = 0.0   # the total number of winning of each player
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)     # play (a game) > act (single move)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
        player1.backup()    # related to greedy
        player2.backup()
        judger.reset()      # estimations are not reset
    player1.save_policy()
    player2.save_policy()


def compete(turns):     # Test game? figure out who wins more
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()      # this resets only greedys and states
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))


# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# So we test whether the AI can guarantee at least a tie if it goes second.
def play():
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0)     # epsilon=0 means 'Play as a policy'
        judger = Judger(player1, player2)
        player2.load_policy()
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")


if __name__ == '__main__':
    train(int(1e5))
    compete(int(1e3))
    play()

