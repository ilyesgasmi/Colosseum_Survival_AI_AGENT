# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import random 
@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    # Opposite Directions
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    preprocessed = False
    chess_board = []
    board_size = 0
    max_step = 0
    choice = tuple()
    p0 = "student"
    p1 = "random"
    results_cache = []
    display_save = False
    turn = 0
    scoreMax = -1000
    moveMax = tuple()
    max_depth = 0
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True
        
    

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        if(not self.preprocessed):
            self.preprocessed = True
            self.chess_board = chess_board
            self.state = chess_board
            self.board_size = len(chess_board)
            self.me_pos = my_pos
            self.adv_pos = adv_pos
            self.max_step = max_step
            if(self.board_size>6):
                self.max_depth = 2
            if(self.board_size<=6):
                self.max_depth = 3

            sys.setrecursionlimit(10000)
        #self.minimax(chess_board,my_pos, adv_pos, 1,0)
        self.alphaBetaMax(chess_board,my_pos,adv_pos,-1000,1000,0)
        return self.choice

    """
    def minimax(self,board, p0_pos, p1_pos, turn,depth):
        #print(depth)
        if(self.check_endgame(board, p0_pos,p1_pos)):
            if(not turn):
                temp = p0_pos
                p0_pos = p1_pos
                p1_pos = temp
            #print(self.stateScore(board, p0_pos, p1_pos))
            #self.ui_engine = UIEngine(self.board_size, self)
            #self.ui_engine.render(board, p0_pos, p1_pos, debug=False)
            score = self.stateScore(board, p0_pos, p1_pos)
            #print("score: ", score)
            return score

            
        scores = []
        scoreMoves = []
        allMoves = self.allPossibleMoves(board, p0_pos, p1_pos)
        random.shuffle(allMoves)
        for m in allMoves:
            b, p0, p1 = self.makeMove(m,deepcopy(board), p0_pos, p1_pos)
            scores.append(self.minimax(b,p1,p0,1-turn, depth+1))
            scoreMoves.append(m)
        
        if(turn == 1):
            max_index = scores.index(max(scores))
            self.choice = scoreMoves[max_index]
            self.scoreMax = scores
            self.moveMax = scoreMoves
            return scores[max_index]
        if(turn == 0):
            min_index = scores.index(min(scores))
            return scores[min_index]
        return "error"
    """
    def alphaBetaMax(self, board, p0_pos, p1_pos, alpha, beta, depth):
        if(self.check_endgame(board, p0_pos,p1_pos)):
            score = self.stateScore(board, p0_pos, p1_pos)
            return score
        if (depth == self.max_depth):
            return 0
        allMoves = self.allPossibleMoves(board, p0_pos, p1_pos)
        random.shuffle(allMoves)
        for m in allMoves:
            b, p0, p1 = self.makeMove(m,deepcopy(board), p0_pos, p1_pos)
            score = self.alphaBetaMin(b, p0, p1, alpha, beta, depth+1)
            if(score>=beta):
                return beta
            if(score>alpha):
                alpha=score
                if(depth==0):
                    self.choice = m
        return alpha
    def alphaBetaMin(self, board, p0_pos, p1_pos, alpha, beta, depth):
        if(self.check_endgame(board, p1_pos,p1_pos)):
            score = -self.stateScore(board, p0_pos, p1_pos)
            return score
        if (depth == self.max_depth):
            return 0
        allMoves = self.allPossibleMoves(board, p0_pos, p1_pos)
        random.shuffle(allMoves)
        for m in allMoves:
            b, p0, p1 = self.makeMove(m,deepcopy(board), p0_pos, p1_pos)
            score = self.alphaBetaMax(b, p0, p1, alpha, beta, depth+1)
            if(score<=alpha):
                return alpha
            if(score<beta):
                beta=score
                if(depth==0):
                    self.choice = m
        return beta
        
    def check_endgame(self, board, p0_pos, p1_pos):
        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        if p0_r == p1_r:
            return False
        #self.ui_engine = UIEngine(self.board_size, self)
        #self.ui_engine.render(board, p0_pos, p1_pos, debug=False)
        return True

    def allPossibleMoves(self, curr_board, my_pos, adv_pos):

        # BFS
        state_queue = [(my_pos, 0)]
        visited = {tuple(my_pos)}
        moves = []
        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if curr_board[r, c, dir]:
                    continue
                next_pos = tuple(np.add(cur_pos,move))
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                x,y = next_pos
                for d in self.dir_map:
                    if(not curr_board[x,y,self.dir_map[d]]):
                        moves.append(tuple((next_pos, self.dir_map[d])))
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        if(not moves):
            next_pos = my_pos
            x,y = next_pos
            for d in self.dir_map:
                if(not curr_board[x,y,self.dir_map[d]]):
                    moves.append(tuple((next_pos, self.dir_map[d])))
                    
        return moves

    def makeMove(self, current_player_move, currBoard, p0_pos, p1_pos):
        next_pos, dir = current_player_move

        # Set the barrier to True
        r, c = next_pos
        currBoard[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        currBoard[r + move[0], c + move[1], self.opposites[dir]] = True

        return (currBoard,p0_pos,p1_pos)


    def stateScore(self, state, me_pos, adv_pos):
        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if state[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        me_r = find(tuple(me_pos))
        adv_r = find(tuple(adv_pos))
        me_score = list(father.values()).count(me_r)
        adv_score = list(father.values()).count(adv_r)
        #print("score: ", me_score - adv_score)
        return me_score - adv_score
