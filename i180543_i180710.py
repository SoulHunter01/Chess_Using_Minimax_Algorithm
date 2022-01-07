import numpy as np
from PIL import Image # for images
from tkinter import * # to create a new window
from PIL import ImageTk # for images in a new window

class Movement:
    def __init__(self, initial_x, initial_y, final_x, final_y, castling_move):
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.final_x = final_x
        self.final_y = final_y
        self.castling_move = castling_move

    def isSame(self, move2):
        return self.initial_x == move2.initial_x and self.initial_y == move2.initial_y and self.final_x == move2.final_x and self.final_y == move2.final_y

    def printMove(self):
        return "Moving from: ("+str(self.initial_x)+","+str(self.initial_y)+") to: ("+str(self.final_x)+","+str(self.final_y)+")"

class Piece:
    WHITE = "W"
    BLACK = "B"

    def __init__(self, pos_x, pos_y, color, p_type, point):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.color = color
        self.p_type = p_type
        self.point = point

    def removeEmpty(self, moves):
        val_to_be_removed = 0
        try:
            while True:
                moves.remove(val_to_be_removed)
        except ValueError:
            pass

        return moves

    def getMove(self, board, final_x, final_y):
        move = 0

        if board.isIncluded(final_x, final_y):
            piece = board.getPiece(final_x, final_y)

            if piece == 0:
                move = Movement(self.pos_x, self.pos_y, final_x, final_y, False)
            else:
                if piece.color != self.color:
                    move = Movement(self.pos_x, self.pos_y, final_x, final_y, False)

        return move

    def getAxisMoves(self, board):
        moves = []

        for i in range(1, self.pos_x + 1):
            piece = board.getPiece(self.pos_x - i, self.pos_y)
            moves.append(self.getMove(board, self.pos_x - i, self.pos_y))

            if piece != 0:
                break

        for i in range(1, 8 - self.pos_x):
            piece = board.getPiece(self.pos_x + i, self.pos_y)
            moves.append(self.getMove(board, self.pos_x + i, self.pos_y))

            if piece != 0:
                break

        for i in range(1, 8 - self.pos_y):
            piece = board.getPiece(self.pos_x, self.pos_y + i)
            moves.append(self.getMove(board, self.pos_x, self.pos_y+i))

            if piece != 0:
                break

        for i in range(1, self.pos_y + 1):
            piece = board.getPiece(self.pos_x, self.pos_y - i)
            moves.append(self.getMove(board, self.pos_x, self.pos_y-i))

            if piece != 0:
                break

        return self.removeEmpty(moves)

    def getDiagonalMoves(self, board):
        moves = []

        for i in range(1, Board.SIZE):
            if not board.isIncluded(self.pos_x + i, self.pos_y+i):
                break

            piece = board.getPiece(self.pos_x + i, self.pos_y + i)
            moves.append(self.getMove(board, self.pos_x+i, self.pos_y+i))

            if piece != 0:
                break

        for i in range(1, Board.SIZE):
            if not board.isIncluded(self.pos_x + i, self.pos_y - i):
                break

            piece = board.getPiece(self.pos_x + i, self.pos_y - i)
            moves.append(self.getMove(board, self.pos_x + i, self.pos_y - i))

            if piece != 0:
                break

        for i in range(1, Board.SIZE):
            if not board.isIncluded(self.pos_x - i, self.pos_y + i):
                break

            piece = board.getPiece(self.pos_x - i, self.pos_y + i)
            moves.append(self.getMove(board, self.pos_x - i, self.pos_y + i))

            if piece != 0:
                break

        for i in range(1, Board.SIZE):
            if not board.isIncluded(self.pos_x - i, self.pos_y - i):
                break

            piece = board.getPiece(self.pos_x - i, self.pos_y - i)
            moves.append(self.getMove(board, self.pos_x - i, self.pos_y - i))

            if piece != 0:
                break

        return self.removeEmpty(moves)

    def pieceToUnicode(self):
        if self.color == "W" and self.p_type == "P":
            return "\u2659"
        elif self.color == "B" and self.p_type == "P":
            return "\u265F"
        elif self.color == "W" and self.p_type == "R":
            return "\u2656"
        elif self.color == "B" and self.p_type == "R":
            return "\u265C"
        elif self.color == "W" and self.p_type == "N":
            return "\u2658"
        elif self.color == "B" and self.p_type == "N":
            return "\u265E"
        elif self.color == "W" and self.p_type == "B":
            return "\u2657"
        elif self.color == "B" and self.p_type == "B":
            return "\u265D"
        elif self.color == "W" and self.p_type == "Q":
            return "\u2655"
        elif self.color == "B" and self.p_type == "Q":
            return "\u265B"
        elif self.color == "W" and self.p_type == "K":
            return "\u2654"
        elif self.color == "B" and self.p_type == "K":
            return "\u265A"
        else:
            print("Illegal Piece!")

class Pawn(Piece):
    P_TYPE = "P"
    POINT = 1

    def __init__(self, pos_x, pos_y, color):
        super(Pawn, self).__init__(pos_x, pos_y, color, Pawn.P_TYPE, Pawn.POINT)

    def isFirstMove(self):
        if self.color == Piece.WHITE:
            return self.pos_y == Board.SIZE - 2
        else:
            return self.pos_y == 1

    def getPossibleMoves(self, board):
        moves = []

        direction = -1

        if self.color == Piece.BLACK:
            direction = 1

        if board.getPiece(self.pos_x, self.pos_y + direction) == 0:
            moves.append(self.getMove(board, self.pos_x, self.pos_y + direction))

        if self.isFirstMove() and board.getPiece(self.pos_x, self.pos_y + direction) == 0 and board.getPiece(self.pos_x, self.pos_y + direction*2) == 0:
            moves.append(self.getMove(board, self.pos_x, self.pos_y + direction * 2))

        piece = board.getPiece(self.pos_x + 1, self.pos_y + direction)

        if piece != 0:
            moves.append(self.getMove(board, self.pos_x + 1, self.pos_y + direction))

        piece = board.getPiece(self.pos_x - 1, self.pos_y + direction)

        if piece != 0:
            moves.append(self.getMove(board, self.pos_x - 1, self.pos_y + direction))

        return self.removeEmpty(moves)

    def clone(self):
        return Pawn(self.pos_x, self.pos_y, self.color)

class Rook(Piece):
    P_TYPE = "R"
    POINT = 5

    def __init__(self, pos_x, pos_y, color):
        super(Rook, self).__init__(pos_x, pos_y, color, Rook.P_TYPE, Rook.POINT)

    def getPossibleMoves(self, board):
        return self.getAxisMoves(board)

    def clone(self):
        return Rook(self.pos_x, self.pos_y, self.color)


class Knight(Piece):
    P_TYPE = "N"
    POINT = 3

    def __init__(self, pos_x, pos_y, color):
        super(Knight, self).__init__(pos_x, pos_y, color, Knight.P_TYPE, Knight.POINT)

    def getPossibleMoves(self, board):
        moves = []

        moves.append(self.getMove(board, self.pos_x + 2, self.pos_y + 1))
        moves.append(self.getMove(board, self.pos_x - 1, self.pos_y + 2))
        moves.append(self.getMove(board, self.pos_x - 2, self.pos_y + 1))
        moves.append(self.getMove(board, self.pos_x + 1, self.pos_y - 2))
        moves.append(self.getMove(board, self.pos_x + 2, self.pos_y - 1))
        moves.append(self.getMove(board, self.pos_x + 1, self.pos_y + 2))
        moves.append(self.getMove(board, self.pos_x - 2, self.pos_y - 1))
        moves.append(self.getMove(board, self.pos_x - 1, self.pos_y - 2))

        return self.removeEmpty(moves)

    def clone(self):
        return Knight(self.pos_x, self.pos_y, self.color)

class Bishop(Piece):
    P_TYPE = "B"
    POINT = 3

    def __init__(self, pos_x, pos_y, color):
        super(Bishop, self).__init__(pos_x, pos_y, color, Bishop.P_TYPE, Bishop.POINT)

    def getPossibleMoves(self, board):
        return self.getDiagonalMoves(board)

    def clone(self):
        return Bishop(self.pos_x, self.pos_y, self.color)

class Queen(Piece):
    P_TYPE = "Q"
    POINT = 9

    def __init__(self, pos_x, pos_y, color):
        super(Queen, self).__init__(pos_x, pos_y, color, Queen.P_TYPE, Queen.POINT)

    def getPossibleMoves(self, board):
        diagonal = self.getDiagonalMoves(board)
        axis = self.getAxisMoves(board)
        return axis + diagonal

    def clone(self):
        return Queen(self.pos_x, self.pos_y, self.color)

class King(Piece):
    P_TYPE = "K"
    POINT = 2000000

    def __init__(self, pos_x, pos_y, color):
        super(King, self).__init__(pos_x, pos_y, color, King.P_TYPE, King.POINT)

    def getTopCastlingMove(self, board):
        if self.color == Piece.WHITE and board.white_king_moved:
            return 0
        if self.color == Piece.BLACK and board.black_king_moved:
            return 0

        piece = board.getPiece(self.pos_x, self.pos_y - 3)

        if piece != 0:
            if piece.color == self.color and piece.p_type == Rook.P_TYPE:
                if board.getPiece(self.pos_x, self.pos_y - 1) == 0 and board.getPiece(self.pos_x, self.pos_y - 2) == 0:
                    return Movement(self.pos_x, self.pos_y, self.pos_x, self.pos_y - 2, True)

        return 0

    def getBottomCastlingMove(self, board):
        if self.color == Piece.WHITE and board.white_king_moved:
            return 0
        if self.color == Piece.BLACK and board.black_king_moved:
            return 0

        piece = board.getPiece(self.pos_x, self.pos_y + 4)

        if piece != 0:
            if piece.color == self.color and piece.p_type == Rook.P_TYPE:
                if board.getPiece(self.pos_x, self.pos_y + 1) == 0 and board.getPiece(self.pos_x, self.pos_y + 2) == 0 and board.getPiece(self.pos_x, self.pos_y + 3) == 0:
                    return Movement(self.pos_x, self.pos_y, self.pos_x, self.pos_y + 2, True)

        return 0

    def getPossibleMoves(self, board):
        moves = []

        moves.append(self.getMove(board, self.pos_x + 1, self.pos_y))
        moves.append(self.getMove(board, self.pos_x + 1, self.pos_y + 1))
        moves.append(self.getMove(board, self.pos_x, self.pos_y + 1))
        moves.append(self.getMove(board, self.pos_x - 1, self.pos_y + 1))
        moves.append(self.getMove(board, self.pos_x - 1, self.pos_y))
        moves.append(self.getMove(board, self.pos_x - 1, self.pos_y - 1))
        moves.append(self.getMove(board, self.pos_x, self.pos_y - 1))
        moves.append(self.getMove(board, self.pos_x + 1, self.pos_y - 1))
        moves.append(self.getTopCastlingMove(board))
        moves.append(self.getBottomCastlingMove(board))

        return self.removeEmpty(moves)

    def clone(self):
        return King(self.pos_x, self.pos_y, self.color)

class Board:
    SIZE = 8

    def __init__(self, chesspieces, white_king_moved, black_king_moved):
        self.chesspieces = chesspieces
        self.white_king_moved = white_king_moved
        self.black_king_moved = black_king_moved

    @classmethod
    def clone(cls, chessboard):
        chesspieces = [[0 for i in range(Board.SIZE)] for j in range(Board.SIZE)]
        for i in range(Board.SIZE):
            for j in range(Board.SIZE):
                piece = chessboard.chesspieces[i][j]

                if piece != 0:
                    chesspieces[i][j] = piece.clone()
        return cls(chesspieces, chessboard.white_king_moved, chessboard.black_king_moved)

    @classmethod
    def createBoard(cls):
        chess_pieces = [[0 for i in range(Board.SIZE)] for j in range(Board.SIZE)]

        # PAWNS
        for i in range(Board.SIZE):
            chess_pieces[i][Board.SIZE - 2] = Pawn(i, Board.SIZE - 2, Piece.WHITE)
            chess_pieces[i][1] = Pawn(i, 1, Piece.BLACK)

        # ROOKS
        chess_pieces[0][Board.SIZE - 1] = Rook(0, Board.SIZE - 1, Piece.WHITE)
        chess_pieces[0][0] = Rook(0, 0, Piece.BLACK)
        chess_pieces[Board.SIZE - 1][Board.SIZE - 1] = Rook(Board.SIZE - 1, Board.SIZE - 1, Piece.WHITE)
        chess_pieces[Board.SIZE - 1][0] = Rook(Board.SIZE - 1, 0, Piece.BLACK)

        # KNIGHTS
        chess_pieces[1][Board.SIZE - 1] = Knight(1, Board.SIZE - 1, Piece.WHITE)
        chess_pieces[1][0] = Knight(1, 0, Piece.BLACK)
        chess_pieces[Board.SIZE - 2][Board.SIZE - 1] = Knight(Board.SIZE - 2, Board.SIZE - 1, Piece.WHITE)
        chess_pieces[Board.SIZE - 2][0] = Knight(Board.SIZE - 2, 0, Piece.BLACK)

        # BISHOPS
        chess_pieces[2][Board.SIZE - 1] = Bishop(2, Board.SIZE - 1, Piece.WHITE)
        chess_pieces[2][0] = Bishop(2, 0, Piece.BLACK)
        chess_pieces[Board.SIZE - 3][Board.SIZE - 1] = Bishop(Board.SIZE - 3, Board.SIZE - 1, Piece.WHITE)
        chess_pieces[Board.SIZE - 3][0] = Bishop(Board.SIZE - 3, 0, Piece.BLACK)

        # KINGS
        chess_pieces[4][Board.SIZE - 1] = King(4, Board.SIZE - 1, Piece.WHITE)
        chess_pieces[4][0] = King(4, 0, Piece.BLACK)

        # QUEENS
        chess_pieces[3][0] = Queen(3, 0, Piece.BLACK)
        chess_pieces[3][Board.SIZE - 1] = Queen(3, Board.SIZE - 1, Piece.WHITE)

        return cls(chess_pieces, False, False)

    def getPossibleMoves(self, color):
        moves = []
        for i in range(Board.SIZE):
            for j in range(Board.SIZE):
                piece = self.chesspieces[i][j]

                if piece != 0:
                    if piece.color == color:
                        moves += piece.getPossibleMoves(self)

        return moves

    def performMove(self, move):
        piece = self.chesspieces[move.initial_x][move.initial_y]
        piece.pos_x = move.final_x
        piece.pos_y = move.final_y
        self.chesspieces[move.final_x][move.final_y] = piece
        self.chesspieces[move.initial_x][move.initial_y] = 0

        if piece.p_type == Pawn.P_TYPE:
            if piece.pos_y == 0 or piece.pos_y == Board.SIZE-1:
                self.chesspieces[piece.pos_x][piece.pos_y] = Queen(piece.pos_x, piece.pos_y, piece.color)

        if move.castling_move:
            if move.final_x < move.initial_x:
                rook = self.chesspieces[move.initial_x][0]
                rook.pos_x = 2
                self.chesspieces[2][0] = rook
                self.chesspieces[0][0] = 0

            if move.final_x > move.initial_x:
                rook = self.chesspieces[move.initial_x][Board.SIZE-1]
                rook.pos_x = Board.SIZE-4
                self.chesspieces[Board.SIZE-4][Board.SIZE-1] = rook
                self.chesspieces[move.initial_x][Board.SIZE-1] = 0

        if piece.p_type == King.P_TYPE:
            if piece.color == Piece.WHITE:
                self.white_king_moved = True
            else:
                self.black_king_moved = True

    def player(self, color):
        color2 = Piece.WHITE

        if color == Piece.WHITE:
            color2 = Piece.BLACK

        for move in self.getPossibleMoves(color2):
            cloned = Board.clone(self)
            cloned.performMove(move)

            is_king = False

            for i in range(Board.SIZE):
                for j in range(Board.SIZE):
                    piece = cloned.chesspieces[i][j]

                    if piece != 0:
                        if piece.color == color and piece.p_type == King.P_TYPE:
                            is_king = True

            if not is_king:
                return True

        return False

    def getPiece(self, pos_x, pos_y):
        if not self.isIncluded(pos_x, pos_y):
            return 0

        return self.chesspieces[pos_x][pos_y]

    def isIncluded(self, pos_x, pos_y):
        a = (pos_x >= 0)
        b = (pos_y >= 0)
        c = (pos_x < Board.SIZE)
        d = (pos_y < Board.SIZE)

        return a and b and c and d

    def printChess(self):
        print("    A    B    C    D    E    F    G    H")
        print("    ------------------------------------")
        for i in range(Board.SIZE):
            print(str(8 - i)+" | ", end=' ')

            for j in range(Board.SIZE):
                piece = self.chesspieces[j][i]

                if piece != 0:
                    print(str(piece.pieceToUnicode()).ljust(3), end=' ')
                else:
                    if j == 2:
                        print("", end=' ')
                    print("\u2610  ".ljust(2), end=' ')
            print()

class Heuristics:
    PAWNS = np.array([
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 5, 10, 10, -20, -20, 10, 10,  5],
        [ 5, -5, -10,  0,  0, -10, -5,  5],
        [ 0,  0,  0, 20, 20,  0,  0,  0],
        [ 5,  5, 10, 25, 25, 10,  5,  5],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [ 0,  0,  0,  0,  0,  0,  0,  0]
    ])

    KNIGHTS = np.array([
        [-50, -40, -30, -30, -30, -30, -40, -50],
        [-40, -20,   0,   5,   5,   0, -20, -40],
        [-30,   5,  10,  15,  15,  10,   5, -30],
        [-30,   0,  15,  20,  20,  15,   0, -30],
        [-30,   5,  15,  20,  20,  15,   0, -30],
        [-30,   0,  10,  15,  15,  10,   0, -30],
        [-40, -20,   0,   0,   0,   0, -20, -40],
        [-50, -40, -30, -30, -30, -30, -40, -50]
    ])

    BISHOPS = np.array([
        [-20, -10, -10, -10, -10, -10, -10, -20],
        [-10,   5,   0,   0,   0,   0,   5, -10],
        [-10,  10,  10,  10,  10,  10,  10, -10],
        [-10,   0,  10,  10,  10,  10,   0, -10],
        [-10,   5,   5,  10,  10,   5,   5, -10],
        [-10,   0,   5,  10,  10,   5,   0, -10],
        [-10,   0,   0,   0,   0,   0,   0, -10],
        [-20, -10, -10, -10, -10, -10, -10, -20]
    ])

    ROOKS = np.array([
        [ 0,  0,  0,  5,  5,  0,  0,  0],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [ 5, 10, 10, 10, 10, 10, 10,  5],
        [ 0,  0,  0,  0,  0,  0,  0,  0]
    ])

    QUEENS = np.array([
        [-20, -10, -10, -5, -5, -10, -10, -20],
        [-10,   0,   5,  0,  0,   0,   0, -10],
        [-10,   5,   5,  5,  5,   5,   0, -10],
        [0,   0,   5,  5,  5,   5,   0,  -5],
        [-5,   0,   5,  5,  5,   5,   0,  -5],
        [-10,   0,   5,  5,  5,   5,   0, -10],
        [-10,   0,   0,  0,  0,   0,   0, -10],
        [-20, -10, -10, -5, -5, -10, -10, -20]
    ])

    def getPosScore(board, p_type, table):
        white = 0
        black = 0
        for i in range(8):
            for j in range(8):
                piece = board.chesspieces[i][j]

                if piece != 0:
                    if piece.p_type == p_type:
                        if piece.color == Piece.WHITE:
                            white += table[i][j]
                        else:
                            black += table[7 - i][j]

        return white - black

    def getPieceScore(board):
        white = 0
        black = 0
        for i in range(8):
            for j in range(8):
                piece = board.chesspieces[i][j]

                if piece != 0:
                    if piece.color == Piece.WHITE:
                        white += piece.point
                    else:
                        black += piece.point

        return white - black

    def evaluateHeuristics(board):
        material = Heuristics.getPieceScore(board)

        pawns = Heuristics.getPosScore(board, Pawn.P_TYPE, Heuristics.PAWNS)
        knights = Heuristics.getPosScore(board, Knight.P_TYPE, Heuristics.KNIGHTS)
        bishops = Heuristics.getPosScore(board, Bishop.P_TYPE, Heuristics.BISHOPS)
        rooks = Heuristics.getPosScore(board, Rook.P_TYPE, Heuristics.ROOKS)
        queens = Heuristics.getPosScore(board, Queen.P_TYPE, Heuristics.QUEENS)

        total = material + pawns + knights + bishops + rooks + queens

        return total

class AI:
    INFINITY = 10000000

    def isMoveIllegal(move, illegal_moves):
        for illegal in illegal_moves:
            if illegal.isSame(move):
                return True
        return False

    def alphaBetaPruning(chessboard, depth, a, b, maxed):
        if depth == 0:
            return Heuristics.evaluateHeuristics(chessboard)

        if maxed:
            best_score = -AI.INFINITY

            for move in chessboard.getPossibleMoves(Piece.WHITE):
                cloned = Board.clone(chessboard)
                cloned.performMove(move)

                best_score = max(best_score, AI.alphaBetaPruning(cloned, depth - 1, a, b, False))
                a = max(a, best_score)

                if a >= b:
                    break
            return best_score
        else:
            best_score = AI.INFINITY

            for move in chessboard.getPossibleMoves(Piece.BLACK):
                cloned = Board.clone(chessboard)
                cloned.performMove(move)

                best_score = min(best_score, AI.alphaBetaPruning(cloned, depth - 1, a, b, True))
                b = min(b, best_score)

                if b <= a:
                    break
            return best_score

    def getAIMove(chessboard, illegal_moves):
        best_move = 0
        best_score = AI.INFINITY

        for move in chessboard.getPossibleMoves(Piece.BLACK):
            if AI.isMoveIllegal(move, illegal_moves):
                continue

            cloned = Board.clone(chessboard)
            cloned.performMove(move)

            score = AI.alphaBetaPruning(cloned, 2, -AI.INFINITY, AI.INFINITY, True)

            if score < best_score:
                best_score = score
                best_move = move

        if best_move == 0:
            return 0

        cloned = Board.clone(chessboard)
        cloned.performMove(best_move)

        if cloned.player(Piece.BLACK):
            illegal_moves.append(best_move)
            return AI.getAIMove(chessboard, illegal_moves)

        return best_move

def colToInt(alphabet):
    if alphabet == "A" or alphabet == "a":
        return 0
    elif alphabet == "B" or alphabet == "b":
        return 1
    elif alphabet == "C" or alphabet == "c":
        return 2
    elif alphabet == "D" or alphabet == "d":
        return 3
    elif alphabet == "E" or alphabet == "e":
        return 4
    elif alphabet == "F" or alphabet == "f":
        return 5
    elif alphabet == "G" or alphabet == "g":
        return 6
    elif alphabet == "H" or alphabet == "h":
        return 7
    else:
        raise ValueError("Illegal Move! Please Start again!")

def userMove():
    start = input("Enter Piece Position: (B5)\n")
    end = input("Enter Position to place the piece: (B6)\n")

    try:
        initial_x = colToInt(start[0])
        initial_y = 8 - int(start[1])
        final_x = colToInt(end[0])
        final_y = 8 - int(end[1])

        return Movement(initial_x, initial_y, final_x, final_y, False)
    except ValueError:
        print("Illegal Move!")
        return userMove()

def isMoveValid(board):
    while True:
        move = userMove()
        valid = False
        possible_moves = board.getPossibleMoves(Piece.WHITE)

        if not possible_moves:
            return 0

        for possible_move in possible_moves:
            if move.isSame(possible_move):
                move.castling_move = possible_move.castling_move
                valid = True
                break

        if not valid:
            print("Illegal move.")
        else:
            break
    return move

def main():
    board = Board.createBoard()
    board.printChess()

    while True:
        move = isMoveValid(board)

        if move == 0:
            if board.player(Piece.WHITE):
                print("Checkmate. Black Wins.")
                break
            else:
                print("GAME DRAW!")
                break

        board.performMove(move)

        print("User move: ")
        print(move.printMove())
        board.printChess()

        ai_move = AI.getAIMove(board, [])

        if ai_move == 0:
            if board.player(Piece.BLACK):
                print("Checkmate. White wins.")
                break
            else:
                print("GAME DRAW!")
                break

        board.performMove(ai_move)
        print("AI move: ")
        print(ai_move.printMove())
        board.printChess()

if __name__ == "__main__":
    main()