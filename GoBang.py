from tkinter import *
from Train import CNN
import random

Unit_Size = 50
Prob = 0.1


def gameOver(board, pos):
    row = pos[0]
    col = pos[1]
    color = board[row][col]
    # examine if 5 in a column
    num = 1
    up = down = True
    for i in range(1, 5):
        if chessed(board, row, col + i, color):
            num += 1
        else:
            up = False
        if chessed(board, row, col - i, color):
            num += 1
        else:
            down = False
        if not up and not down:
            break
    if num >= 5:
        return True

    # examine if 5 in a row
    num = 1
    up = down = True
    for i in range(1, 5):
        if chessed(board, row + i, col, color):
            num += 1
        else:
            up = False
        if chessed(board, row - i, col, color):
            num += 1
        else:
            down = False
        if not up and not down:
            break
    if num >= 5:
        return True

        # examine if 5 in north-eastern
    num = 1
    up = down = True
    for i in range(1, 5):
        if up and chessed(board, row - i, col - i, color):
            num += 1
        else:
            up = False
        if down and chessed(board, row + i, col + i, color):
            num += 1
        else:
            down = False
        if not up and not down:
            break
    if num >= 5:
        return True

    # examine if 5 in north-western
    num = 1
    up = down = True
    for i in range(1, 5):
        if up and chessed(board, row - i, col + i, color):
            num += 1
        else:
            up = False
        if down and chessed(board, row + i, col - i, color):
            num += 1
        else:
            down = False
        if not up and not down:
            break
    if num >= 5:
        return True

    return False


def chessed(board, row, col, color):
    if row >= len(board) or col >= len(board) or row < 0 or col < 0:
        return False
    elif board[row][col] == color:
        return True
    else:
        return False


def playerScore(player, board, pos):
    score = 0
    row = pos[0]
    col = pos[1]
    color = board[row][col]
    num = 1
    up = down = True
    for i in range(1, 5):
        if chessed(board, row, col + i, color):
            num += 1
        else:
            up = False
        if chessed(board, row, col - i, color):
            num += 1
        else:
            down = False
        if not up and not down:
            break
    if num >= 5:
        score += 10 ** num

    # examine row
    num = 1
    up = down = True
    for i in range(1, 5):
        if chessed(board, row + i, col, color):
            num += 1
        else:
            up = False
        if chessed(board, row - i, col, color):
            num += 1
        else:
            down = False
        if not up and not down:
            break
    if num >= 5:
        score += 10 ** num

    # examine north-eastern
    num = 1
    up = down = True
    for i in range(1, 5):
        if up and chessed(board, row - i, col - i, color):
            num += 1
        else:
            up = False
        if down and chessed(board, row + i, col + i, color):
            num += 1
        else:
            down = False
        if not up and not down:
            break
    if num >= 5:
        score += 10 ** num

    # examine north-western
    num = 1
    up = down = True
    for i in range(1, 5):
        if up and chessed(board, row - i, col + i, color):
            num += 1
        else:
            up = False
        if down and chessed(board, row + i, col - i, color):
            num += 1
        else:
            down = False
        if not up and not down:
            break
    if num >= 5:
        score += 10 ** num
    return score


class AI(object):
    def __init__(self, player, trained, cnn):
        self.trained = trained
        self.node = 0
        self.player = player
        self.opponent = 1 if player == 2 else 2
        self.cnn = cnn

    def initilaize(self, player, trained):
        self.trained = trained
        self.node = 0
        self.player = player
        self.opponent = 1 if player == 2 else 2

    def findNext(self):
        if self.trained == True:
            return self.cnn.prediction(self.board)
        else:
            unchessed = []
            for row in range(len(self.board)):
                for col in range(len(self.board)):
                    if self.board[row][col] == 0:
                        unchessed.append([row, col])
            return unchessed

    def getScore(self, board, point):
        player = board[point[0]][point[1]]
        opponent = 1 if player == 2 else 2
        playerValue = playerScore(player, board, point)
        enmyValue = playerScore(opponent, board, point)
        value = playerValue + enmyValue
        if board[point[0]][point[1]] == self.player:
            return value
        elif board[point[0]][point[1]] == self.opponent:
            return value * -1
        return 0

    # find next point that has highest value
    def search(self, deep):
        self.node = 0
        nextPoints = self.findNext()
        max = -30000000
        bestPoint = nextPoints[0]
        for point in nextPoints:
            self.board[point[0]][point[1]] = self.player
            self.unNum -= 1
            value = self.chanceNode(deep, point)
            if (max < value):
                max = value
                bestPoint = [point[0], point[1]]
            self.board[point[0]][point[1]] = 0.0
            self.unNum += 1
        return bestPoint

    def chanceNode(self, deep, point):
        max_pro = 0.0
        min_pro = 0.0
        if self.board[point[0]][point[1]] == self.player:
            max_pro = Prob
            min_pro = (1 - Prob)
        else:
            max_pro = (1 - Prob)
            min_pro = Prob
        self.node += 1
        if gameOver(self.board, point):
            return 1000000 if self.board[point[0]][point[1]] == self.player else -1000000
        if self.unNum == 0:
            return 0
        if deep == 0:
            return self.getScore(self.board, point)
        return max_pro * self.findMax(self.board, deep - 1) + min_pro * self.findMin(self.board, deep - 1)

    def findMin(self, board, deep):
        nextPoints = self.findNext()
        min = 30000000
        for point in nextPoints:
            board[point[0]][point[1]] = self.opponent
            self.unNum -= 1
            value = self.chanceNode(deep, point)
            if (min > value):
                min = value
            board[point[0]][point[1]] = 0.0
            self.unNum += 1
        return min

    def findMax(self, board, deep):
        nextPoints = self.findNext()
        max = -300000
        for point in nextPoints:
            board[point[0]][point[1]] = self.player
            self.unNum -= 1
            value = self.chanceNode(deep, point)
            if (max < value):
                max = value
            board[point[0]][point[1]] = 0.0
            self.unNum += 1
        return max

    def bestStep(self, board, unchessed):
        self.board = board
        self.unNum = len(unchessed)
        best_point = self.search(1)
        return best_point, self.node


class GoBang():

    def __init__(self, size, cnn):
        self.size = size
        self.playerOne = 1  # AI player
        self.playerTwo = 2  # human player or baseline player
        self.player = self.playerOne
        self.nodes = 0
        self.ai = AI(self.playerOne, 1, cnn)
        self.winner = 0

    def Interface(self):
        # UI interface
        self.interface = Tk()
        self.interface.title("Gobang 2.0")

        # Initialize chess board
        self.board = [[0.0 for i in range(self.size)] for i in range(self.size)]
        self.width = (self.size - 1) * Unit_Size + 20
        self.height = (self.size - 1) * Unit_Size + 20
        self.can = Canvas(self.interface, bg="grey", width=self.width, height=self.height)
        self.can.grid(row=0, column=0)

        # draw the board
        for i in range(self.size):
            # draw horizonlyy
            self.can.create_line((10, 10 + i * Unit_Size), (self.width - 10, 10 + i * Unit_Size), width=2)
            # draw vertically
            self.can.create_line((10 + i * Unit_Size, 10), (10 + i * Unit_Size, self.height - 10), width=2)

    def drawChess(self, pos):
        x = 10 + pos[1] * Unit_Size
        y = 10 + pos[0] * Unit_Size
        radius = 20
        if self.board[pos[0]][pos[1]] == self.playerOne:
            self.can.create_oval(x - radius, y - radius, x + radius, y + radius, outline="black", fill="black")
        else:
            self.can.create_oval(x - radius, y - radius, x + radius, y + radius, outline="white", fill="white")

    def humanPlay(self):
        # first playing with human
        self.Interface()
        initPos = self.size // 2
        self.board[initPos][initPos] = self.playerOne
        self.player = self.playerTwo
        #   pos = self.transferAxisToPos([initPos, initPos])
        self.unchessed = [i for i in range(self.size * self.size)]
        chessed = initPos * self.size + initPos
        self.unchessed.remove(chessed)
        self.drawChess([initPos, initPos])
        self.can.bind("<Button-1>", self.playEvent)
        self.interface.mainloop()

    def playEvent(self, event):
        if self.player == self.playerOne or self.winner != 0:
            return
        x = event.x
        y = event.y
        if 10 <= x <= self.width - 10 and 10 <= y <= self.height - 10:
            row = round((y - 10) / Unit_Size)
            col = round((x - 10) / Unit_Size)
            if self.board[row][col] == 0:
                self.board[row][col] = self.playerTwo
                self.drawChess([row, col])
                if gameOver(self.board, [row, col]):
                    print("You Win!")
                    self.winner = self.playerOne
                    return
                if random.random() > Prob:
                    #   self.player = self.playerOne
                    self.randomPlayer(self.playerOne)

    def randomPlayer(self, color):
        pos = self.aiTurn(color)
        self.drawChess(pos)
        if gameOver(self.board, pos):
            print("You Lose!")
            self.winner = self.playerOne
            return
        if random.random() > Prob:
            self.player = self.playerTwo
        else:
            self.player = self.playerOne
            self.randomPlayer(self.playerOne)

    def baselinePlay(self):
        winner = 0
        while True:
            print("problem size = %d, The %dth games" % (self.size, self.step))
            if (len(self.unchessed) == 0):
                break
            if self.player == self.playerOne:
                next = self.aiTurn(self.player)
            else:
                next = self.baseTurn(self.player)
            if gameOver(self.board, next):
                winner = self.player
                break
            else:
                if random.random() > Prob:
                    self.player = self.playerOne if self.player == self.playerTwo else self.playerTwo
        return winner, self.nodes

    def ExperiInit(self, size, trained, step):
        self.step = step
        self.nodes = 0
        self.ai.initilaize(self.playerOne, trained)
        self.size = size
        self.player = self.playerTwo
        self.board = [[0.0 for i in range(self.size)] for i in range(self.size)]
        center = size // 2
        self.board[center][center] = self.playerOne
        self.unchessed = [i for i in range(self.size * self.size)]
        self.unchessed.remove(center * self.size + center)

    def baseTurn(self, color):
        r = random.randint(0, len(self.unchessed) - 1)
        pos = self.unchessed[r]
        #   point = self.transferPosToAxis(pos)
        col = pos % self.size
        if pos < self.size:
            row = 0
        else:
            row = int((pos - col) / self.size)
        point = [row, col]
        self.board[point[0]][point[1]] = color
        self.unchessed.remove(pos)
        return point

    def aiTurn(self, color):
        point, nodes = self.ai.bestStep(self.board, self.unchessed)
        self.board[point[0]][point[1]] = color
        self.nodes += nodes
        pos = point[0] * self.size + point[1]
        self.unchessed.remove(pos)
        return point


if __name__ == "__main__":
    model = CNN()
    model.restore(".\model\model.ckpt")
    humanplay = GoBang(15, model)
    humanplay.humanPlay()
