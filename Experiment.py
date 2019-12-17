from GoBang import GoBang
from Train import CNN
import matplotlib.pyplot as plt

probSize = [5, 7, 9, 11, 13]
completeGames = 25
modelPath = ".\model\model.ckpt"
trained = False

if __name__ == '__main__':
    trainedWin = []
    trainedNodes = []
    untrainedWin = []
    untrainedNodes = []
    cnn = CNN()
    cnn.restore(modelPath)
    gobang = GoBang(probSize[0], cnn)

    # Trained model
    baseWin = aiWin = node = 0
    for i in range(len(probSize)):
        node = 0
        aiWin = 0
        baseWin = 0
        for j in range(completeGames):
            gobang.ExperiInit(probSize[i], False, j)
            winner, nodes = gobang.baselinePlay()
            node += nodes
            if winner == 1:
                aiWin += 1
            else:
                baseWin += 1
        trainedWin.append(aiWin)
        trainedNodes.append(node)
    # Untrained Model
    aiWin = node = baseWin = 0
    for i in range(len(probSize)):
        baseWin = 0
        aiWin = 0
        node = 0
        for j in range(completeGames):
            gobang.ExperiInit(probSize[i], True, j)
            winner, nodes = gobang.baselinePlay()
            node += nodes
            if winner == 1:
                aiWin += 1
            else:
                baseWin += 1
        untrainedWin.append(aiWin)
        untrainedNodes.append(node)

    plt.figure("Win Rate")
    plt.title("Win Rate")
    plt.xlabel("Problem Size")
    plt.ylabel("Win Times")
    plt.bar(probSize, trainedWin, label="trained")
    plt.bar(probSize, untrainedWin, label="untrained")
    for x, y in zip(probSize, trainedWin):
        plt.text(x, y + 0.1, "%d" % y)
    for x, y in zip(probSize, untrainedWin):
        plt.text(x, y + 0.1, "%d" % y)
    plt.savefig("Win Rate")

    plt.figure("Efficiency")
    plt.title("Win Efficiency")
    plt.xlabel("Problem Size")
    plt.ylabel("Nodes")
    plt.bar(probSize, trainedNodes, label="trained")
    plt.bar(probSize, untrainedNodes, label="untrained")
    for x, y in zip(probSize, trainedNodes):
        plt.text(x, y + 0.1, "%d" % y)
    for x, y in zip(probSize, untrainedNodes):
        plt.text(x, y + 0.1, "%d" % y)
    plt.savefig("Efficiency")
    plt.show()
