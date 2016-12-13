# The fuzzy integral function that takes the scores per feature from the neural net
# and returns the score per label

import numpy as np
import math

# scoreMatrix is a 10xN numpy.matrix
# N is the number of features in a digit
def fuzzyIntegral(scoreMatrix):
    scoreShape = np.shape(scoreMatrix)
    scoreMatrix = np.sort(scoreMatrix)

    for i in range(scoreshape(0)):
        for i in range(scoreshape(1)):
            scoreMatrix[i,j] = scoreMatrix[i,j] * math.e **-j

    scoreVector = np.sum(scoreMatrix, axis=1)
    return scoreVector
