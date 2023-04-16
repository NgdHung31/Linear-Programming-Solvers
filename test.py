import numpy as np
import pandas as pd

data = pd.read_csv("test.csv")
matrix = np.array(data, dtype=float)
print(matrix.T)
row, col = np.shape(matrix)


def KiemTraHoanThanhChua(a):
    for i in range(row-1):
        if a[i+1][0] < 0:
            return False
    return True


def LayViTriGTNNTrongHamMT(a):
    temp = a[1][0]
    vitri = 1
    for i in range(2, row):
        if temp > a[i][0]:
            temp = a[i][0]
            vitri = i
    return vitri


def TimDiemXoay(a, vitri):
    S = []
    Si = []
    for i in range(1, col):
        if a[vitri][i] < 0:
            S.append(a[0][i]/abs(a[vitri][i]))
            Si.append(i)
    min = S[0]
    temp = 0
    for i in range(1, len(S)):
        if min > S[i]:
            min = S[i]
            temp = i
    return Si[temp]


while(KiemTraHoanThanhChua(matrix) == False):
    muitenr = LayViTriGTNNTrongHamMT(matrix)
    muitenc = TimDiemXoay(matrix, muitenr)
    print(muitenr, ";", muitenc)

    for i in range(row):
        if i != muitenr:
            matrix[i][muitenc] = matrix[i][muitenc] / \
                abs(matrix[muitenr][muitenc])

    matrix[muitenr][muitenc] = 1/matrix[muitenr][muitenc]

    for i in range(col):
        if i != muitenc:
            for j in range(row):
                if j != muitenr:
                    matrix[j][i] = matrix[j][i] + \
                        (matrix[muitenr][i] * matrix[j][muitenc])

    for i in range(col):
        if i != muitenc:
            matrix[muitenr][i] = matrix[muitenr][i]*matrix[muitenr][muitenc]

    print("\n\n", matrix.T)
