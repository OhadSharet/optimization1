import numpy as np
import math


def ex1b():
    A = np.array([[1, 2, 3, 4],
                  [2, 4, -4, 8],
                  [-5, 4, 1, 5],
                  [5, 0, -3, -7]])
    AT = A.transpose()
    ATA = AT @ A
    eigen_vals, w = np.linalg.eig(ATA)

    print("eigen_vals : \n%s " % str(eigen_vals))
    print("ATA : \n %s" % str(ATA))
    max_sigma = math.sqrt(max(eigen_vals))
    print("max sigma : %s" % max_sigma)
    x = w @ np.array([[1], [0], [0], [0]])
    print("x : %s" % str(x))


def gram_schmidt(A):
    a_cols = A.transpose()
    new_cols = np.array([a_cols[0] / np.linalg.norm(a_cols[0], ord=2)])
    a_cols = np.delete(a_cols, 0, 0)
    for a in a_cols:
        q = get_ortogonal_vector(a,new_cols)
        new_cols = np.vstack([new_cols,q])
    return new_cols.transpose()


def get_ortogonal_vector(vec, mat):
    v2 = vec
    for v1 in mat:
        v2 = v2 - (v1.transpose() @ v2) * v1
    return v2/np.linalg.norm(v2,ord=2)


def print_hi(name):
    ex1b()

    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    A = np.array([[1,1,0],[0,1,0],[0,0,1]])
    print(A)
    print (gram_schmidt(A))
