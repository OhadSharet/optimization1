import numpy as np
import math

def ex1b():
    A = np.array([[1,2,3,4],
               [2,4,-4,8],
               [-5,4,1,5],
               [5,0,-3,-7]])
    AT = A.transpose()
    ATA = AT  @ A
    eigen_vals,w = np.linalg.eig(ATA)

    print("eigen_vals : \n%s "%str(eigen_vals))
    print ("ATA : \n %s"%str(ATA))
    max_sigma = math.sqrt(max(eigen_vals))
    print ( "max sigma : %s"%max_sigma)
    x = w@np.array([[1],[0],[0],[0]])
    print ("x : %s"%str(x))




def gram_schmidt(A):
    n = len(A)
    R = np.zeros(np.zeros((n,n)))

def print_hi(name):
    ex1b()

    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    print_hi('PyCharm')
