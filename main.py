import numpy as np
import math


def ex1b():
    A = np.array([[1, 2, 3, 4],
                  [2, 4, -4, 8],
                  [-5, 4, 1, 5],
                  [5, 0, -3, -7]])
    AT = A.transpose()
    ATA = AT @ A
    eigenvalues, w = np.linalg.eig(ATA)

    print("ATA eigenvalues : \n %s " % str(eigenvalues))
    print("ATA : \n %s" % str(ATA))
    max_sigma = math.sqrt(max(eigenvalues))
    print("sigma max : %s" % max_sigma)
    x = w @ np.array([[1], [0], [0], [0]])
    print("x : \n %s" % str(x))


def ex4a():
    A = np.array([[2, 1, 2],
                  [1, -2, 1],
                  [1, 2, 3],
                  [1, 1, 1]])
    b = np.array([[6],
                  [1],
                  [5],
                  [2]])
    AT = A.transpose()
    ATA = AT @ A
    ATb = AT @ b
    L = np.linalg.cholesky(ATA)
    LT = L.transpose()
    L_inverse = np.linalg.inv(L)
    LT_inverse = np.linalg.inv(LT)

    # y = L^-1 @ ATb
    # x = (L @ LT)^-1 @ ATb => x = LT^-1 @ L^-1 @ Ab => x = LT^-1 @ y
    y = L_inverse @ ATb
    x = LT_inverse @ y
    print("ATA : \n %s" % str(ATA))
    print("L : \n %s" % str(L))
    print("x : \n %s" % str(x))


def ex4b():
    A = np.array([[2, 1, 2],
                  [1, -2, 1],
                  [1, 2, 3],
                  [1, 1, 1]])
    b = np.array([[6],
                  [1],
                  [5],
                  [2]])
    AT = A.transpose()
    ATA = AT @ A
    ATb = AT @ b
    Q, R = np.linalg.qr(ATA)
    Q_inverse = np.linalg.inv(Q)
    R_inverse = np.linalg.inv(R)

    # y = Q^-1 @ ATb
    # x = (Q @ R)^-1 @ ATb => x = R^-1 @ Q^-1 @ Ab => x = R^-1 @ y
    y = Q_inverse @ ATb
    x = R_inverse @ y
    print("ATA : \n %s" % str(ATA))
    print("Q : \n %s" % str(Q))
    print("R : \n %s" % str(R))
    print("x : \n %s" % str(x))


def gram_schmidt(A, make_orthogonal):
    a_cols = A.transpose()
    n = len(A[0])
    new_cols = np.array([a_cols[0] / np.linalg.norm(a_cols[0], ord=2)])
    rs = [np.zeros(n)]
    rs[0][0] = np.linalg.norm(a_cols[0], ord=2)
    a_cols = np.delete(a_cols, 0, 0)

    for a in a_cols:
        q, r = make_orthogonal(a, new_cols, n)
        new_cols = np.vstack([new_cols, q])
        rs.append(r)
    finale_R = np.stack(rs, axis=0)
    return new_cols.transpose(), finale_R.transpose()


# def gram_schmidt2(A):
#     n=len(A)
#     R = np.zeros((n,n))
#     a_cols = A.transpose()
#     r[0][0] = np.linalg.norm(a_cols[0],ord=2)
#     final_q = [a_cols[0]/r[0][0]]
#     for i in range(1,len(a_cols)):
#         ai = a_cols[i]
#         qi=ai
#         for j in range(i-1):
#             qj = final_q[j]
#             R[j][i] = qj.transpose()@ai
#             qi=qi-R[j][i]*qj
#         R[i][i] = np.linalg.norm(qi,ord=2)
#         qi=qi/R[i][i]
#         final_q = np.vstack([final_q,qi])
#     return final_q.transpose(),R


def get_orthogonal_vector(vec, mat, rlen):
    '''
    this function calculate the orthogonal vector according to the *original*
    gram_schmidt algorithm
    :param vec: the vector we want to make orthogonal to the other vectors
    :param mat: all past vectors
    :return:v2: the orthogonal vector
    '''
    v2 = vec
    r = np.zeros(rlen)
    j = 0
    for v1 in mat:
        r[j] = v1.transpose() @ vec
        v2 = v2 - r[j] * v1
        j += 1
    r[j] = np.linalg.norm(v2, ord=2)
    q = v2 / r[j]
    return q, r


def get_orthogonal_vector_modified(vec, mat, rlen):
    '''
    this function calculate the orthogonal vector according to the *modified*
    gram_schmidt algorithm
    :param vec: the vector we want to make orthogonal to the other vectors
    :param mat: all past vectors
    :return:v2: the orthogonal vector
    '''
    v2 = vec
    r = np.zeros(rlen)
    j = 0
    for v1 in mat:
        r[j] = v1.transpose() @ v2
        v2 = v2 - r[j] * v1
        j += 1
    r[j] = np.linalg.norm(v2, ord=2)
    q = v2 / r[j]
    return q, r


def gram_schmidt_original(A):
    return gram_schmidt(A, get_orthogonal_vector)


def gram_schmidt_modified(A):
    return gram_schmidt(A, get_orthogonal_vector_modified)


def ex5(eps=1):
    A = np.array([[1, 1, 1], [eps, 0, 0], [0, eps, 0], [0, 0, eps]])
    q1, r1 = gram_schmidt_original(A)
    print("\n==============q================= - gram_schmidt_original \n")
    print(q1)
    print("\n==============r================= - gram_schmidt_original\n")
    print(r1)

    print("\n===========Q@R============== gram_schmidt_original\n")
    print(q1 @ r1)
    qtq = q1.transpose() @ q1
    print("\n||QTQ-I|| - gram_schmidt_original: %s" % str(np.linalg.norm(qtq - np.identity(len(qtq[0])), ord='fro')))

    q2, r2 = gram_schmidt_modified(A)
    print("\n==============q================= - gram_schmidt_modified\n")
    print(q2)
    print("\n==============r================= - gram_schmidt_modified\n")
    print(r2)

    print("\n===========Q@R============== gram_schmidt_modified\n")
    print(q2 @ r2)

    qtq = q2.transpose() @ q2
    print("\n||QTQ-I|| - gram_schmidt_modified : %s" % str(np.linalg.norm(qtq - np.identity(len(qtq[0])), ord='fro')))


if __name__ == '__main__':
    ex4b()
    #ex5()
    #ex5(float('1e-10'))
