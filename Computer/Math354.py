# Math354.py
# Shaun Harker
# MIT LICENSE 2017

from IPython.display import display, Math, Latex, Markdown

import numpy as np
from numpy import matlib
from numpy import transpose as FloatTranspose
Matrix = lambda data : np.matrix(data, 'float64')

import sympy
from sympy import Rational
from sympy import Matrix as RationalMatrix
from sympy import latex
from sympy.functions import transpose as RationalTranspose

def Transpose(M):
    """
    Given a matrix M, return the transpose of M
    """
    if isinstance(M, type(Matrix([[1]]))):
        return FloatTranspose(M)
    elif isinstance(M, type(RationalMatrix([[1]]))):
        return RationalMatrix(RationalTranspose(M))
    else:
        raise TypeError("Transpose: Unrecognized Matrix Type")

def MatrixToLatex(M):
    """
    Obtain LaTeX string for matrix (for typesetting)
    """
    latex_string = ''
    if isinstance(M, type(Matrix([[1]]))):
        if len(M.shape) > 2:
            raise ValueError('bmatrix can at most display two dimensions')
        lines = str(M).replace('[', '').replace(']', '').splitlines()
        rv = [r'\begin{bmatrix}']
        rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
        rv +=  [r'\end{bmatrix}']
        latex_string = '\n'.join(rv)
    elif isinstance(M, type(RationalMatrix([[1]]))):
        latex_string = latex(M)
    else:
        latex_string = latex(M)
        #print(type(M))
        #raise TypeError("MatrixToLatex: Unrecognized Matrix Type")
    return latex_string

def DisplayMath(latex_string):
    """
    Pretty-print a matrix to IPython
    """
    display(Math(latex_string))
    
def DisplayMatrix(M):
    """
    Pretty-print a matrix to IPython
    """
    DisplayMath(MatrixToLatex(M))

def DisplayText(text):
    """
    Pretty-print a latex string to IPython
    """
    display(Markdown(text))

def Identity(n):
    """
    Return an identity Matrix of size nxn
    """
    return matlib.identity(n, 'float64')

def RationalIdentity(n):
    """
    Return a identity RationalMatrix of size nxn
    """
    return sympy.eye(n)

def FindPivotRow(M, j, r):
    """
    Given a matrix M, a column number j, and
    a row number r,
    find a row i >= r such that M[i,j] != 0.
    If no such row exists, return -1
    """
    # Get the number of rows and columns, m and n, of M,
    # by querying its "shape" attribute:
    m,n = M.shape
    
    # Make a list of all the row numbers for which there is
    # a non-zero entry in the jth column, and affix a "-1" to
    # the end of this list.
    pivots = [ i for i in range(r,m) if M[i,j] != 0 ] + [-1]
    
    # Return the first thing in the list. If there were no
    # pivots, then the "-1" will be returned. Otherwise, it
    # will return the smallest i such that M[i,j] != 0
    return pivots[0]

def AugmentedMatrix(A):
    """
    Given a matrix A, return a new matrix
    [A I], where I is an identity matrix
    with the same number of rows as A
    """
    m,n = A.shape
    M = A.copy()
    if isinstance(M, type(Matrix([[1]]))):
        return np.hstack((M, Identity(m)))
    elif isinstance(M, type(RationalMatrix([[1]]))):
        return M.row_join(RationalIdentity(m))
    else:
        print(type(A))
        raise TypeError("AugmentedMatrix: Unrecognized Matrix Type")

def RowSwitching(M, i1, i2):
    """
    Swap row "i1" and row "i2" in matrix "M"
    """
    temporary_row = M[i1,:].copy()
    M[i1,:] = M[i2,:]
    M[i2,:] = temporary_row

def RowMultiplication(M,i,c):
    """
    Multiply row "i" of matrix "M" by "c"
    """
    M[i,:] *= c
    
def RowAddition(M,i1,i2,c):
    """
    Add "c" times row i2 to row i1 in matrix M
    """
    M[i1,:] += c*M[i2,:]

def GaussJordanReduction(A, verbose = None ):
    """
    Given a matrix A, compute a row-equivalent matrix M
    that is in reduced row echelon form.
    In particular, return (R, E, r), where:
      R is a reduced row echelon form matrix row-equivalent to A
      E is a non-singular matrix such that R = E*A
      r is the rank of the matrix A (and also the matrix R)
    """
    # Make a copy M of the original matrix A
    # in case the caller of this function does not
    # want the original matrix overwritten
    M = AugmentedMatrix(A)
    n,m = A.shape
    r = 0
    
    # Display
    if verbose: DisplayText('## GaussJordanReduction:\n We will put the following matrix into reduced row echelon form: ')
    if verbose: DisplayMatrix(M[0:n,0:m])

    # Loop through columns 1...m
    for j in range(0,m):

        # Search for a pivot row in the jth column
        i = FindPivotRow(M, j, r)
        
        # If there is no pivot element, we will not 
        # find a linearly independent row by analyzing 
        # the jth column
        if i == -1: 
            if verbose: DisplayText('No pivot element available in column ' + str(j) + '.' )
            continue
        
        if verbose: DisplayText('Selected pivot value ' + str(M[i,j]) + ' at (' + str(i) + ', ' + str(j) + ')' + '.')
        
        # TYPE I ROW OPERATION
        # Swap rows
        # --> Don't bother switching a row with itself (does nothing)
        if r != i:
            if verbose: DisplayText('Swapping rows ' + str(r) + ' and ' + str(i) + '.')
            RowSwitching(M, r, i)
            if verbose: DisplayMatrix(M[0:n,0:m])
        
        # TYPE II ROW OPERATION
        # Use a row multiplication operation in order to
        # make the leading entry of row r "1" as desired
        # --> Don't bother dividing by 1
        if M[r,j] != 1:
            if verbose: DisplayText('Multiplying row ' + str(r) + ' by ' + str(1/M[r,j]) + '.')
            RowMultiplication(M, r, 1/M[r,j])
            if verbose: DisplayMatrix(M[0:n,0:m])

        # TYPE III ROW OPERATIONS
        # Loop through all rows and use row operations
        # to eliminate ("zero-out") all entries in M[:,j]
        # except for M[r,j] (which is the pivot element)
        for k in range(0,n):
            # Don't eliminate the pivot row itself!
            if k != r:
                # --> Don't bother subtracting 0
                if M[k,j]:
                    if verbose: DisplayText('Adding ' + str(-M[k,j]) + ' times pivot row ' + str(r) + ' to row ' + str(k) + '.')
                    RowAddition(M,k,r,-M[k,j])
                    if verbose: DisplayMatrix(M[0:n,0:m])
        
        # We've established a linearly independent row, so increment rank count
        r = r + 1
        
        # If we have reached full rank, terminate
        if r == n: break
    if verbose: 
        DisplayText('GaussJordanReduction Algorithm complete. Matrix is now in reduced row echelon form.' )
        DisplayText('The product of elementary matrices used to put the input matrix into reduced row echelon form is ' )
        DisplayMath(MatrixToLatex(M[0:n,m:]) + MatrixToLatex(A) + ' = ' + MatrixToLatex(M[0:n,0:m]))
        

    return((M[0:n,0:m], M[0:n,m:], r))

def Project(P, u):
    """
    Return (v, w) such that u = v + w, v \in range P, and w \in null P
    """
    TP = Transpose(P)
    dot = (TP*u)[0,0]
    v = P * dot / (TP * P)[0,0]
    w =  u - v
    return (v, w)

def Solve(A, b):
    """
    Return x such that Ax = b
    If there is no solution, return []
    """
    m,n = A.shape
    (R, E, rank) = GaussJordanReduction(A)
    c = E*b
    if any( c[r] != 0 for r in range(rank, m)):
        return [] # no solution
    x = Transpose(A[0,:]) * 0 
    for j in range(0,n):
        Rj = R[:,j]
        (v, w) = Project(Rj, c)
        x += v
        c -= w
    return x
