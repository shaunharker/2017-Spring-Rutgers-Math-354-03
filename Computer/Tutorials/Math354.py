# Math354.py
# Shaun Harker
# MIT LICENSE 2017

from IPython.display import display, Math, Latex, Markdown

import numpy as np
from numpy import matlib
from numpy import transpose as FloatTranspose
FloatMatrix = lambda data : np.matrix(data, 'float64')

import sympy
from sympy import Rational
from sympy import Matrix as RationalMatrix
from sympy import latex
from sympy.functions import transpose as RationalTranspose

def Transpose(M):
    """
    Given a matrix M, return the transpose of M
    """
    if isinstance(M, type(FloatMatrix([[1]]))):
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
    if isinstance(M, type(FloatMatrix([[1]]))):
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

def FloatIdentity(n):
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
    if isinstance(M, type(FloatMatrix([[1]]))):
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

Matrix = RationalMatrix
Identity = RationalIdentity

# Linear Programming

class LinearProgram:
    def __init__(self, A, b, c, I, J, K, L):
        self.A_ = A
        self.b_ = b
        self.c_ = c
        self.I_ = I
        self.J_ = J
        self.K_ = K
        self.L_ = L

    def __str__(self):
        return 'Maximize $c^T x$ subject to $A_I x \leq b_I$, $A_J x = b_J$, $x_K \geq 0$, $x_L$ unrestricted, where\n\n' + \
               '$$(A,b,c,I,J,K,L) = \\left(' + MatrixToLatex(self.A_) + ", " + MatrixToLatex(self.b_) + ", " + MatrixToLatex(self.c_) + ", " + str(self.I_) + ", " + str(self.J_) + ", " + str(self.K_) + ", "+ str(self.L_) + "\\right)$$"
    
    def __repr__(self):
        return str(self)

    def _repr_markdown_(self):
        return str(self)

def Dual(linear_program):
    A = -Transpose(linear_program.A_)
    b = -linear_program.c_.copy()
    c = -linear_program.b_.copy()
    I = linear_program.K_[:]
    J = linear_program.L_[:]
    K = linear_program.I_[:]
    L = linear_program.J_[:]
    return LinearProgram(A,b,c,I,J,K,L)

def EliminateInequalities(linear_program):
    """
    Return a new linear program equivalent to the input in which the
    inequalities have been turned into equalities by the addition of slack variables
    """
    number_of_inequality_constraints = len(linear_program.I_)
    A = linear_program.A_
    (m,n) = A.shape

    A_slack = Matrix(m, number_of_inequality_constraints, lambda i, j: 0)
    for count, i in enumerate(linear_program.I_):
        A_slack[i,count] = 1
    A = A.row_join(A_slack)
    b = linear_program.b_
    c = linear_program.c_
    c = c.col_join(Matrix(number_of_inequality_constraints,1, lambda i, j: 0))
    I = []
    J = list(range(0,m))
    K = linear_program.K_ + list(range(n, n + number_of_inequality_constraints))
    L = linear_program.L_
    return LinearProgram(A,b,c,I,J,K,L)

def EliminateUnrestricted(linear_program):
    """
    Return a new linear program equivalent to the input in which the
    inequalities have been turned into equalities by the addition of slack variables
    """
    number_of_unrestricted_variables = len(linear_program.L_)
    A = linear_program.A_
    c = linear_program.c_ 
    (m,n) = A.shape
    A_neg = Matrix(m, number_of_unrestricted_variables, lambda i, j: 0)
    c_neg = Matrix(number_of_unrestricted_variables, 1, lambda i, j: 0)
    for count, i in enumerate(linear_program.L_):
        A_neg[:,count] = -A[:,count]
        c_neg[count] = -c[count]
    A = A.row_join(A_neg)
    b = linear_program.b_
    c = c.col_join(c_neg)
    I = linear_program.I_
    J = linear_program.J_
    K = list(range(0, n + number_of_unrestricted_variables ))
    L = []
    return LinearProgram(A,b,c,I,J,K,L)

def ConvertToCanonicalForm(linear_program):
    return EliminateInequalities(EliminateUnrestricted(linear_program))

# Exercise. Convert to DualCanonicalForm.

def StandardForm(A,b,c):
    (m,n) = A.shape
    I = list(range(0,m))
    J = []
    K = list(range(0,n))
    L = []
    return LinearProgram(A,b,c,I,J,K,L)

def CanonicalForm(A,b,c):
    (m,n) = A.shape
    I = []
    J = list(range(0,m))
    K = list(range(0,n))
    L = []
    return LinearProgram(A,b,c,I,J,K,L)

def DualStandardForm(A,b,c):
    (m,n) = A.shape
    I = list(range(0,m))
    J = []
    K = list(range(0,n))
    L = []
    return LinearProgram(-A,-b,-c,I,J,K,L)

def DualCanonicalForm(A,b,c):
    (m,n) = A.shape
    I = list(range(0,m))
    J = []
    K = []
    L = list(range(0,n))
    return LinearProgram(-A,-b,-c,I,J,K,L)

def CanonicalWithNonnegativeRHS(linear_program):
    """
    Return an equivalent canonical form linear program where b >= 0
    """
    canonical_form = ConvertToCanonicalForm(linear_program)
    (m,n) = canonical_form.A_.shape
    A = canonical_form.A_.copy()
    b = canonical_form.b_.copy()
    for i in range(0, m):
        if b[i] < 0:
            b[i] = -b[i]
            A[i,:] = -A[i,:]
    c = canonical_form.c_
    return CanonicalForm(A,b,c)

def AuxiliaryForm(linear_program):
    """
    Return a Linear Program corresponding to Phase I solution during simplex method
    """
    canonical_form = CanonicalWithNonnegativeRHS(linear_program)
    (m,n) = canonical_form.A_.shape
    variables = list(range(0, n))
    A = canonical_form.A_.copy()
    # Greedy selection of basic variables
    unselected = set(range(0,m))
    def IsBasisVector(v):
        (m,n) = v.shape
        return (n==1) and all( v[i]*v[i] == v[i] for i in range(0,m) ) and (sum(v[i] for i in range(0,m)) == 1)
    for j in range(0, n):
        if IsBasisVector(A[:,j]):
            for i in range(0,m):
                if A[i,j]:
                    unselected.discard(i)
    # Need auxiliary variables for constraints left in "unselected"
    num_aux_variables = len(unselected)
    A_aux = Matrix(m, num_aux_variables, lambda i, j: 0)
    for count, aux in enumerate(unselected):
        A_aux[aux,count] = 1
    A = A.row_join(A_aux)
    b = canonical_form.b_.copy()
    c = Transpose(Matrix([[0 for i in range(0,n)] + [-1 for i in range(0, num_aux_variables)]]))
    return CanonicalForm(A,b,c)

def InitialBasicFeasibleSolution(linear_program):
    """
    Find initial BFS when A has identity submatrix, b >= 0
    Assumes canonical form
    """
    linear_program = CanonicalWithNonnegativeRHS(linear_program)
    (m,n) = linear_program.A_.shape
    A = linear_program.A_
    b = linear_program.b_ 
    B = [-1 for i in range(0,m)]
    # DRY
    def IsBasisVector(v):
        (m,n) = v.shape
        return (n==1) and all( v[i]*v[i] == v[i] for i in range(0,m) ) and (sum(v[i] for i in range(0,m)) == 1)
    for j in range(0, n):
        if IsBasisVector(A[:,j]):
            for i in range(0,m):
                if A[i,j]:
                    B[i] = j
    if min(b) < 0 or min(B) == -1:
        raise ValueError("User Error: Cannot automatically detect initial basic feasible solution")
    return B

class Tableau:
    def __init__(self, linear_program, choice_of_basic_variables):
        self.LP_ = linear_program
        self.B_ = choice_of_basic_variables
        (m,n) = self.LP_.A_.shape
        self.m_ = m
        self.n_ = n
        self.compute()

    def compute(self):
        A_B = self.LP_.A_[:,self.B_];
        self.A_B_inv_ = A_B.inv()
        self.A_ = self.A_B_inv_ * self.LP_.A_;
        self.b_ = self.A_B_inv_ * self.LP_.b_;
        self.c_ = (Transpose(self.LP_.c_[self.B_,0]) * self.A_B_inv_) * self.LP_.A_ - Transpose(self.LP_.c_)
        self.d_ = Transpose(self.LP_.c_[self.B_,0]) * self.b_

    def dual_feasible(self):
        return min(self.c_) >= 0

    def feasible(self):
        return min(self.b_) >= 0 

    def optimal(self):
        return self.dual_feasible() and self.feasible() 

    def objective(self):
        return self.d_

    def solution(self):
        result = Matrix(self.n_,1, lambda i,j: 0)
        for k,j in enumerate(self.B_):
            result[j] = self.b_[k]
        return result

    def dual_solution(self):
        result = Matrix(self.m_,1, lambda i,j: 0)
        return Transpose(self.A_B_inv_)*self.LP_.c_[self.B_, 0]

    def choose_entering(self):
        """
        Choose an entering variable
        """
        min_value = min(self.c_)
        list_of_values = list(self.c_)
        #print(min_value)
        #print(list_of_values)
        bland_choice = min( [ j for j in range(0,self.n_) if j not in self.B_ and self.c_[j] == min_value ])
        #print(bland_choice)
        #print("returning from choose_entering")
        return bland_choice

    def choose_departing(self, entering):
        """
        Given a choice of entering variable, choose departing variable.
        If no departing variable may be chosen, there is an unbounded ray.
        In this case, return -1
        """
        #print(self.b_)
        #print(self.A_[:,entering])

        theta_ratios = [ self.b_[i] / self.A_[i,entering] for i in range(0, self.m_ )]
        #print(theta_ratios)
        #print([ t for t in theta_ratios if t != sympy.zoo and t >= 0] )
        #print(len([ t for t in theta_ratios if t >= 0] ))
        if len([ t for t in theta_ratios if t != sympy.zoo and t >= 0] ) == 0:
            return -1
        min_ratio = min([ t for t in theta_ratios if t != sympy.zoo and t >= 0])
        choice = min( [ i for i in range(0,self.m_) if theta_ratios[i] == min_ratio ])
        return self.B_[choice]

    def choose_dual_departing(self):
        min_value = min(self.b_)
        list_of_values = list(self.b_)
        bland_choice = min( [ i for i in range(0,self.m_) if self.b_[i] == min_value ])
        return self.B_[bland_choice]

    def choose_dual_entering(self, departing):
        """
        Given a choice of entering variable, choose departing variable.
        If no departing variable may be chosen, there is an unbounded ray.
        In this case, return -1
        """
        i = self.B_.index(departing)
        theta_ratios = [ self.c_[j] / self.A_[i, j] for j in range(0, self.n_ )]
        if len([ t for t in theta_ratios if t != sympy.nan and t < 0] ) == 0:
            return -1
        max_ratio = max([ t for t in theta_ratios if t != sympy.nan and t < 0])
        choice = min( [ j for j in range(0,self.n_) if theta_ratios[j] == max_ratio ])
        return choice

    def pivot(self, entering, departing):
        self.B_[self.B_.index(departing)] = entering
        self.compute()

    def markdown(self):
        string = "|"
        for j in range(0, self.n_):
            string += '| $x_' + str(j) + '$'
        string += ' | |\n'
        
        string += '| ---- '
        for j in range(0, self.n_):
            string += '| ----'
        string += '| ---- |\n'

        for i in range(0, self.m_):
            string += '| $x_' + str(self.B_[i]) + '$ |'
            for j in range(0, self.n_):
                string += '$' + str(self.A_[i,j]) + '$ | '
            string +='$' + str(self.b_[i]) + '$ |\n'

        string += '|  '
        for j in range(0, self.n_):
            string +='| $' + str(self.c_[j]) + '$ '
        string += '| $' + str(self.d_[0]) + '$|\n'
        return string

    def _repr_markdown_(self):
        return self.markdown()

def SimplexMethod(linear_program, initial_bfs = None):
    """
    Perform simplex method when basic method applies (A has identity submatrix, b >= 0)
    """    
    # Find initial basic feasible solution
    if initial_bfs == None:
        B = InitialBasicFeasibleSolution(linear_program)
        print("Automatically detected initial basic feasible solution with basic variables " + str(B))
    else:
        B = initial_bfs
        print("Using user-supplied initial basic feasible solution with basic variables " + str(B))
    tableau = Tableau(linear_program, B)
    while not tableau.optimal():
        entering = tableau.choose_entering()
        departing = tableau.choose_departing(entering)
        if departing == -1:
            raise ValueError("Linear Program is objectively unbounded")
        display(Markdown(tableau.markdown()))
        display(Markdown("Choosing entering variable to be $x_" + str(entering) + "$"))
        display(Markdown("Choosing departing variable to be $x_" + str(departing) + "$"))
        tableau.pivot(entering, departing)
    display(Markdown("Final tableau:"))
    display(Markdown(tableau.markdown()))
    return (tableau.solution(), tableau.dual_solution())

def DualSimplexMethod(linear_program, initial_bdfs = None):
    """
    Perform simplex method when basic method applies (A has identity submatrix, b >= 0)
    """    
    # Find initial basic feasible solution
    if initial_bdfs == None:
        B = InitialBasicDualFeasibleSolution(linear_program)
        print("Automatically detected initial basic dual feasible solution with basic variables " + str(B))
    else:
        B = initial_bdfs
        print("Using user-supplied initial basic dual feasible solution with basic variables " + str(B))
    tableau = Tableau(linear_program, B)
    while not tableau.optimal():
        departing = tableau.choose_dual_departing()
        entering = tableau.choose_dual_entering(departing)
        if entering == -1:
            raise ValueError("Linear Program is infeasible")
        display(Markdown(tableau.markdown()))
        display(Markdown("Choosing departing variable to be $x_" + str(departing) + "$"))
        display(Markdown("Choosing entering variable to be $x_" + str(entering) + "$"))
        tableau.pivot(entering, departing)
    display(Markdown("Final tableau:"))
    display(Markdown(tableau.markdown()))
    return (tableau.solution(), tableau.dual_solution())




