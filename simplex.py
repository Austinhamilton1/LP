import numpy as np

class f:
    def __init__(self, **vars):
        '''
        Represents a function for an LP problem
        '''
        self.vars = {}
        self.b = None
        for var, coeff in vars.items():
            if var == 'b__gte':
                self.b = coeff
                self.type = '>='
            elif var == 'b__lte':
                self.b = coeff
                self.type = '<='
            elif var == 'b__eq':
                self.b = coeff
                self.type = '='
            else:
                self.vars[var] = coeff
        self.slack_vars = {}
        self.artificial_vars = {}

    def __str__(self):
        '''
        Convert the equation to a string representation
        '''
        result = ''
        for i, var in enumerate(sorted(self.vars.keys(), key=lambda x: str(x))):
            sign = ''
            if i > 0 and self.vars[var] >= 0:
                sign = '+'
            result += f'{sign}{self.vars[var]}{var}'
        for var in sorted(self.slack_vars.keys(), key=lambda x: str(x)):
            sign = '' if self.slack_vars[var] < 0 else '+'
            result += f'{sign}{self.slack_vars[var]}{var}'
        for var in sorted(self.artificial_vars.keys(), key=lambda x: str(x)):
            sign = '' if self.artificial_vars[var] < 0 else '+'
            result += f'{sign}{self.artificial_vars[var]}{var}'
        if self.b != None:
            result += f'{self.type}{self.b}'
        return result
    
    def __call__(self, **vars):
        result = 0
        for var, value in vars.items():
            result += self.vars[var] * value
        if self.b == None:
            return result
        if self.type == '>=':
            return result >= self.b
        elif self.type == '<=':
            return result <= self.b
        return result == self.b

class Simplex:
    def __init__(self, obj_fn, *constraints):
        '''
        A Simplex method solver to a linear programming problem
        '''
        self.obj_fn = obj_fn
        self.constraints = constraints
        #get global variables
        vars = set()
        for constraint in constraints:
            for var in constraint.vars:
                vars.add(var)

        #set global variables in every constraint
        for i in range(len(constraints)):
            for var in vars:
                if var not in constraints[i].vars:
                    constraints[i].vars[var] = 0

        #set global variables in objective function
        for var in vars:
            if var not in obj_fn.vars:
                obj_fn.vars[var] = 0

        #add slack variables to the constraints
        for i in range(len(constraints)):
            if constraints[i].type == '<=':
                constraints[i].slack_vars[f's{i+1}'] = 1
            elif constraints[i].type == '>=':
                constraints[i].slack_vars[f's{i+1}'] = -1
                constraints[i].artificial_vars[f'a{i+1}'] = 1
            elif constraints[i].type == '=':
                constraints[i].artificial_vars[f'a{i+1}'] = 1

        #get global slack variables
        slack_vars = set()
        for constraint in constraints:
            for var in constraint.slack_vars:
                slack_vars.add(var)
        
        #set global slack variables in objective function
        for var in slack_vars:
            if var not in obj_fn.slack_vars:
                obj_fn.slack_vars[var] = 0

        #set global slack variables in every constraint
        for i in range(len(constraints)):
            for var in slack_vars:
                if var not in constraints[i].slack_vars:
                    constraints[i].slack_vars[var] = 0

        #get global artificial variables
        artificial_vars = set()
        for constraint in constraints:
            for var in constraint.artificial_vars:
                artificial_vars.add(var)
        
        #set global artificial variables in objective function
        for var in artificial_vars:
            if var not in obj_fn.artificial_vars:
                obj_fn.artificial_vars[var] = 0

        #set global artificial variables in every constraint
        for i in range(len(constraints)):
            for var in artificial_vars:
                if var not in constraints[i].artificial_vars:
                    constraints[i].artificial_vars[var] = 0

        #get all variables and add them to a list
        self.vars = list(sorted(vars, key=lambda x: str(x)))
        self.slack_vars = list(sorted(slack_vars, key=lambda x: str(x)))
        self.artificial_vars = list(sorted(artificial_vars, key=lambda x: str(x)))

        #There is one row per constraint and one row for the objective function
        row_count = len(constraints) + 1

        #There is one column per original variable, one column per slack variable, one column
        #per artificial variable, one column for the objective function and one column for the
        #b_i values
        col_count = len(obj_fn.vars) + len(obj_fn.slack_vars) + len(obj_fn.artificial_vars) + 1 + 1

        self.canonical = True

        #this means we first need to convert to canonical form
        if len(obj_fn.artificial_vars) > 0:
            self.canonical = False
            row_count += 1
            col_count += 1

        #this keeps track of the state of the LP problem
        self.tableau = np.zeros((row_count, col_count))

        #build a non-canonical tableau
        if not self.canonical:
            self.tableau[0,0] = 1
            for i in range(len(obj_fn.artificial_vars)):
                self.tableau[0, 2 + len(obj_fn.vars) + len(obj_fn.slack_vars) + i] = -1
            self.tableau[1,1] = 1
            for i, var in enumerate(self.vars):
                self.tableau[1,2+i] = obj_fn.vars[var]
            for i, var in enumerate(self.slack_vars):
                self.tableau[1,2+len(self.vars)+i] = obj_fn.slack_vars[var]
            for i, var in enumerate(self.artificial_vars):
                self.tableau[1,2+len(self.vars)+len(self.slack_vars)+i] = obj_fn.artificial_vars[var]
            for i in range(len(constraints)):
                for j, var in enumerate(self.vars):
                    self.tableau[2+i,2+j] = constraints[i].vars[var]
                for j, var in enumerate(self.slack_vars):
                    self.tableau[2+i,2+len(self.vars)+j] = constraints[i].slack_vars[var]
                for j, var in enumerate(self.artificial_vars):
                    self.tableau[2+i,2+len(self.vars)+len(self.slack_vars)+j] = constraints[i].artificial_vars[var]
                self.tableau[2+i,-1] = constraints[i].b
        #build a canonical tableau
        else:
            self.tableau[0,0] = 1
            for i, var in enumerate(self.vars):
                self.tableau[0,1+i] = obj_fn.vars[var]
            for i, var in enumerate(self.slack_vars):
                self.tableau[0,1+len(self.vars)+i] = obj_fn.slack_vars[var]
            for i in range(len(constraints)):
                for j, var in enumerate(self.vars):
                    self.tableau[1+i,1+j] = constraints[i].vars[var]
                for j, var in enumerate(self.slack_vars):
                    self.tableau[1+i,1+len(self.vars)+j] = constraints[i].slack_vars[var]
                self.tableau[1+i,-1] = constraints[i].b

    def solve(self, type='maximize'):
        '''
        Solve an LP problem. Returns None if the solution is infeasible, otherwise
        return a dictionary of variables mapped to values
        '''
        #minimization is the same as the maximization of the negative coefficients
        if type == 'minimize':
            self.tableau[1,2:-1] *= -1

        #convert the tableau into canonical form if necessary
        if not self.canonical:
            self.tableau = self._to_canonical(self.tableau)
            if self.tableau is None:
                return None
        
        #try to solve the tableau (a cycle will indicate it can't be solved)
        pivots = set()
        while not self._is_solved(self.tableau):
            row, col = self._pivot(self.tableau)
            if (row, col) in pivots:
                return None
            pivots.add((row, col))

        #build the solution
        solution = {
            'optimal': self.tableau[0,-1] / self.tableau[0,0],
        }

        #parse the solution (solved system of equations)
        for i in range(len(self.vars)):
            if np.count_nonzero(self.tableau[:,1+i]) == 1:
                nonzero = np.flatnonzero(self.tableau[:,1+i])[0]
                solution[self.vars[i]] = self.tableau[nonzero,-1] / self.tableau[nonzero,1+i]
            else:
                solution[self.vars[i]] = 0

        return solution

    def _pivot_column(self, tableau):
        '''
        Return the pivot column for a tableau (this is choosing the entering variable)
        '''
        mask = np.where(tableau[0] <= 0, -np.inf, tableau[0])
        #last column is the value of the objective function (don't return this ever)
        mask[-1] = -np.inf
        return np.argmax(mask)
    
    def _pivot_row(self, tableau, column):
        '''
        Return the pivot row for a tableau and column (this is choosing the leaving variable)
        '''
        mask = np.where(tableau[:,column] <= 0, np.inf, tableau[:,-1] / tableau[:,column])
        #first row is the objective function (don't return this ever)
        mask[0] = np.inf
        #ratio must be strictly positive
        mask = np.where(mask <= 0, np.inf, mask)
        return np.argmin(mask)
    
    def _pivot(self, tableau):
        '''
        Perform a pivot operation on the tableau. Update the tableau in-place, return
        the row and column
        '''
        #select a pivot row and column
        pivot_column = self._pivot_column(tableau)
        pivot_row = self._pivot_row(tableau, pivot_column)

        #set the tableau[pivot_row, pivot_column] = 1
        tableau[pivot_row] *= 1 / tableau[pivot_row,pivot_column]

        #apply row operation to every row
        for i in range(len(tableau)):
            if i == pivot_row:
                continue
            #set the pivot_column to all 0's except for the pivot_row
            tableau[i] += -tableau[i,pivot_column] * tableau[pivot_row]
        return pivot_row, pivot_column
    
    def _is_solved(self, tableau):
        '''
        Determine if a tableau is solved (this is equivalent to determining if the identity
        matrix exists in the tableau)
        '''
        return np.size(np.where(tableau[0,1:-1] > 0)[0], axis=0) == 0
    
    def _to_canonical(self, tableau):
        '''
        Convert a tableau into canonical form return new tableau if possible, None otherwise
        '''
        #add the rows with artificial variables to the objective row
        artificial_vars = np.where(tableau[0] == -1)[0]
        artificial_rows = []
        for i in artificial_vars:
            artificial_rows.append(np.where(tableau[:,i] == 1)[0][0])
        for i in artificial_rows:
            tableau[0] += tableau[i]

        #pivot until we cycle or until the value of the objective function is sufficiently
        #close to 0 (should be 0, but we must account for floating point error)
        pivots = set()
        while tableau[0,-1] > 0.00001:
            row, col = self._pivot(tableau)
            if (row, col) in pivots:
                return None
            pivots.add((row, col))

        #solution is feasible, remove artificial variables from the tableau
        new_tableau = np.array([
            [tableau[i,j] for j in range(1, np.size(tableau, axis=1)) if j not in artificial_vars]
            for i in range(1, np.size(tableau, axis=0))
        ], dtype=np.float64)
        return new_tableau

z = f(x1=1, x2=1, x3=1, x4=1, x5=1, x6=1, x7=1)
f1 = f(x1=1, x4=1, x5=1, x6=1, x7=1, b__gte=30)
f2 = f(x1=1, x2=1, x5=1, x6=1, x7=1, b__gte=20)
f3 = f(x1=1, x2=1, x3=1, x6=1, x7=1, b__gte=15)
f4 = f(x1=1, x2=1, x3=1, x4=1, x7=1, b__gte=17)
f5 = f(x1=1, x2=1, x3=1, x4=1, x5=1, b__gte=19)
f6 = f(x2=1, x3=1, x4=1, x5=1, x6=1, b__gte=25)
f7 = f(x3=1, x4=1, x5=1, x6=1, x7=1, b__gte=30)

solver = Simplex(z, f1, f2, f3, f4, f5, f6, f7)

print(solver.solve(type='minimize'))