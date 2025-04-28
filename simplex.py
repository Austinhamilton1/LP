import numpy as np

class f:
    def __init__(self, b, **vars):
        '''
        Represents a function for an LP problem
        '''
        self.b = b
        self.vars = {}
        for var, coeff in vars.items():
            self.vars[var] = coeff

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
        if self.b != None:
            result += f'<={self.b}'
        return result
    
    def __call__(self, **kwargs):
        pass

class SimplexSolver:
    def __init__(self, obj_fn, *constraint_fn):
        '''
        Solves an LP problem using the simplex method
        '''
        self.obj_fn = obj_fn
        self.constraint_fn = constraint_fn

obj_fn = f(x1=4, x2=1, x3=4, b=None)
f1 = f(x1=2, x2=-1, x3=1, b=2)
f2 = f(x1=1, x2=2, x3=3, b=4)
f3 = f(x1=2, x2=2, x3=1, b=8)

print(obj_fn)
print(f1)
print(f2)
print(f3)