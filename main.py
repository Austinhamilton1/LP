from simplex import f, Simplex

z = f(x1=1, x2=1, x3=1, x4=1)
f1 = f(x1=-2, x2=8, x4=10, b__gte=50_000)
f2 = f(x1=5, x2=2, b__gte=100_000)
f3 = f(x1=3, x2=-5, x3=10, x4=2, b__gte=25_000)

solver = Simplex(None)

problem = '''
Minimize x1 + x2 + x3 + x4
subject to
    -2x1 + 8x2 + 10x4 >= 50000
    5x1 + 2x2 >= 100000
    3x1 - 5x2 + 10x3 + 2x4 >= 25000
'''

solver = Simplex(None)

solver.parse(problem)

print(solver.solve())