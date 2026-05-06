import gurobipy as gp
from gurobipy import GRB

m = gp.Model("first_model")

x = m.addVar(lb=0, name="x")
y = m.addVar(lb=0, name="y")

m.setObjective(3*x + 2*y, GRB.MAXIMIZE)

m.addConstr(x + 2*y <= 4)
m.addConstr(3*x + y <= 5)

m.optimize()

for v in m.getVars():
    print(v.VarName, v.X)

print("Objective:", m.ObjVal)
