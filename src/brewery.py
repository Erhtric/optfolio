import pulp as pl

# prob contains the problem data, problem defined as maximization
# you can also define it as a minimization problem, less computationally expensive (?)
prob = pl.LpProblem("The Brewery Problem", pl.LpMaximize)

x1 = pl.LpVariable("Beer", 0)
x2 = pl.LpVariable("Ale", 0)

prob += 23*x1 + 13*x2, "Total profit per unit of product"

# constraints
prob += 15*x1 +  5*x2 <=  480, "Corn availability"
prob +=  4*x1 +  4*x2 <=  160, "Hops availability"
prob += 20*x1 + 35*x2 <= 1190, "Malt availability"

prob.writeLP("./results/Brewery.lp")

# problem solved using PuLP's choice of solver or
# prob.solve()
# prob.solve(CPLEX())
prob.solve()

print("Status:", pl.LpStatus[prob.status])

print("Total revenue = ", -1*pl.value(prob.objective))

# primal and dual variables optimal values
for v in prob.variables():
    print(v.name, "=", v.varValue)
for name, c in list(prob.constraints.items()):
    print(name, ":", c, "\t dual", c.pi, "\tslack", c.slack)