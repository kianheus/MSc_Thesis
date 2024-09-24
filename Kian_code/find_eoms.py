import sympy as sp
from sympy.printing.str import StrPrinter


class CustomStrPrinter(StrPrinter):
    def _print_Pow(self, expr):
        base, exp = expr.as_base_exp()
        if exp == 1:
            return self._print(base)
        elif exp == -1:
            return f"1/({self._print(base)})"
        else:
            return f"{self._print(base)}^{self._print(exp)}"

# Use the custom printer
def custom_pretty(expr):
    return CustomStrPrinter().doprint(expr)


t = sp.Symbol('t')
q1= sp.Function("q1")(t)
q2= sp.Function("q2")(t)
q1_dot = sp.diff(q1, t)
q2_dot = sp.diff(q2, t)
q1_ddot = sp.diff(q1_dot, t)
q2_ddot = sp.diff(q2_dot, t)

# Define the link lengths and masses
m1, m2, g = sp.symbols('m1 m2 g')
l1, l2, lc1, lc2 = sp.symbols('l1 l2 lc1 lc2')
J1, J2 = sp.symbols('J1 J2')

# Positions of the center of mass for each link
# Defined in absolute angles from the horizontal 
x1 = lc1 * sp.cos(q1)
y1 = lc1 * sp.sin(q1)
x2 = l1 * sp.cos(q1) + lc2 * sp.cos(q2)
y2 = l1 * sp.sin(q1) + lc2 * sp.sin(q2)

# Velocities of the center of mass for each link
vx1 = sp.diff(x1, t)
vy1 = sp.diff(y1, t)
vx2 = sp.diff(x2, t)
vy2 = sp.diff(y2, t)

# Kinetic energy of each link
T1 = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * J1 * q1_dot**2
T2 = 0.5 * m2 * (vx2**2 + vy2**2) + 0.5 * J2 * q2_dot**2

# Total kinetic energy
T = T1 + T2

# Potential energy of each link
V1 = m1 * g * y1
V2 = m2 * g * y2

# Total potential energy
V = V1 + V2

# Lagrangian
L = T - V

# Generalized coordinates and velocities
q = [q1, q2]
q_dot = sp.Matrix([q1_dot, q2_dot])
q_ddot = sp.Matrix([q1_ddot, q2_ddot])
replacements = ()
for i in range(2):
    replacements += ((q[i].diff(t).diff(t), sp.Symbol(f'ddq{i + 1}')),
                    (q[i].diff(t), sp.Symbol(f'dq{i + 1}')),
                    (q[i], sp.Symbol(f'q{i + 1}')))

    
# Lagrangian equations
eoms = []
for i in range(2):
    L_qi = L.diff(q[i].diff(t)).diff(t) - L.diff(q[i])
    L_qi = L_qi.simplify().subs(replacements)
    eoms.append(L_qi)


# L_q2 = sp.diff(sp.diff(L, q2_dot), t) - sp.diff(L, q2)
# L_q2 = L_q2.simplify().subs({sp.diff(q2_dot, t): q2_ddot})
# L_q3 = sp.diff(sp.diff(L, q3_dot), t) - sp.diff(L, q3)
# L_q3 = L_q3.simplify().subs({sp.diff(q3_dot, t): q3_ddot})
# Pretty print the mass matrix, Coriolis matrix, and equations of motion


def format_lagrange(eom, name):
    eom = str(eom).replace('1.0*','')
    terms = str(eom).split('+')
    formatted = f"{name} = " 
    for i, term in enumerate(terms):
        if i == 0:
            formatted += f"{term.strip()}\n"
        else:
            formatted += f"       + {term.strip()}\n"
    return formatted

L_q1_formatted = format_lagrange(eoms[0], "L_q1")
L_q2_formatted = format_lagrange(eoms[1], "L_q2")

if __name__ == "__main__":
    # Print the formatted equations
    print(L_q1_formatted)
    print(L_q2_formatted)
# Custom printer to replace ** with ^ and remove superscript formatting



# Generalized velocity and acceleration
q_dot = [q[0].diff(t), q[1].diff(t)]
q_ddot = [q_dot[0].diff(t), q_dot[1].diff(t)]
# Mass matrix (inertia matrix)
M = sp.zeros(2)
M4C = sp.zeros(2)
for i in range(2):
    for j in range(2):
        M4C[i, j] = (T.diff(q_dot[i]).diff(q_dot[j])).simplify()
        M[i, j] = M4C[i, j].subs(replacements)
        # M[i, j] = sp.diff(T, q_dot[i], q_dot[j])
M = sp.nsimplify(M)

# Coriolis matrix
C = sp.zeros(2, 2)
for i in range(2):
    for j in range(2):
        for k in range(2):
            C[i, j] += 0.5 * (M4C[i, j].diff(q[k]) + M4C[i, k].diff(q[j]) - M4C[j, k].diff(q[i])) * q_dot[k]
        C[i, j] = C[i, j].simplify().subs(replacements)
C = sp.nsimplify(C)

# Gravitational force vector
G = sp.zeros(2, 1)
for i in range(2):
    G[i] = V.diff(q[i]).simplify().subs(replacements)
G = sp.nsimplify(G)

if __name__ == "__main__":
    # Print the matrices using the custom printer
    print("\nMass Matrix (M):")
    print(custom_pretty(M))

    print("\nCoriolis Matrix (C):")
    print(custom_pretty(C))

    print("\nGravitational Vector (G):")
    print(custom_pretty(G))