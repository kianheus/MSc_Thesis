from find_eoms import *
import sympy as sp
from sympy.printing.str import StrPrinter


# Actuator parameters
xa, ya = sp.symbols('xa ya') #x, y coordinate of actuator

# Rope connection length along joint 2
lr2 = sp.symbols('lr2')
kx = l1 * sp.cos(q1) + lr2 * sp.cos(q2)
ky = l1 * sp.sin(q1) + lr2 * sp.sin(q2)

k = sp.Matrix([[kx],
               [ky]])

dkdq1 = sp.diff(k, q1)
dkdq2 = sp.diff(k, q2)

R = sp.Matrix([[xa - kx],
               [ya - ky]])
Rx = R[0]
Ry = R[1]
r = R/R.norm()

l = sp.sqrt((xa - kx)**2 + (ya - ky)**2)
A1 = sp.diff(l, q1).simplify()
A2 = sp.diff(l, q2).simplify()
if __name__ == "__main__":
    sp.pprint(A1, wrap_line = False)
    sp.pprint(A2, wrap_line = False)

A = sp.Matrix([[A1],
               [A2]])

A = sp.simplify(A)
if __name__ == "__main__":
    print(custom_pretty(A))
    print("\n")
    sp.pprint(A, wrap_line=False)


h1 = sp.atan2(Rx, Ry)
h1 = l
h2 = q1
theta = sp.Matrix([[h1],
                   [h2]])

Jh = theta.jacobian(q)
Jh = sp.simplify(Jh)
if __name__ == "__main__":
    sp.pprint(Jh, wrap_line = False)

Jh_inv = Jh.inv()
Jh_invtrans = Jh_inv.transpose()

A_theta = Jh_invtrans * A
A_theta = sp.simplify(A_theta)
if __name__ == "__main__":
    sp.pprint(A_theta, wrap_line = False)