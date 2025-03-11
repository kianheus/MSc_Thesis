"""
LUMPED_PARAMETERS = dict(
    l1 = 3.13, #Length of link 1 [m]
    l2 = 1.98, #Length of link 2 [m]
    m = 3., #System mass [kg] (attached at the end of link 2)
    g = 9.81, #Gravitational acceleration [m/s^2]
    xa = 5., #actuator rope attachment point x-location [m]
    ya = 1., #actuator rope attachment point y-location [m]
)
"""

LUMPED_PARAMETERS = dict(
    l1 = 3.1, #Length of link 1 [m]
    l2 = 2., #Length of link 2 [m]
    m1 = 0.1, #System mass [kg] (attached at the elbow)
    m2 = 3., #System mass [kg] (attached at the end of link 2)
    g = 9.81, #Gravitational acceleration [m/s^2]
    xa = 2., #actuator rope attachment point x-location [m]
    ya = 5, #actuator rope attachment point y-location [m]
)