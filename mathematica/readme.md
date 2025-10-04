# Analytic Findings

This directory is dedicated to Wolfram Mathematica files which show a number of important properties of the double pendulum system. These are required to prove that inertial- and input decoupling is possible. Furthermore, this allows for the creation of a set of training data. 

This readme gives a short description of the notebooks.


## Find_EoMs.nb
This notebook shows that for a particular choice of new coordinates $\theta = h(q)$, inertial- and input decoupling can be obtained. These coordinates are the length of the actuator tendon $(\theta_0)$, and its angle $(\theta_1)$. 

The resulting mass matrix and input matrix can be determined as follows:

$ M(\theta) = J_h^{-T}(q) M(q) J_h^{-1}(q)$

$ A(\theta) = J_h^{-T}(q) A(q)$

Where $J_h$ is the Jacobian of $h(q)$. 

## Find_phi.nb
The resulting mass matrix in $\theta$-coordinates $M(\theta)$ is inertially decoupled. However, there is a problem. The mass term of the unactuated DoF depends on the actuated DoF. This prevents a formulation in normal form which would allow non-collocated control. 

In order to overcome this problem, another coordinate change may provide a solution. This set of coordinates has to further decoupled the DoFs so that the terms in $M(\theta)$ only depend on their own DoF. 


## Normal_form.nb
This notebook shows how a formulation of the dynamics in normal form allows one to write a controller for the unactuated DoF $(\theta_1)$. #TODO: FINISH THIS FILE