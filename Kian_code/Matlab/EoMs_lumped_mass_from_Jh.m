% Clear previous definitions
clear;
clc;

% Define symbolic variables
syms P Q R S T U V A B l_1 l_2 q_1 q_2 m

% Shorthand description of Matrices, will be expanded below
Jh_inv = [P, Q; R, S];
Jh_invtrans = [P, R; Q, S];
M_q = [T, U; U, V];

% Verify that matrix multiplication works as intended
M_theta = simplify(Jh_invtrans * M_q * Jh_inv);
disp(M_theta);

%%
% Define the substitution expressions for P, Q, R, S
P_expr = (A * l_2 * cos(q_2) + B * l_2 * sin(q_2)) / (sqrt(A^2 + B^2) * l_1 * l_2 * sin(q_1 - q_2));
Q_expr = (A * l_2 * sin(q_2) - B * l_2 * cos(q_2)) / (l_1 * l_2 * sin(q_1 - q_2));
R_expr = (-A * l_1 * cos(q_1) - B * l_1 * sin(q_1)) / (sqrt(A^2 + B^2) * l_1 * l_2 * sin(q_1 - q_2));
S_expr = (-A * l_1 * sin(q_1) + B * l_1 * cos(q_1)) / (l_1 * l_2 * sin(q_1 - q_2));
T_expr = m * l_1^2;
U_expr = m * l_1 * l_2 * cos(q_1 - q_2);
V_expr = m * l_2^2;


% Perform substitution in Jh_inv, Jh_invtrans and M_q
Jh_inv = subs(Jh_inv, [P, Q, R, S], [P_expr, Q_expr, R_expr, S_expr]);
Jh_invtrans = subs(Jh_invtrans, [P, Q, R, S], [P_expr, Q_expr, R_expr, S_expr]);
M_q = subs(M_q, [T, U, V], [T_expr, U_expr, V_expr]);

% Calculate M_theta
M_theta = simplify(Jh_invtrans * M_q * Jh_inv);

% Display the simplified result
disp(M_theta);
