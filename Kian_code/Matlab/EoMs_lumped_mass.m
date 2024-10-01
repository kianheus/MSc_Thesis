clc
% Define symbolic variables and functions
syms t q1(t) q2(t) l1 l2 J1 J2 m g xa ya

q = [q1(t); q2(t)];

%% 

% Rope connection length along joint 2
kx = l1 * cos(q1(t)) + l2 * cos(q2(t));
ky = l1 * sin(q1(t)) + l2 * sin(q2(t));

k = [kx; ky];

% Derivatives of k with respect to q1 and q2
dkdq1 = diff(k, q1(t));
dkdq2 = diff(k, q2(t));

%%

% Length of the rope (distance between actuator and kinematic point)
l = sqrt((xa - kx)^2 + (ya - ky)^2);

% Angle alpha (angle of the rope)
alpha = atan2(ya - ky, xa - kx);

% Compute A1 and A2, derivatives of l with respect to q1 and q2
A1 = simplify(diff(l, q1(t)));
A2 = simplify(diff(l, q2(t)));

% Display A1 and A2
disp('A1:');
pretty(A1)
disp('A2:');
pretty(A2)

% Define matrix A and simplify
A = simplify([A1; A2]);

% Display matrix A
disp('A matrix:');
pretty(simplify(A))

%%

% Define h1 and h2
h1 = l;              % The length of the rope
h2 = alpha;          % The angle alpha

% Define h matrix
h = [h1; h2];

% Compute Jacobian Jh of theta with respect to generalized coordinates q
Jh = jacobian(h, q);

% Simplify the Jacobian matrix
Jh = simplify(Jh);

% Display Jh
disp('Jacobian Jh:');
pretty(Jh)

% Compute the inverse and the transpose of the inverse of Jh
Jh_inv = simplify(inv(Jh));
Jh_invtrans = simplify(transpose(Jh_inv));

% Compute A_theta
A_theta = Jh_invtrans * A;

% Simplify A_theta
A_theta = simplify(A_theta);

% Display A_theta
disp('A_theta matrix:');
pretty(A_theta)

%% NOW TO FIND THE EL MATRICES

% Define positions of the endpoint of each link
x1 = l1 * cos(q1(t));  % x position of link 1
y1 = l1 * sin(q1(t));  % y position of link 1
x2 = l1 * cos(q1(t)) + l2 * cos(q2(t));  % x position of link 2
y2 = l1 * sin(q1(t)) + l2 * sin(q2(t));  % y position of link 2

% Velocities of the endpoint of each link
vx1 = diff(x1, t);
vy1 = diff(y1, t);
vx2 = diff(x2, t);
vy2 = diff(y2, t);

% Total kinetic energy
T = 0.5 * m * (vx2^2 + vy2^2);

% Total potential energy
V = m * g * y2;

% Lagrangian
L = T - V;

% Generalized coordinates and velocities
q_dot = [diff(q1, t); diff(q2, t)];
q_ddot = [diff(q_dot(1), t); diff(q_dot(2), t)];

% Replacements for Lagrangian equations
replacements = cell(2 * 3, 2);
old_vars = cell(6, 1);
new_vars = cell(6, 1);

% Populate the replacements array
for i = 1:2
    old_vars{(i-1)*3 + 1} = diff(q(i), t, 2); % 2nd derivative of q_i
    new_vars{(i-1)*3 + 1} = sym(['ddq', num2str(i)]); % Symbol for 2nd derivative

    old_vars{(i-1)*3 + 2} = diff(q(i), t); % 1st derivative of q_i
    new_vars{(i-1)*3 + 2} = sym(['dq', num2str(i)]); % Symbol for 1st derivative

    old_vars{(i-1)*3 + 3} = q(i); % q_i
    new_vars{(i-1)*3 + 3}  = sym(['q', num2str(i)]); % Symbol for q_i
end

% Substitute in L
%L = subs(L, old_vars, new_vars);

disp(L);
%%
% Lagrangian equations
eoms = cell(length(q), 1);
for i = 1:2
    dL_dqi_dot = diff(L, diff(q(i), t));
    d_dt_dL_dqi_dot = diff(dL_dqi_dot, t);
    dL_dqi = diff(L, q(i));

    eoms{i} = simplify(d_dt_dL_dqi_dot - dL_dqi);
end

eoms_subbed = subs(eoms, old_vars, new_vars);

disp(eoms_subbed);
%%

% Mass matrix (inertia matrix)
M = sym(zeros(2));    % Use symbolic zeros for M
M4C = sym(zeros(2));  % Use symbolic zeros for M4C

for i = 1:2
    for j = 1:2
        M4C(i, j) = simplify(diff(diff(T, diff(q(i), t)), diff(q(j), t)));
        M(i, j) = subs(M4C(i, j), old_vars, new_vars);
    end
end

disp(M);
%%
% Coriolis matrix
C = sym(zeros(2));
for i = 1:2
    for j = 1:2
        for k = 1:2            
            coriolis_term = 0.5 * (diff(M4C(i, j), q(k)) + diff(M4C(i, k), q(j)) ...
                - diff(M4C(j, k), q(i))) * diff(q(k), t);
            C(i, j) = C(i, j) + coriolis_term;
        end
        C(i, j) = simplify(subs(C(i, j), old_vars, new_vars));
    end
end

disp(C);
%%

% Gravitational force vector
G = sym(zeros(2, 1));
for i = 1:2
    G(i) = simplify(diff(V, q(i)));
    G(i) = subs(G(i), old_vars, new_vars);
end

% Print the matrices
disp('Mass Matrix (M):');
pretty(M);

disp('Coriolis Matrix (C):');
pretty(C);

disp('Gravitational Vector (G):');
pretty(G);

%%

M_theta = Jh_invtrans * M * Jh_inv;

disp(M_theta);

