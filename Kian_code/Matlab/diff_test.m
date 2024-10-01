% Clear previous definitions
clear;
clc;

% Define time as a symbolic variable
syms t

% Define q1 and q2 as functions of time
syms q1(t) q2(t)  % Define q1 and q2 as time-dependent functions

% Create the vector of generalized coordinates
q = [q1; q2];  % Create a vector containing q1(t) and q2(t)

% To access the derivatives correctly, explicitly evaluate as functions of time
% Call them with time explicitly
dq1_dt = diff(q1(t), t);  % Computes diff(q1(t), t)
dq2_dt = diff(q2(t), t);  % Computes diff(q2(t), t)

% Display the derivatives
disp('dq1_dt:');
disp(dq1_dt);  % Should output: diff(q1(t), t)
disp('dq2_dt:');
disp(dq2_dt);  % Should output: diff(q2(t), t)
