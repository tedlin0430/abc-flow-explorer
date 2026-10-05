clear; clc; close all;

%% ============================================================
% ABC flow front-speed map -- large-batch version
%
% Grid      : 400 x 400
% Section   : x0 = 0
% Domain    : (y0,z0) in [0,2*pi] x [0,2*pi]
% epsilon   : 1
% T         : 100
% Direction : x direction, e = (1,0,0)
%
% NOTE:
%   Both 0 and 2*pi are intentionally retained in the grid.
%   They are periodic copies of the same boundary, but keeping
%   both makes the displayed color map include the full boundary.
%
% Output:
%   frontspeed_N400_T100_eps_1.mat
% ============================================================

%% Parameters
L    = 2*pi;
epsi = 1;
T    = 100;
N    = 400;

x0 = 0;

% Direction used for the long-time average speed
e1 = 1;
e2 = 0;
e3 = 0;

%% Large-batch settings
batch_size  = 2000;
useParallel = true;

%% ODE settings
RelTol  = 1e-7;
AbsTol  = 1e-9;
MaxStep = 1.0;

options = odeset( ...
    'RelTol', RelTol, ...
    'AbsTol', AbsTol, ...
    'MaxStep', MaxStep, ...
    'Stats', 'off');

%% ============================================================
% 400 x 400 grid INCLUDING the boundaries 0 and 2*pi
% ============================================================
y = linspace(0, L, N);
z = linspace(0, L, N);

[Y0, Z0] = meshgrid(y, z);
X0 = x0 * ones(size(Y0));

X0_vec = X0(:);
Y0_vec = Y0(:);
Z0_vec = Z0(:);

num_points  = numel(X0_vec);
num_batches = ceil(num_points / batch_size);

fprintf('====================================================\n');
fprintf('ABC flow front-speed map: large-batch computation\n');
fprintf('====================================================\n');
fprintf('Grid             : %d x %d\n', N, N);
fprintf('Trajectories     : %d\n', num_points);
fprintf('Domain           : [0, 2*pi] x [0, 2*pi]\n');
fprintf('Boundary included: yes\n');
fprintf('x0               : %.6g\n', x0);
fprintf('epsilon          : %.6g\n', epsi);
fprintf('T                : %.6g\n', T);
fprintf('Direction        : (%.3g, %.3g, %.3g)\n', e1, e2, e3);
fprintf('Batch size       : %d\n', batch_size);
fprintf('Batches          : %d\n', num_batches);
fprintf('====================================================\n\n');

%% Allocate final states
X_final = zeros(num_points,1);
Y_final = zeros(num_points,1);
Z_final = zeros(num_points,1);

% Three requested output times keep the returned solution small.
tspan = [0, T/2, T];

%% Parallel availability
parallelAvailable = false;

if useParallel
    try
        parallelAvailable = license('test','Distrib_Computing_Toolbox');
    catch
        parallelAvailable = false;
    end
end

if parallelAvailable
    fprintf('Parallel Computing Toolbox detected: using PARFOR.\n\n');
else
    fprintf('Running serially.\n\n');
end

%% ============================================================
% Integrate all trajectories in batches
% ============================================================
tic;

if parallelAvailable

    X_batch = cell(num_batches,1);
    Y_batch = cell(num_batches,1);
    Z_batch = cell(num_batches,1);

    parfor b = 1:num_batches
        i1 = (b-1)*batch_size + 1;
        i2 = min(b*batch_size, num_points);

        [xf,yf,zf] = integrate_batch( ...
            X0_vec(i1:i2), ...
            Y0_vec(i1:i2), ...
            Z0_vec(i1:i2), ...
            tspan, options, epsi);

        X_batch{b} = xf;
        Y_batch{b} = yf;
        Z_batch{b} = zf;
    end

    for b = 1:num_batches
        i1 = (b-1)*batch_size + 1;
        i2 = min(b*batch_size, num_points);

        X_final(i1:i2) = X_batch{b};
        Y_final(i1:i2) = Y_batch{b};
        Z_final(i1:i2) = Z_batch{b};
    end

else

    for b = 1:num_batches
        i1 = (b-1)*batch_size + 1;
        i2 = min(b*batch_size, num_points);

        [xf,yf,zf] = integrate_batch( ...
            X0_vec(i1:i2), ...
            Y0_vec(i1:i2), ...
            Z0_vec(i1:i2), ...
            tspan, options, epsi);

        X_final(i1:i2) = xf;
        Y_final(i1:i2) = yf;
        Z_final(i1:i2) = zf;

        fprintf('Batch %3d / %3d completed (%6.2f%%)\n', ...
            b, num_batches, 100*b/num_batches);
    end
end

elapsed = toc;

fprintf('\nFinished in %.2f s = %.2f min\n', elapsed, elapsed/60);

%% ============================================================
% Long-time directional front speed
% ============================================================
% Do NOT wrap the final coordinates modulo 2*pi.
dX = X_final - X0_vec;
dY = Y_final - Y0_vec;
dZ = Z_final - Z0_vec;

speed_vector = (e1*dX + e2*dY + e3*dZ) / T;
speed_abc_dir = reshape(speed_vector, size(Y0));

%% ============================================================
% Save numerical map
% ============================================================
thisFile = mfilename('fullpath');

if isempty(thisFile)
    saveDir = pwd;
else
    saveDir = fileparts(thisFile);
end

matName = sprintf('frontspeed_N%d_T%d_eps_%g.mat', N, round(T), epsi);
matPath = fullfile(saveDir, matName);

save(matPath, ...
    'speed_abc_dir', ...
    'speed_vector', ...
    'y','z', ...
    'N','T','epsi','x0', ...
    'e1','e2','e3', ...
    'RelTol','AbsTol','MaxStep', ...
    '-v7.3');

fprintf('Saved map to:\n%s\n', matPath);

%% ============================================================
% Plot
% ============================================================
fig = figure( ...
    'Color','w', ...
    'Position',[180 80 820 720]);

ax = axes(fig);

imagesc(ax, y, z, speed_abc_dir);
set(ax,'YDir','normal');

axis(ax,'equal');
xlim(ax,[0 L]);
ylim(ax,[0 L]);
box(ax,'on');

colormap(ax,turbo(256));
colorbar(ax);

title(ax, ...
    sprintf('$T=%g,\\quad \\epsilon=%g$',T,epsi), ...
    'Interpreter','latex', ...
    'FontSize',18);

xlabel(ax,'Initial $y$', ...
    'Interpreter','latex', ...
    'FontSize',15);

ylabel(ax,'Initial $z$', ...
    'Interpreter','latex', ...
    'FontSize',15);

ticks = [0 pi/2 pi 3*pi/2 2*pi];
labels = {'0','\pi/2','\pi','3\pi/2','2\pi'};

xticks(ax,ticks);
yticks(ax,ticks);
xticklabels(ax,labels);
yticklabels(ax,labels);

ax.FontSize = 13;
ax.LineWidth = 1.0;

% No maximum point / maximum label is drawn.

%% ============================================================
% Local functions
% ============================================================
function [Xf,Yf,Zf] = integrate_batch(X0,Y0,Z0,tspan,options,epsi)

    n = numel(X0);
    U0 = [X0(:); Y0(:); Z0(:)];

    [~,U] = ode89( ...
        @(t,U) abc_all_trajectories(t,U,n,epsi), ...
        tspan, U0, options);

    Uf = U(end,:).';

    Xf = Uf(1:n);
    Yf = Uf(n+1:2*n);
    Zf = Uf(2*n+1:3*n);
end

function dUdt = abc_all_trajectories(~,U,n,epsi)

    X = U(1:n);
    Y = U(n+1:2*n);
    Z = U(2*n+1:3*n);

    % ABC flow convention used in the existing project:
    % dx/dt = epsilon*sin(z) + cos(y)
    % dy/dt = sin(x) + epsilon*cos(z)
    % dz/dt = sin(y) + cos(x)
    dX = epsi*sin(Z) + cos(Y);
    dY = sin(X) + epsi*cos(Z);
    dZ = sin(Y) + cos(X);

    dUdt = [dX; dY; dZ];
end
