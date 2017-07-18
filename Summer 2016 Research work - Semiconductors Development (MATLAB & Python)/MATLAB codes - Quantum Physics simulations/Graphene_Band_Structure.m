%=================================================================
% Simulation of the Band Structure of Graphene
%=================================================================

clear all
clf;		

%-------------------- Plot Paramaters ---------------------------%

FS      = 18;	%label fontsize
FSN     = 16;	%number fontsize
LW      = 2;	%linewidth

set(0,'DefaultAxesFontName', 'Arial');
set(0,'DefaultAxesFontSize', FSN);

set(0,'DefaultTextFontname', 'Arial');
set(0,'DefaultTextFontSize', FSN);

hbarChar=['\fontname{MT Extra}h\fontname{Arial}'];

	%=================================================================
% Generating the E-k diagram of Graphene
%=================================================================

%------------------ Input parameters --------------------------%

npoints = 1750;			%number of points in energy plot 1750
hbar = 1.0545715e-34;		%Planck's constant (Js)
echarge = 1.6021764e-19;	%electron charge (C)
m0 = 9.109382e-31;		%bare electron mass (kg)

E0 = 0; 				% coloumb integral
V = -2.7; 				% hopping integral [eV]
acc = 1.41; 			% c-c bond length [Angstrom]
lattice = acc*sqrt(3); 	% Lattice constant

%---------------- The necessary k-vectors ---------------------%

k_vec_x = linspace(-2*pi/lattice,2*pi/lattice,100);
k_vec_y = linspace(-2*pi/lattice,2*pi/lattice,100);
[k_mesh_x,k_mesh_y] = meshgrid(k_vec_x, k_vec_y);

%------- Energy values with the preset parameters  ------------%
 
energy_mesh = NaN([size(k_mesh_x,1),size(k_mesh_y,2),2]);

for a = 1 : size(k_mesh_x,1)

    energy_mesh(:,a,1) = (E0 + V*sqrt(1 + ...
        (4.*((cos(k_mesh_y(:,a)/2*lattice)).^2)) + ...
        (4.*(cos(sqrt(3)/2*lattice*k_mesh_x(:,a))).*...
        (cos(k_mesh_y(:,a)/2*lattice)))));

    energy_mesh(:,a,2) = (E0 - V*sqrt(1 + ...
        (4.*((cos(k_mesh_y(:,a)/2*lattice)).^2)) + ...
        (4.*(cos(sqrt(3)/2*lattice*k_mesh_x(:,a))).*...
        (cos(k_mesh_y(:,a)/2*lattice)))));

end

%------------------------ Plotting -----------------------------%

surf(k_mesh_x, k_mesh_y, real(energy_mesh(:,:,1)));

hold on

surf(k_mesh_x, k_mesh_y, real(energy_mesh(:,:,2)));
colormap('jet');
shading interp
hx = xlabel('k_x');
hy = ylabel('k_y');
hz = zlabel('E(k)');
% axis equal

hold off