%=================================================================
% Simulation of the KronigPenneyBandStructure
% Electron in a periodic potential with rectangular barriers
% Includes Real and Imaginary band structures
% V0 = 3 eV, Lb = 0.1*e-9 m, Lw = 0.9*e-9 m

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
% Generating the KP Model 
%=================================================================

%------------------ Input parameters --------------------------%

npoints = 1750;			%number of points in energy plot 1750
hbar = 1.0545715e-34;		%Planck's constant (Js)
echarge = 1.6021764e-19;	%electron charge (C)
m0 = 9.109382e-31;		%bare electron mass (kg)

V0eV = 3.0;               	%barrier energy [eV] 1.0
V0 = V0eV*echarge;        	%barrier energy [J]
Lb = 0.1e-9;              	%barrier thickness [m] 0.4
Lw = 0.9e-9;              	%well thickness [m] 0.6
L = Lb+Lw;                	%cell period [m]

%---------------- The necessary k-vectors ---------------------%

k1 = linspace(0,3.5*pi/L,npoints);
%range of k-values considered determines E

E = (k1.^2)*(hbar^2)/(2*m0);     %electron energy, E [J]
EeV = E/echarge;max(EeV)
k2 = sqrt(2*m0*(E-V0))/hbar; 	%k2 real for E<V0

k12=k1.^2;
k22=k2.^2;
Aconst=(k12+k22)./(2*k1.*k2);

theta=(cos(k2*Lb).*cos(k1*Lw))-(Aconst.*sin(k2*Lb).*sin(k1*Lw));

%--------------- Initializing Band Structures------------------%

kLre=zeros(npoints);    %initialize real band structure
kLim=zeros(npoints);    %initialize imaginary band structure

if any(abs(theta) < 1.0)    
    kLre=real((acos(theta))/pi);
end

if any(abs(theta) >= 1.0)   
    kLim=real((acosh(theta))/pi);
end

%---------------- Plotting the Band Structure ------------------%

ttl=['V_0 = ',num2str(V0eV,'%3.1f'),' eV, L_b = ',...
 num2str(Lb*1e9,'%3.1f'),' nm, L_w = ',...
 num2str(Lw*1e9,'%3.1f'),' nm, L = ',num2str((Lw+Lb)*1e9,'%3.1f'),' nm'];

figure(1)
plot(kLim, EeV, 'r','LineWidth',LW);
hold on;

plot(kLre, EeV, 'b','LineWidth',LW);
axis([0,1.0,0,max(EeV)]);%/3 /5
grid on

xlabel('Wave vector, kL/\pi');
%yttl=['Energy, E (',hbarChar,'^2/2m_0)'];
yttl=['E (eV)'];
ylabel(yttl);

hold off;

title (ttl);