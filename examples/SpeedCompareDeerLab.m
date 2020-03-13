tic
[traw,Vraw] = deerload('Example_DEER.DTA');

traw = traw/1000; %ns -> us

r = time2dist(traw);
%Optimization & Correction of phase
V = correctphase(Vraw);
%Optimization & Correctionof zero-time
t = correctzerotime(V,traw);
%Optimization & Correction of Y-axis scale
V = correctscale(V,t);

[B,lambda, param] = fitbackground(V,t,@td_strexp); 
disp(param)
KB = dipolarkernel(t,r,lambda,B);

Pfit = fitregmodel(V,KB,r,'tikhonov','aic');

Vfit = KB*Pfit;
toc

figure(1)
plot(traw, Vfit, traw, V, traw,(1 - lambda)* B)

figure(2)
plot(r, Pfit)