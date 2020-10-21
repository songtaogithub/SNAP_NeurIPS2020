% experiment with NMF
clear, clc
rng(10)
% params
addpath('./lib')
M = 20; N = 50; R = 10; percent = 0.015;
[X, W_r, H_r] = genData(M, N, R, percent);
nmf_problem = NMF(X, R);
proj = @(x)(max(x, 0));
%params
eps_g = 2e-2;
T = 2.5e4;
L1 = norm(X, 'fro')^2;
alpha = 1e-2; % TODO
scaling = 1e-5 ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mnrm = norm(nmf_problem.X,'fro');
W0 = max(0, randn(M, R));
H0 = max(0, randn(N, R));
W0 = scaling*W0/norm(W0,'fro')*sqrt(Mnrm);
H0 = scaling*H0/norm(H0,'fro')*sqrt(Mnrm);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Algorithms
[time_nc, cost_nc,use_eig]           = ppgd_eig([W0; H0], nmf_problem, proj, eps_g, alpha, T);
[time_neon, cost_neon, use_eig_neon] = pnc_neon2([W0; H0], nmf_problem, proj, eps_g, alpha, T);
[time_pgd, x, cost_pgd]              = pgd([W0; H0], nmf_problem, proj, alpha, 3*T, eps_g);
[time_bt, x_bt, cost_bt]             = pgd_bt([W0; H0], nmf_problem, proj, alpha, T, eps_g);
[time_agd, x_agd, cost_agd]              = agd([W0; H0], nmf_problem, proj, 2*alpha, 2*T, eps_g);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ss = step_size(~isnan(step_size));
% cost_ppgd_pgd_rel = cost_ppgd_pgd / Mnrm;
% vmax =  max([cost_ppgd_eig_rel, cost_ppgd_pgd_rel, cost_pgd_rel]);
% vmin =  min([cost_ppgd_eig_rel, cost_ppgd_pgd_rel, cost_pgd_rel]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = figure();
%cmap = colormap('lines');
% subplot(1, 2, 1)
semilogy(cost_nc, '-ro', 'linewidth', 2, 'MarkerIndices', [T/20:T/10:T]);
hold on;
semilogy(cost_neon, '-.s','color',rgb('orange'), 'linewidth', 2, 'MarkerIndices', [T/20:T/20:T]);
hold on
semilogy(cost_pgd, '--bd', 'linewidth', 2, 'MarkerIndices', [T/20:T/10:T]);
hold on;
semilogy(cost_bt, '-.m^', 'linewidth', 2, 'MarkerIndices', [T/20:T/10:T]);
hold on;
semilogy(cost_agd, '-.m>','color',rgb('Brown'), 'linewidth', 2, 'MarkerIndices', [T/20:T/10:T]);
% ylim([vmin, vmax])
xlabel('Iteration')
ylabel('Loss value')
legend('NPGD', 'NEON','PGD','PGD bt','alt-min')
axis([0 2.5e4 1e-25 1e3])
grid on
% legend([h1, h2, h3 , h4], {'PPGD-BT', 'PGD-BT', 'PPGD', 'PGD'})
% add verticle line
% neg_its = find(use_eig); % iteration that used neg-curvature
% for i = 1: min(length(neg_its), 5) % show at most 5 lines to avoid clutter
%     h = semilogy([neg_its(i), neg_its(i)], [vmin, vmax], '--', 'linewidth', 2, 'color', cmap(4, :));
%     set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% end
%savename = strcat('PGD_M=', num2str(M), '_N=', num2str(N), '_R=', num2str(R), '_pct=', num2str(percent), '.fig');
%savefig(f, savename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = figure();
cmap = colormap('lines');
% subplot(1, 2, 1)
semilogy(time_nc,cost_nc, '-r', 'linewidth', 2);
hold on;
semilogy(time_neon,cost_neon, '-.','color',rgb('orange'), 'linewidth', 2);
hold on
semilogy(time_pgd,cost_pgd, '--b', 'linewidth', 2);
hold on;
semilogy(time_bt,cost_bt, '-.m', 'linewidth', 2);
hold on;
semilogy(time_agd, cost_agd, '-.','color',rgb('Brown'), 'linewidth', 2);
% ylim([vmin, vmax])
xlabel('Run time')
ylabel('Loss value')
legend('NPGD', 'NEON','PGD','PGD bt','alt-min')
grid on
axis([0 6 1e-25 1e3])
%legend([h1, h2, h3 , h4], {'PPGD-BT', 'PGD-BT', 'PPGD', 'PGD'})
% add verticle line
% neg_its = find(use_eig); % iteration that used neg-curvature
% for i = 1: min(length(neg_its), 5) % show at most 5 lines to avoid clutter
%     h = semilogy([neg_its(i), neg_its(i)], [vmin, vmax], '--', 'linewidth', 2, 'color', cmap(4, :));
%     set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% end
% savename = strcat('PGD_M=', num2str(M), '_N=', num2str(N), '_R=', num2str(R), '_pct=', num2str(percent), '.fig');
% savefig(f, savename)

