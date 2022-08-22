close all
clc


data = get_emeka_inits();
[q_mat,~,con_graph] = get_q_alphas_connectivity(data.location,data.Power_Exponent,data.Power_Threshold, data.TX_Power(1));
alpha_mat = gen_random_softmax_weights(con_graph);
t_MATLAB = Run_Sim_FD(data.sim_length, data.eps(1), alpha_mat , q_mat, data.Initial_Period, data.Initial_Time);

Plot_Sim(t_MATLAB, "MATLAB Simulated " + "FD" , "mat_sim.png")
function alpha_mat = gen_random_softmax_weights(connectivity_mat)
    alpha_mat = zeros(size(connectivity_mat));
    for i = 1 : size(alpha_mat, 1)
        for j = 1 : size(alpha_mat, 2)
            if connectivity_mat(i, j) == 1
                alpha_mat(i, j) = abs(randn(1));
            else
                alpha_mat(i, j) = -99999999;
            end
        end
    end
    alpha_mat = softmax(alpha_mat);
end

function Plot_Sim(t, custom_title, filename)
a = 1;
figure()
subplot(221)
plot(diff(t'));
periods_last_mean = mean(t(:,end) - t(:,end-1));
title("Periods, Final sample standard deviation is: "+string(sqrt(var(diff(t(end-5:end-4,:))))))
xlabel("n")
ylabel("\Delta" + "t")
xlim([1 100])

subplot(222)
plot(diff(t'));
var_num = sqrt(var(diff(t(end-5:end-4,:))));
title("Periods, Final sample standard deviation is: "+string(var_num))
xlabel("n")
ylabel("\Delta" + "t")
xlim([2500 2600])
ylim([periods_last_mean-var_num^2 periods_last_mean+var_num^2])



subplot(223)
periods_last_mean = mean(t(:,end) - t(:,end-1));
V = mod((t'), periods_last_mean) / periods_last_mean * 100;
plot(V);
D = string(sqrt(var(V(end,:))));
title("Phases, calculated by \Phi = Mod(t, \Deltat[end]) / \Deltat[end] * 100 %", " Final sample standard deviation is: " + D + "%")
xlabel("n")
ylabel("\Phi = Mod(t, \Deltat[end])")
xlim([1 100])
simple_set_yticklabels("%")

subplot(224)
periods_last_mean = mean(t(:,end) - t(:,end-1));
V = mod((t'), periods_last_mean) / periods_last_mean * 100;
plot(V);
D = string(sqrt(var(V(end,:))));
title("Phases, calculated by \Phi = Mod(t, \Deltat[end]) / \Deltat[end] * 100 %", " Final sample standard deviation is: " + D + "%")
xlabel("n")
ylabel("\Phi = Mod(t, \Deltat[end])")
xlim([2500 2600])
simple_set_yticklabels("%")

cur_fig = gcf;
cur_fig.Name = custom_title;
exportgraphics(cur_fig, filename,'Resolution',300);

end

function plot_loss(loss_mat, filename)
figure()
cur_fig = gcf;
cur_fig.Name = "Loss of learning process";
plot(loss_mat')
xlabel("iterations")
ylabel("Loss")
title("Loss vector of the process")
exportgraphics(cur_fig, filename,'Resolution',300);
end

function t = Run_Sim_TDMA(sig_length, eps_0, alpha_mat, q_mat, T_vec, t0_vec)
N = size(alpha_mat, 2);
t = zeros(N, sig_length);
t_update = zeros(N, N);
t(:,1) = t0_vec;
for n = 2 : sig_length
    % Transmitter setting
    transmitter_idx = mod(n, N) + 1; % Matlab indexing system. (0->15) mapped to (1->16)

    % Saves loop - This is before the Updates loop to write less
    % redundant code.

    for i = 1 : N
        t_update(i, transmitter_idx) = q_mat(i, transmitter_idx) + t(transmitter_idx, n-1) - t(i,n-1);
    end

    % Updates loop

    for i = 1 : N
        t(i,n)   =  t(i, n - 1) + T_vec(i); % Self - Update
        for j = 1 : N % This is just a for-loop sum.
            t(i, n) = t(i, n) + eps_0 * alpha_mat(i, j) * t_update(i, j);
        end
    end

end

end

function t = Run_Sim_FD(sig_length, eps_0, alpha_mat, q_mat, T_vec, t0_vec)
N = size(alpha_mat, 2);
t = zeros(N, sig_length);
t_update = zeros(N, N);
t(:,1) = t0_vec;
for n = 2 : sig_length
    % Transmitter setting
    %         transmitter_idx = mod(n, N) + 1; % Matlab indexing system. (0->15) mapped to (1->16)

    % Saves loop - This is before the Updates loop to write less
    % redundant code.
    for transmitter_idx = 1 : N
        for i = 1 : N
            t_update(i, transmitter_idx) = q_mat(i, transmitter_idx) + t(transmitter_idx, n-1) - t(i,n-1);
        end
    end
    % Updates loop

    for i = 1 : N
        t(i,n)   =  t(i, n - 1) + T_vec(i); % Self - Update
        for j = 1 : N % This is just a for-loop sum.
            t(i, n) = t(i, n) + eps_0 * alpha_mat(i, j) * t_update(i, j);
        end
    end

end

end

function distance_mat = calc_distances(xy_mat)
N = size(xy_mat,1);
distance_mat = zeros(N);
for i = 1 : N
    for j = 1 :N
        distance_mat(i,j) = sqrt ( (xy_mat(i,1) - xy_mat(j,1))^2 + ( xy_mat(i,2) - xy_mat(j,2) )^2 );
    end
end

end

function qs_mat    = calc_qs    (distance_mat)
N = size(distance_mat,2);
qs_mat = zeros(N);
c = 3 * 10 ^ 8;
for i = 1 : N
    for j = 1 : N
        qs_mat(i,j) = distance_mat(i,j) / c ;
    end
end
end

function alpha_mat = calc_alphas(con_graph, distance_mat, power_exponent, p0)
N = size(con_graph,2);
alpha_mat = zeros(N);
neigh_pow = zeros(1,N);
for i = 1 : N
    neigh_pow(i) = power_sum(i,con_graph,distance_mat,power_exponent, p0); % No need to calculate multiple times.
    for j = 1 : N
        if con_graph(i,j) == 1
            alpha_mat(i,j) = p0 * distance_mat(i,j)^(-power_exponent) / neigh_pow(i) ;
        end
    end
end
end

function out_sum = power_sum(i,con_graph,distance_mat,power_exponent, p0)
out_sum = 0;
N = size(con_graph, 2);
for k = 1 : N
    if con_graph(i,k) == 1
        out_sum = out_sum + p0 * distance_mat(i,k) ^ (-power_exponent);
    end
end
end

function [q_mat,alpha_mat,con_graph] = get_q_alphas_connectivity(xy,power_exponent,power_threshold, p0)
N = size(xy,1);
con_graph = ones(N) - eye(N);
distance_mat = calc_distances(xy);
for i = 1 : N
    for j = 1 : N
        if con_graph(i,j) == 1
            if p0 * distance_mat(i,j)^(-power_exponent) < power_threshold
                con_graph(i,j) = 0;
            end
        end
    end
end
q_mat = calc_qs(distance_mat);
alpha_mat = calc_alphas(con_graph, distance_mat, power_exponent, p0);
end

function simple_set_yticklabels(str)
yticklabels_new = yticklabels;
for i = 1 : length(yticklabels_new)
    yticklabels_new{i} = yticklabels_new{i} + str;
end
yticklabels(yticklabels_new)
end

function data = get_emeka_inits()
data = struct;
data.N = 16;
data.TX_Power = [2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2];
data.eps = [0.3; 0.3; 0.3; 0.3; 0.3; 0.3; 0.3; 0.3; 0.3; 0.3; 0.3; 0.3; ...
            0.3; 0.3; 0.3; 0.3];
data.Initial_Period = [0.0049997777678072453; 0.0049998555332422256; 0.005000249482691288; ...
                       0.0050004320219159126; 0.0050001917406916618; 0.0050000068731606007; ...
                       0.0050000175833702087; 0.0049999835900962353; 0.0050000441260635853; ...
                       0.0049998373724520206; 0.0049999658949673176; 0.0049995705485343933; ...
                       0.0050002639181911945; 0.0050001638010144234; 0.0049999882467091084; ...
                       0.0050000129267573357];
data.Initial_Time = [0.00017011463933158666; 0.0047212308272719383; 0.0044008991681039333; ...
                     6.1801074480172247E-6; 0.0029679301660507917; 0.0020788498222827911; ...
                     0.0020885970443487167; 0.0013556077610701323; 0.0034613902680575848; ...
                     0.0010192411718890071; 0.0034164781682193279; 0.003764270106330514; ...
                     0.0042896787635982037; 0.0034347777254879475; 2.5661885956651531E-5; ...
                     0.00087825773516669869];
data.location = ...
  [2782.52490234375 7744.8603515625;
   4819.58837890625 4368.9130859375;
   8197.8037109375 5190.90771484375;
   9970.666015625 6158.5234375;
   6984.41064453125 8101.8828125;
   5675.46435546875 9800.9697265625;
   8352.431640625 1146.8822021484375;
   2055.98828125 3167.651123046875;
   5931.72021484375 6965.0498046875;
   1123.472412109375 9142.7470703125;
   1534.5692138671875 9351.0361328125;
   2417.082275390625 9411.7841796875;
   7262.365234375 5995.07275390625;
   7010.80224609375 652.08673095703125;
   2038.237548828125 5459.96240234375;
   6510.53515625 1871.9732666015625];
data.x_max = 10000;
data.y_max = 10000;
data.sim_length = 2810;
data.Data_ACQ_Time = 16;
data.Power_Threshold = 3.9810717055E-15;
data.Power_Exponent = 4;
data.Speed_Of_Light = 3.0E+8;
end