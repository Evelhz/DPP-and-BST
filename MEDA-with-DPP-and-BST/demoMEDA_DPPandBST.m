clc;clear
addpath utils
addpath ../data/
str_domains = {'Caltech10', 'amazon', 'webcam', 'dslr'};
list_acc = [];
list_acc_dpp = [];
results = [];

for i =1:4
    for j = 1:4
        if i == j
            continue;
        end
        src = str_domains{i};
        tgt = str_domains{j};
        load([ src '_SURF_L10.mat']);     % source domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
        Xs = zscore(fts,1);    clear fts
        Ys = labels;           clear labels

        load([ tgt '_SURF_L10.mat']);     % target domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
        Xt = zscore(fts,1);     clear fts
        Yt = labels;            clear labels


        % meda
        options.d = 20; % GFK
        options.p = 10; % GFK

        options.rho = 0.2;    % graph Laplacian regularization
        options.lambda = 2;   % MMD
        options.eta = 0.1;    % balanced parameter for classifier
        options.gamma = 1e-3; % dpp
        options.T = 10;       % # iterations
        options.epsilon = 10; % epsilon-dragging
        options.GFK = 1;      % using GFK or not
        options.gpu = 0;      % 1 for GPU and 0 for CPU
        options.classifier = '1NN' ; % 1NN SVM
        [~,~,~,~,Acc_dpp] = MEDA_DPP(Xs,Ys,Xt,Yt,options);
        list_acc_dpp = [list_acc_dpp Acc_dpp*100];
        results = [results,Acc_dpp*100];
        fprintf('%s --> %s: %.2f accuracy \n\n', src, tgt, Acc_dpp * 100);
    end
end
fprintf('mean accuracy of MEDA with DPP and BST is: %.2f%% \n',mean(list_acc_dpp));
