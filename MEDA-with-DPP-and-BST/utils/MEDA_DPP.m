function [Acc,acc_iter,Beta,Yt_pred,Acc_dpp] = MEDA_DPP(Xs,Ys,Xt,Yt,options)
rng('default');

%% Algorithm starts here
fprintf('MEDA starts...\n');

%% Load algorithm options
if ~isfield(options,'p')
    options.p = 10;
end
if ~isfield(options,'eta')
    options.eta = 0.1;
end
if ~isfield(options,'lambda')
    options.lambda = 1.0;
end
if ~isfield(options,'rho')
    options.rho = 1.0;
end
if ~isfield(options,'T')
    options.T = 10;
end
if ~isfield(options,'d')
    options.d = 20;
end
if ~isfield(options,'gpu')
    options.gpu = 0;
end
if ~isfield(options,'classifier')
    options.classifier = '1NN';
end
if ~isfield(options,'refined')
    options.refined = 1;
end

if options.GFK
    % Manifold feature learning
    [Xs_new,Xt_new,~] = GFK_Map(Xs,Xt,options.d);
    Xs = double(Xs_new');
    Xt = double(Xt_new');
else
    Xs=Xs';Xt=Xt';
end
X = [Xs,Xt];
n = size(Xs,2);
m = size(Xt,2);
C = length(unique(Ys));
acc_iter = [];
YY = [];
for c = 1 : C
    YY = [YY,Ys==c];
end
YY = [YY;zeros(m,C)];

%% Data normalization
X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));

%% Construct graph Laplacian
if options.rho > 0
    manifold.k = options.p;
    manifold.Metric = 'Cosine';
    manifold.NeighborMode = 'KNN';
    manifold.WeightMode = 'Cosine';
    W = lapgraph(X',manifold);
    Dw = diag(sparse(sqrt(1 ./ sum(W))));
    L = eye(n + m) - Dw * W * Dw;
else
    L = 0;
end

% % initialize Gt
if strcmp(options.classifier,'1NN')
    if options.gpu
        X = gpuArray(X);
    end
    knn_model = fitcknn(X(:,1:n)',Ys,'NumNeighbors',1);
    Cls = knn_model.predict(X(:,n + 1:end)');
    Yt0 = Cls;
    Acc = length(find(Yt == Cls))/m;
end


% Construct kernel
K = kernel_meda('rbf',X,sqrt(sum(sum(X .^ 2).^0.5)/(n + m)));
E = diag(sparse([ones(n,1);zeros(m,1)]));


for c = reshape(unique(Ys),1,C)
    Es(find(Ys==c),c) = 1/length(find(Ys==c));
end
EE = [Es;zeros(m,C)];
V = blkdiag(zeros(n,n),eye(m));
Gt = full(sparse(1:m,Cls,1));
if size(Gt,2) < C
    Gt = [Gt,zeros(nt,C-size(Gt,2))];
end
G = [zeros(n,C);Gt];

T = YY(1:n,:)';
B = 2 * YY(1:n,:)' - ones(C, n);
Beta = zeros( n+m,C);
MM = ones(C, n);


% construct DPP matrix R
if options.gamma > 0
    [R,Es] = DPP(Xs,Xt,Ys,Yt0);
else
    R = 0;
end


for t = 1 : options.T
    % % Construct MMD matrix
    e = [1 / n * ones(n,1); -1 / m * ones(m,1)];
    M = e * e' * length(unique(Ys));
    N = 0;
    for c = reshape(unique(Ys),1,length(unique(Ys)))
        e = zeros(n + m,1);
        e(Ys == c) = 1 / length(find(Ys == c));
        e(n + find(Cls == c)) = -1 / length(find(Cls == c));
        e(isinf(e)) = 0;
        N = N + e * e';
    end
    M = M+N;
    M = M / norm(M,'fro');



    % Compute coefficients vector Beta
    Beta = ((E  + options.gamma * R + options.rho * L + options.lambda* M  ) * K  + options.eta * speye(n + m,n + m)) \ (E * [T,zeros(C,m)]');

 
    if options.gamma>0
        Z = Beta'*K;
        F = Z(:,1:n)*Es;
        Zt = Z(:,n+1:end);
        for j = 1:m
            Yt0(j) = searchBestIndicator(Zt(:,j),F,C);
        end
    end


    %update T
    if options.epsilon > 0
        RT = YY(1:n,:)' + B .* MM;
        T = (1+options.epsilon ) \ (Beta' * K(:,1:n) + options.epsilon * RT );


        %update M
        S = T - YY(1:n,:)';
        Mtemp = B .* S;
        MM = max(Mtemp, 0);
    else
        T = YY(1:n,:)';
    end
    FF = K * Beta;
    [~,Cls] = max(FF,[],2);


    if options.refined == 1
        logicYtpredict = ((Cls(n+1:end)==Yt0));
        ind = find(logicYtpredict==1);
        indcmp = find(~logicYtpredict);
        Mdl = fitcknn(Xt(:,ind)',Yt0(ind));
        Yt0cmp = Mdl.predict(Xt(:,indcmp)');
        Yt0(indcmp) = Yt0cmp;
        Cls(n+1:end) = Yt0;
    end


    %% Compute accuracy
    Acc = numel(find(Cls(n+1:end)==Yt)) / m;
    Acc_dpp =  numel(find(Yt0==Yt)) / m;
    Cls = Cls(n+1:end);
    acc_iter = [acc_iter;Acc];
    fprintf('Iteration:[%02d]>>Acc=%f\n',t,Acc_dpp);
end
Yt_pred = Cls;
fprintf('MEDA ends!\n');
end

function K = kernel_meda(ker,X,sigma)
switch ker
    case 'linear'
        K = X' * X;
    case 'rbf'
        n1sq = sum(X.^2,1);
        n1 = size(X,2);
        D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        K = exp(-D/(2*sigma^2));
    case 'sam'
        D = X'*X;
        K = exp(-acos(D).^2/(2*sigma^2));
    otherwise
        error(['Unsupported kernel ' ker])
end
end