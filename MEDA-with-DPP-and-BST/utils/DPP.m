function [R,Es] = DPP(Xs,Xt,Ys,Cls)
n = size(Xs,2);
m = size(Xt,2);
C = length(unique(Ys));
for c = reshape(unique(Ys),1,C)
    Es(find(Ys==c),c) = 1/length(find(Ys==c));
end
EE = [Es;zeros(m,C)];
V = blkdiag(zeros(n,n),eye(m));
Gt = full(sparse(1:m,Cls,1));
theta = [zeros(n,C);Gt];
EGt = EE*theta';
VmEGt = V-EGt;
R = (VmEGt)*VmEGt';
R = R/norm(R,'fro');