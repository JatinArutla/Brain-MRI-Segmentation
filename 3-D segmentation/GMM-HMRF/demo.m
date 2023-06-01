%% This is a demo of 3D volume segmentation
%   Copyright by Quan Wang, 2012/12/16
%   Please cite: Quan Wang. GMM-Based Hidden Markov Random Field for 
%   Color Image and 3D Volume Segmentation. arXiv:1212.4527 [cs.CV], 2012.
clear;clc;close all;

st = load("Brain.mat");
im = st.T1;
la = st.label;
inner = st.T1;
outer = st.T1;
biggest = im;

for i = 1:10
    Laplacian=[0 1 0; 1 -4 1; 0 1 0];
    la_im=conv2(im(:,:,i), Laplacian, 'same');
    im(:,:,i) = im(:,:,i) + la_im;
end

se = strel('disk', 2);
im = imdilate(im, se);
level = multithresh(im, 1);
seg_I = imquantize(im,level);
seg_I = seg_I-1;
for i = 1:10
    biggest(:,:,i) = bwareafilt(logical(seg_I(:,:,i)), 1);
end
inner(biggest~=1) = 0;
outer(biggest~=0) = 0;

se = strel('disk', 2);
outer = imopen(outer, se);
se = strel('disk', 1);
outer = imerode(outer, se);

k=4; % k: number of regions
g=1; % g: number of GMM components
beta=1; % beta: unitary vs. pairwise
EM_iter=5; % max num of iterations
MAP_iter=5; % max num of iterations
tic
fprintf('Performing k-means segmentation\n');
[X_outer GMM_outer]=image_kmeans(outer,k,g);
X_kmeans=X_outer;
[X_outer GMM_outer]=HMRF_EM(X_outer,outer,GMM_outer,k,g,EM_iter,MAP_iter,beta);
toc

l = [];

for k = 1:length(GMM_outer)
    cont = GMM_outer{k};
    l = [l, cont.mu];
end
[Centers, sortidx] = sort(l, 'ascend');
[rows, cols, dim] = size(X_outer);
for i = 1:rows
    for j = 1:cols
        for k = 1:dim
            if X_outer(i,j,k) == sortidx(1)
                X_outer(i,j,k) = 10;
            end
            if X_outer(i,j,k) == sortidx(2)
                X_outer(i,j,k) = 11;
            end
            if X_outer(i,j,k) == sortidx(3)
                X_outer(i,j,k) = 12;
            end
            if X_outer(i,j,k) == sortidx(4)
                X_outer(i,j,k) = 13;
            end
        end
    end
end

X_outer(X_outer==10) = 0;
X_outer(X_outer==11) = 2;
X_outer(X_outer==12) = 1;
X_outer(X_outer==13) = 1;

se = strel('disk', 1);
X_outer = imerode(X_outer, se);

k=4; % k: number of region3
g=1; % g: number of GMM components
beta=1; % beta: unitary vs. pairwise
EM_iter=5; % max num of iterations
MAP_iter=5; % max num of iterations
tic
fprintf('Performing k-means segmentation\n');
[X_inner GMM_inner]=image_kmeans(inner,k,g);
X_kmeans=X_inner;
[X_inner GMM_inner]=HMRF_EM(X_inner,inner,GMM_inner,k,g,EM_iter,MAP_iter,beta);
toc

l = [];

for k = 1:length(GMM_inner)
    cont = GMM_inner{k};
    l = [l, cont.mu];
end
[Centers, sortidx] = sort(l, 'ascend');
[rows, cols, dim] = size(X_inner);
for i = 1:rows
    for j = 1:cols
        for k = 1:dim
            if X_inner(i,j,k) == sortidx(1)
                X_inner(i,j,k) = 10;
            end
            if X_inner(i,j,k) == sortidx(2)
                X_inner(i,j,k) = 11;
            end
            if X_inner(i,j,k) == sortidx(3)
                X_inner(i,j,k) = 12;
            end
            if X_inner(i,j,k) == sortidx(4)
                X_inner(i,j,k) = 13;
            end
        end
    end
end

X_inner(X_inner==10) = 0;
X_inner(X_inner==11) = 3;
X_inner(X_inner==12) = 4;
X_inner(X_inner==13) = 5;

[rows,cols, dim] = size(X_inner);
for i = 1:rows
   for j = 1:cols
       for k = 1:dim
            if X_inner(i,j,k) == 0
                X_inner(i,j,k) = X_outer(i,j,k);
            end
       end
   end
end

L = X_inner+1;
la = la+1;

jac_similarity = jaccard(double(L), double(la))
dice_similarity = dice(double(L), double(la))