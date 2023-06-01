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

level = multithresh(outer, 3);
L_outer = imquantize(outer,level);
L_outer = L_outer-1;
L_outer(L_outer==0) = 10;
L_outer(L_outer==1) = 11;
L_outer(L_outer==2) = 12;
L_outer(L_outer==3) = 13;
L_outer(L_outer==10) = 0;
L_outer(L_outer==11) = 2;
L_outer(L_outer==12) = 1;
L_outer(L_outer==13) = 1;

se = strel('disk', 1);
L_outer = imerode(L_outer, se);

% Inner part
level = multithresh(inner, 3);
L = imquantize(inner,level);
L = L-1;
L(L==0) = 10;
L(L==1) = 13;
L(L==2) = 14;
L(L==3) = 15;
L(L==10) = 0;
L(L==13) = 3;
L(L==14) = 4;
L(L==15) = 5;

size(L)

[rows,cols, dim] = size(L);
for i = 1:rows
   for j = 1:cols
       for k = 1:dim
            if L(i,j,k) == 0
                L(i,j,k) = L_outer(i,j,k);
            end
       end
   end
end

L = imfill(L, 8);

L = L+1;
la = la+1;

jac_similarity = jaccard(double(L), double(la))
dice_similarity = dice(double(L), double(la))