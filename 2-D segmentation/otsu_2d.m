st = load("Brain.mat");
images = st.T1;
labels = st.label;

jac_sum = 0;
dice_sum = 0;

sensitivity_sum = 0;
specificity_sum = 0;

for w = 1:6

    im = images(:,:,w);
    t = images(:,:,w);
    inner = images(:,:,w);
    outer = images(:,:,w);
    la = labels(:,:,w);
    
    Laplacian=[0 1 0; 1 -4 1; 0 1 0];
    la_im=conv2(im, Laplacian, 'same');
    im = im + la_im ;

    I_filt = medfilt2(im, [3 3]);
    
    % Compute the gradient magnitude of the filtered image
    gradmag = imgradient(I_filt);
    
    % Threshold the gradient magnitude to obtain the marker image
    thresh = multithresh(gradmag, 5);
    marker = imquantize(gradmag, thresh);
    marker = marker - 1;
    im = im + marker;
    
    se = strel('disk', 2);
    im = imdilate(im, se);
    level = multithresh(im, 1);
    seg_I = imquantize(im,level);

    seg_I = seg_I-1;
    biggest = bwareafilt(logical(seg_I), 1);
    inner(biggest~=1) = 0;
    outer(biggest~=0) = 0;


    se = strel('disk', 2);
    outer = imopen(outer, se);
    se = strel('disk', 1);
    outer = imerode(outer, se);

    ax1 = subplot(2,2,1);
    imshow(double(inner), [])
    title('Inner region')
    ax2 = subplot(2,2,2);
    imshow(double(outer), [])
    title('Outer region')

    
    % Outer part
  
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
    
    [rows,cols] = size(L);
    for i = 1:rows
       for j = 1:cols
           if L(i,j) == 0
               L(i,j) = L_outer(i,j);
           end
       end
    end
    
    L = imfill(L, 8);
    ax1 = subplot(2,2,1);
    imagesc(double(t))
    title('Image')
    ax2 = subplot(2,2,2);
    imagesc(double(L))
    title('Otsu Segmented Image')
    ax3 = subplot(2,2,3);
    imagesc(double(la))
    title('Ground Truth')
    
    L = L+1;
    la = la+1;
    
    jac_similarity = jaccard(double(L), double(la));
    dice_similarity = dice(double(L), double(la));
    temp_jac = 0;
    temp_dice = 0;

    for p = 1:size(jac_similarity)
        temp_jac = temp_jac + jac_similarity(p);
        temp_dice = temp_dice + dice_similarity(p);
    end
    temp_jac = temp_jac / 6;
    temp_dice = temp_dice / 6;
    jac_sum = jac_sum + temp_jac;
    dice_sum = dice_sum + temp_dice;
    cp = classperf(double(la), double(L));
    sensitivity_sum = sensitivity_sum + cp.Sensitivity;
    specificity_sum = specificity_sum + cp.Specificity;
end

sensitivity = sensitivity_sum / 6
specificity = specificity_sum / 6
final_jac = jac_sum / 6
final_dice = dice_sum / 6