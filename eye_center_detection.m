detector=vision.CascadeObjectDetector('LeftEyeCART');
input_image=imread('test4.jpg');
gauss_image=imgaussfilt(input_image,1);
eyes=step(detector,gauss_image);

size_eyes = size(eyes);
imshow(input_image);
hold on;
for i =1:size_eyes(1);
    eyes_final=eyes(i,1:2);
    image = imcrop(gauss_image,eyes(i,:));
    %IEyes = insertObjectAnnotation(image, 'rectangle', eyes, 'eye');
    %figure, imshow(IEyes);
    
    image = rgb2gray(image);
    image = im2double(image);
    [grad_x,grad_y] = gradient(image);
    
    g = grad_x.*grad_x + grad_y.*grad_y;
    g = sqrt(g);
    
    std_g = std(g(:));
    mean_g = mean(g(:));
    
    g_1 = g;
    g_1(g_1 < mean_g + 0.5*std_g) = 0;
    
    g_2 = g;
    g_2(g_2 > mean_g - 0.5*std_g) = 0;
    
    g = g_1 + g_2;
    temp_g = g;
    w = 1-image;
    temp_g(temp_g ~= 0) = 1;
    grad_x = grad_x.*temp_g;
    grad_y = grad_y.*temp_g;
    
    size_img = size(image);
    Dx = repmat(1:size_img(2),size_img(1),1);
    Dy = repmat(transpose(1:size_img(1)),1,size_img(2));
    C_max = 0;
    X = 0;
    Y = 0;
    for x = 1:size_img(2)
        for y = 1:size_img(1)
            D_x = Dx - x;
            D_y = Dy - y;
            
            D_t = D_x.*D_x + D_y.*D_y;
            E = sqrt(D_t);
            
            D_x = D_x./E;
            D_y = D_y./E;
            
            C_x = D_x.*grad_x;
            C_y = D_y.*grad_y;
            
            
            C = w(y,x)*(C_x + C_y);
            C(C < 0) = 0;
            ind = find(isnan(C));
            C(ind)=0;
            % C(isNaN(C)) = 0;
            
            total_C = sum(sum(C));
            
            if(total_C > C_max)
                C_max = total_C;
                X = x;
                Y = y;
            end
            
        end
    end
    plot(X+eyes_final(1),Y+eyes_final(2),'r+','MarkerSize',10);
end



