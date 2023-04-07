Nt = 16;
Nr = 16;
CYCLE = 5000;
snr = 100;
y_data = zeros(CYCLE, 2*Nr*Nt+2);


for cycle = 1:CYCLE
    theta = rand() * pi - pi/2;
    phi = rand() * pi - pi/2;
    
    
    a_t = zeros(Nt, 1);
    a_r = zeros(Nr, 1);
    for k = 1:Nt
        a_t(k) = exp(-1i*pi*sin(theta)*(k-1));
    end
    for j = 1:Nr
        a_r(j) = exp(-1i*pi*sin(phi)*(j-1));
    end
    
    alpha = randn() + 1i * randn();
    H = sqrt(Nr*Nt) * alpha * a_r * a_t';
    
    n = randn(Nr, Nt) + 1i * randn(Nr, Nt);
    power_H = sum(abs(H).^2, "all");
    power_n = sum(abs(n).^2, "all");
    n = n / sqrt(power_n) * sqrt(power_H) /sqrt(snr);  
    power_H = sum(abs(H).^2, "all");
    power_n = sum(abs(n).^2, "all");
    y = H + n;
   
    theta = theta * 180 / pi;
    phi = phi * 180 / pi;

    y_real = real(y);
    y_img = imag(y);
    y_real = reshape(y_real, 1, []);
    y_img = reshape(y_img, 1, []);
    
    y_onepair = [y_real, y_img, theta, phi];
    y_data(cycle,:) = y_onepair;
end

writematrix(y_data, 'sample_y.csv');
figure()
imagesc(abs(y_data).^2)


