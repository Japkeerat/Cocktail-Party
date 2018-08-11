[x1, Fs1] = audioread('mix1.wav'); %Reading audio files
[x2, Fs2] = audioread('mix2.wav');
x = [x1, x2]';
y = sqrtm(inv(cov(x')))*(x-repmat(mean(x,2),1,size(x,2)));
[W,s,v] = svd((repmat(sum(y.*y,1),size(y,1),1).*y)*y'); %This line applies the machine learning algorithm on the audiofiles

a = W*x;
%W is unmixing matrix
subplot(2,2,1); plot(x1); title('mixed audio - mic 1');
subplot(2,2,2); plot(x2); title('mixed audio - mic 2');
subplot(2,2,3); plot(a(1,:), 'g'); title('unmixed wave 1');
subplot(2,2,4); plot(a(2,:),'r'); title('unmixed wave 2');
%Outputting Values and .wav files
audiowrite('refined1.wav', a(1,:), Fs1);
audiowrite('refined2.wav', a(2,:), Fs1);