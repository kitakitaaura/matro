% Butterworth zero-phase filtering + Welch PSD + Bode response
fs = 200;
t = (0:399) / fs;
clean_ref = sin(2*pi*5*t);
noisy = clean_ref + 0.75 * sin(2*pi*65*t);

[b, a] = butter(4, 20 / (fs / 2));
filtered = filtfilt(b, a, noisy);
fprintf('Relative RMS error after zero-phase Butterworth: %.4f\n', ...
        norm(filtered(40:360) - clean_ref(40:360)) / norm(clean_ref(40:360)));

[pxx, f] = pwelch(noisy, 128, 64, 256, fs);
[pks, locs] = findpeaks(10*log10(pxx + eps), 'MinPeakHeight', -15);
fprintf('Detected spectral peaks (Hz): ');
disp(f(locs)');

subplot(2, 1, 1);
plot(t(1:160), noisy(1:160), 'r--');
hold('on');
plot(t(1:160), filtered(1:160), 'b');
legend('noisy', 'filtfilt (4th-order Butterworth)');
title('Zero-Phase Butterworth Low-Pass Filter');
ylabel('Amplitude');
grid('on');

subplot(2, 1, 2);
plot(f, 10*log10(pxx + eps), 'b');
title('Welch Power Spectral Density (pwelch)');
xlabel('Frequency (Hz)');
ylabel('PSD (dB/Hz)');
grid('on');
