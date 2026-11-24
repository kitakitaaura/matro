% Control Systems: 2nd-order transfer function step & Bode response
sys = tf(4, [1, 0.8, 4]);
[y, t] = step(sys, linspace(0, 15, 250));
[mag, phase, w] = bode(sys, logspace(-1, 2, 120));
fprintf('Step steady-state y(15s) = %.4f, peak = %.4f\n', y(end), max(y));

subplot(2, 1, 1);
plot(t, y, 'b');
title('Underdamped 2nd-Order Step Response: G(s) = 4 / (s^2 + 0.8s + 4)');
xlabel('Time (s)');
ylabel('y(t)');
grid('on');

subplot(2, 1, 2);
plot(w, 20*log10(mag), 'r');
title('Bode Magnitude (dB)');
xlabel('Frequency (rad/s)');
ylabel('Mag (dB)');
grid('on');
