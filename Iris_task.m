% Define the parameters
amplitude = 1;          % Amplitude of the sine wave
frequency = 1;          % Frequency of the sine wave (in Hz)
phase = 0;              % Phase shift of the sine wave (in radians)
sampling_rate = 1000;   % Sampling rate (in Hz)
duration = 2;           % Duration of the signal (in seconds)

% Generate time vector
t = linspace(0, duration, duration * sampling_rate);

% Generate the sine wave
sine_wave = amplitude * sin(2 * pi * frequency * t + phase);

% Plot the sine wave
plot(t, sine_wave);
title('Sine Wave');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;