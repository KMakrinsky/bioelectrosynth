
import numpy as np
import matplotlib.pyplot as plt

def generate_1f_noise(n_samples, alpha=1.0):
    """Generate colored noise (1/f^alpha)."""
    white = np.random.standard_normal(n_samples)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    
    with np.errstate(divide='ignore'):
        scale = 1.0 / np.power(np.abs(freqs), alpha / 2.0)
    scale[0] = 0.0
    
    fft_colored = fft_white * scale
    colored = np.fft.irfft(fft_colored, n=n_samples)
    return colored / np.std(colored)

def get_discrete_chebyshev_basis(N: int) -> np.ndarray:
    """
    Orthonormal DISCRETE Chebyshev (Gram) polynomials on x=0..N-1.

    Recurrence (fixed N):
      t0(x)=1
      t1(x)=2x-N+1
      (n+1)t_{n+1}(x) = (2n+1)(2x-N+1)t_n(x) - n(N^2-n^2)t_{n-1}(x)

    Orthonormalization:
      C[n,x] = t_n(x) / H_n
      H_n^2 = N * Π_{k=1..n}(N^2-k^2)/(2n+1)
    """
    x = np.arange(N, dtype=float)
    tnx = np.zeros((N, N), dtype=float)
    tnx[0, :] = 1.0
    if N > 1:
        tnx[1, :] = 2.0 * x - (N - 1)
    for n in range(1, N - 1):
        tnx[n + 1, :] = (
            (2 * n + 1) * (2.0 * x - (N - 1)) * tnx[n, :]
            - n * (N**2 - n**2) * tnx[n - 1, :]
        ) / (n + 1)

    log_h2 = np.zeros(N, dtype=float)
    for n in range(N):
        if n == 0:
            log_h2[n] = np.log(N) - np.log(1.0)
        else:
            ks = np.arange(1, n + 1, dtype=float)
            log_h2[n] = np.log(N) + np.sum(np.log(N**2 - ks**2)) - np.log(2 * n + 1)
    h = np.exp(0.5 * log_h2)

    return tnx / h[:, None]

def chebyshev_noise_spectroscopy(signal, N=16):
    """
    Performs Chebyshev Noise Spectroscopy.
    
    Args:
        signal: 1D numpy array
        N: Segment length (default 16)
        
    Returns:
        spectra: Array of squared coefficients (Intensity) for each segment.
        avg_spectrum: Median spectrum.
    """
    L = len(signal)
    M = L // N # Number of segments
    
    # 1. Basis (discrete Chebyshev / Gram)
    basis_matrix = get_discrete_chebyshev_basis(N)
    
    # 2. Split and Transform
    segments = signal[:M*N].reshape(M, N)
    
    # Y = C * u_segments^T  (project each segment)
    # segments is (M, N). We want (N, N) * (N, M) -> (N, M)
    # Transpose segments to (N, M)
    coeffs = basis_matrix @ segments.T # (N, M)
    
    # 3. Square
    intensities = coeffs**2
    
    # 4. Median Average along M (segments)
    median_spectrum = np.median(intensities, axis=1)
    
    return intensities, median_spectrum

def test_reproduction():
    # Parameters mimicking the paper
    fs = 10.0 # Hz (assumed)
    duration = 1000.0 # seconds
    N = 16
    
    # Generate Synthetic Data
    # "Without coating": Higher noise, maybe 1/f^1.5
    # "With coating": Lower noise, maybe 1/f^1.0 but lower amplitude
    
    # Case 1: Without Coating (High Corrosion)
    # Add some trend
    t = np.linspace(0, duration, int(duration*fs))
    trend1 = 1e-6 * (t/duration)**2 
    noise1 = 1e-5 * generate_1f_noise(len(t), alpha=1.5)
    sig1 = trend1 + noise1
    
    # Case 2: With Coating (Passivated)
    trend2 = 1e-7 * t/duration
    noise2 = 1e-6 * generate_1f_noise(len(t), alpha=1.2)
    sig2 = trend2 + noise2
    
    # Perform Analysis
    int1, med1 = chebyshev_noise_spectroscopy(sig1, N)
    int2, med2 = chebyshev_noise_spectroscopy(sig2, N)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Plot individual segments (thin lines) - maybe just first 10 for clarity
    # Note: The plot x-axis is index k = 2..15
    k_indices = np.arange(2, N)
    
    # Plot Case 1 (Black)
    # We plot median (thick)
    plt.plot(k_indices, med1[2:], 'k-', linewidth=3, label='Без покрытия (Median)')
    # Plot variation (thin lines, maybe just random 5 segments)
    for i in range(5):
        plt.plot(k_indices, int1[2:, i*10], 'k-', linewidth=0.5, alpha=0.3)
        
    # Plot Case 2 (Gray)
    plt.plot(k_indices, med2[2:], 'gray', linewidth=3, label='С покрытием (Median)')
    for i in range(5):
        plt.plot(k_indices, int2[2:, i*10], 'gray', linewidth=0.5, alpha=0.3)
        
    plt.yscale('log')
    plt.xscale('log')
    
    plt.xlabel('Спектральная линия (k)')
    plt.ylabel('Интенсивность ($A^2$)')
    plt.title('Шумовая Спектроскопия Чебышёва (Test Reconstruct)')
    
    # Set ticks to match the paper (2, 3, 4, 5, 6, 10, 15)
    plt.xticks([2, 3, 4, 5, 6, 10, 15], [2, 3, 4, 5, 6, 10, 15])
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    plt.savefig('test_cheb.png')
    print("Test plot saved to test_cheb.png")

if __name__ == "__main__":
    test_reproduction()
