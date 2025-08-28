package org.tslearn.metrics;

import org.jtransforms.fft.DoubleFFT_1D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Fast cross-correlation using FFT (Fast Fourier Transform)
 * Equivalent to Python tslearn implementation using numpy.fft
 */
public class FastCrossCorrelation {
    
    private static final Logger logger = LoggerFactory.getLogger(FastCrossCorrelation.class);
    private static final double EPSILON = 1e-9;
    
    /**
     * Compute normalized cross-correlation using FFT
     * This is much faster than naive O(n²) approach for longer time series
     * 
     * @param s1 First time series
     * @param s2 Second time series  
     * @param norm1 Norm of first series (-1 to auto-compute)
     * @param norm2 Norm of second series (-1 to auto-compute)
     * @return Cross-correlation array with all possible shifts
     */
    public static double[] normalizedCrossCorrelationFFT(double[] s1, double[] s2, 
                                                        double norm1, double norm2) {
        if (s1.length != s2.length) {
            throw new IllegalArgumentException("Time series must have same length");
        }
        
        int sz = s1.length;
        
        // Compute norms if needed
        if (norm1 < 0.0) {
            norm1 = computeNorm(s1);
        }
        if (norm2 < 0.0) {
            norm2 = computeNorm(s2);
        }
        
        double denom = norm1 * norm2;
        if (denom < EPSILON) {
            // Return zeros if either series has zero norm
            return new double[2 * sz - 1];
        }
        
        // Determine FFT size (next power of 2 >= 2*sz-1)
        int nBits = Integer.SIZE - Integer.numberOfLeadingZeros(2 * sz - 2);
        int fftSize = 1 << nBits;
        
        // Prepare FFT inputs (real-imaginary interleaved format)
        double[] fft1 = new double[2 * fftSize];
        double[] fft2 = new double[2 * fftSize];
        
        // Copy data to FFT arrays (only real parts, imaginary = 0)
        for (int i = 0; i < sz; i++) {
            fft1[2 * i] = s1[i];     // Real part
            fft1[2 * i + 1] = 0.0;   // Imaginary part
            
            fft2[2 * i] = s2[i];     // Real part  
            fft2[2 * i + 1] = 0.0;   // Imaginary part
        }
        
        // Zero-pad the rest
        for (int i = sz; i < fftSize; i++) {
            fft1[2 * i] = 0.0;
            fft1[2 * i + 1] = 0.0;
            fft2[2 * i] = 0.0;
            fft2[2 * i + 1] = 0.0;
        }
        
        // Compute FFTs
        DoubleFFT_1D fftTransform = new DoubleFFT_1D(fftSize);
        fftTransform.complexForward(fft1);
        fftTransform.complexForward(fft2);
        
        // Compute element-wise multiplication: FFT1 * conj(FFT2)
        double[] result = new double[2 * fftSize];
        for (int i = 0; i < fftSize; i++) {
            double real1 = fft1[2 * i];
            double imag1 = fft1[2 * i + 1];
            double real2 = fft2[2 * i];
            double imag2 = fft2[2 * i + 1];
            
            // Complex multiplication: (a + bi) * (c - di) = (ac + bd) + (bc - ad)i
            result[2 * i] = real1 * real2 + imag1 * imag2;       // Real part
            result[2 * i + 1] = imag1 * real2 - real1 * imag2;   // Imaginary part
        }
        
        // Compute inverse FFT
        fftTransform.complexInverse(result, true); // true = scale by 1/n
        
        // Extract and reorder cross-correlation result
        double[] cc = new double[2 * sz - 1];
        
        // Reorder: [cc[-(sz-1):], cc[:sz]] as in Python implementation
        for (int i = 0; i < sz - 1; i++) {
            cc[i] = result[2 * (fftSize - (sz - 1) + i)] / denom; // Take only real part
        }
        for (int i = 0; i < sz; i++) {
            cc[sz - 1 + i] = result[2 * i] / denom; // Take only real part
        }
        
        return cc;
    }
    
    /**
     * Find maximum normalized cross-correlation (equivalent to old implementation)
     */
    public static double normalizedCrossCorrelationMax(double[] s1, double[] s2, 
                                                      double norm1, double norm2) {
        double[] cc = normalizedCrossCorrelationFFT(s1, s2, norm1, norm2);
        
        double maxCorr = Double.NEGATIVE_INFINITY;
        for (double val : cc) {
            if (val > maxCorr) {
                maxCorr = val;
            }
        }
        
        return maxCorr;
    }
    
    /**
     * Find the best shift that maximizes cross-correlation
     */
    public static int findBestShiftFFT(double[] centroid, double[] ts, 
                                      double normCentroid, double normTs) {
        if (normCentroid <= 0 || normTs <= 0) {
            return 0;
        }
        
        double[] cc = normalizedCrossCorrelationFFT(centroid, ts, normCentroid, normTs);
        
        double maxCorr = Double.NEGATIVE_INFINITY;
        int bestShift = 0;
        
        for (int i = 0; i < cc.length; i++) {
            if (cc[i] > maxCorr) {
                maxCorr = cc[i];
                bestShift = i - (centroid.length - 1); // Convert index to shift
            }
        }
        
        return bestShift;
    }
    
    /**
     * Compute L2 norm of array
     */
    private static double computeNorm(double[] array) {
        double sum = 0.0;
        for (double val : array) {
            sum += val * val;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Check if FFT optimization should be used based on time series length
     * For short series, naive approach might be faster due to FFT overhead
     */
    public static boolean shouldUseFFT(int timeSeriesLength) {
        // Use FFT for series longer than 64 points (empirical threshold from tests)
        // Below this, the overhead of FFT setup outweighs the benefits
        return timeSeriesLength > 64;
    }
}
