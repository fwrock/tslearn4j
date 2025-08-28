package org.tslearn.metrics;

import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.utils.MatrixUtils;

/**
 * Cross-correlation and distance metrics for time series
 */
public class CrossCorrelation {
    
    private static final Logger logger = LoggerFactory.getLogger(CrossCorrelation.class);
    
    /**
     * Compute normalized cross-correlation distance matrix
     * 
     * @param X1 First dataset 
     * @param X2 Second dataset  
     * @param norms1 Norms of X1 series
     * @param norms2 Norms of X2 series
     * @param selfSimilarity Whether computing self-similarity
     * @return Distance matrix [n_ts1, n_ts2]
     */
    public static double[][] cdistNormalizedCC(RealMatrix[] X1, RealMatrix[] X2, 
                                             double[] norms1, double[] norms2, 
                                             boolean selfSimilarity) {
        
        int n1 = X1.length;
        int n2 = X2.length;
        
        // Verify univariate time series
        if (X1[0].getColumnDimension() != 1) {
            throw new IllegalArgumentException("Only univariate time series supported (d=1)");
        }
        
        double[][] distances = new double[n1][n2];
        
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                if (selfSimilarity && i == j) {
                    distances[i][j] = 1.0;
                    continue;
                }
                
                double[] ts1 = MatrixUtils.matrixToArray(X1[i]);
                double[] ts2 = MatrixUtils.matrixToArray(X2[j]);
                
                double norm1 = norms1[i];
                double norm2 = norms2[j];
                
                double maxCorr = normalizedCrossCorrelation(ts1, ts2, norm1, norm2);
                distances[i][j] = maxCorr;
            }
        }
        
        return distances;
    }
    
    /**
     * Compute normalized cross-correlation between two time series
     * Uses FFT optimization for longer series, naive approach for shorter ones
     */
    public static double normalizedCrossCorrelation(double[] ts1, double[] ts2, 
                                                   double norm1, double norm2) {
        if (norm1 == 0.0 || norm2 == 0.0) {
            return 0.0;
        }
        
        int sz = ts1.length;
        
        // Use FFT for longer series, naive approach for shorter ones
        if (FastCrossCorrelation.shouldUseFFT(sz)) {
            return FastCrossCorrelation.normalizedCrossCorrelationMax(ts1, ts2, norm1, norm2);
        } else {
            // Original naive approach for short series
            double maxCorr = 0.0;
            
            // Compute cross-correlation for all possible shifts
            for (int shift = -(sz-1); shift < sz; shift++) {
                double corr = 0.0;
                
                for (int i = 0; i < sz; i++) {
                    int j = i + shift;
                    if (j >= 0 && j < sz) {
                        corr += ts1[i] * ts2[j];
                    }
                }
                
                double normalizedCorr = corr / (norm1 * norm2);
                maxCorr = Math.max(maxCorr, normalizedCorr);
            }
            
            return maxCorr;
        }
    }
    
    /**
     * Compute y-shifted SBD (Shape-Based Distance) vectors
     * This is used in the shape extraction step of KShape
     */
    public static RealMatrix[] yShiftedSbdVec(RealMatrix centroid, RealMatrix[] dataset, 
                                            double normRef, double[] normsDataset) {
        
        int nTs = dataset.length;
        int sz = dataset[0].getRowDimension();
        int d = dataset[0].getColumnDimension();
        
        if (d != 1) {
            throw new IllegalArgumentException("Only univariate time series supported");
        }
        
        RealMatrix[] result = new RealMatrix[nTs];
        double[] centroidArray = MatrixUtils.matrixToArray(centroid);
        double centroidNorm = MatrixUtils.norm(centroid);
        
        for (int i = 0; i < nTs; i++) {
            double[] ts = MatrixUtils.matrixToArray(dataset[i]);
            double normTs = normsDataset[i];
            
            // Find best shift
            int bestShift = findBestShift(centroidArray, ts, centroidNorm, normTs);
            
            // Apply shift
            double[] shifted = applyShift(ts, bestShift);
            
            // Convert back to matrix
            result[i] = MatrixUtils.arrayToMatrix(shifted);
        }
        
        return result;
    }
    
    /**
     * Find the best shift that maximizes normalized cross-correlation
     * Uses FFT optimization when appropriate
     */
    private static int findBestShift(double[] centroid, double[] ts, 
                                    double normCentroid, double normTs) {
        if (normCentroid <= 0 || normTs <= 0) {
            return 0;
        }
        
        int sz = centroid.length;
        
        // Use FFT for longer series
        if (FastCrossCorrelation.shouldUseFFT(sz)) {
            return FastCrossCorrelation.findBestShiftFFT(centroid, ts, normCentroid, normTs);
        } else {
            // Original naive approach for short series
            double maxCorr = Double.NEGATIVE_INFINITY;
            int bestShift = 0;
            
            for (int shift = -(sz-1); shift < sz; shift++) {
                double corr = 0.0;
                
                for (int i = 0; i < sz; i++) {
                    int j = i + shift;
                    if (j >= 0 && j < sz) {
                        corr += centroid[i] * ts[j];
                    }
                }
                
                double normalizedCorr = corr / (normCentroid * normTs);
                if (normalizedCorr > maxCorr) {
                    maxCorr = normalizedCorr;
                    bestShift = shift;
                }
            }
            
            return bestShift;
        }
    }
    
    /**
     * Apply time shift to a time series
     */
    private static double[] applyShift(double[] ts, int shift) {
        int sz = ts.length;
        double[] shifted = new double[sz];
        
        for (int i = 0; i < sz; i++) {
            int sourceIdx = i - shift;
            if (sourceIdx >= 0 && sourceIdx < sz) {
                shifted[i] = ts[sourceIdx];
            }
            // else: shifted[i] remains 0.0 (default value)
        }
        
        return shifted;
    }
}
