package org.tslearn.utils;

import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility class for time series operations using Apache Commons Math
 * @deprecated Use MatrixUtils instead
 */
@Deprecated
public class TimeSeriesUtils {
    
    private static final Logger logger = LoggerFactory.getLogger(TimeSeriesUtils.class);
    
    /**
     * Convert time series dataset to proper RealMatrix format
     * 
     * @param data Input data as double array [n_ts, sz, d]
     * @return RealMatrix array representation
     */
    public static RealMatrix[] toTimeSeriesDataset(double[][][] data) {
        return MatrixUtils.toTimeSeriesDataset(data);
    }
    
    /**
     * Convert 2D double array to RealMatrix array
     */
    public static RealMatrix[] toTimeSeriesDataset(double[][] data) {
        return MatrixUtils.toTimeSeriesDataset(data);
    }
    
    /**
     * Check if dimensions are compatible
     */
    public static void checkDims(RealMatrix[] X, int[] expectedDims) {
        if (X.length == 0) return;
        
        RealMatrix first = X[0];
        if (first.getRowDimension() != expectedDims[1] || 
            first.getColumnDimension() != expectedDims[2]) {
            throw new IllegalArgumentException(
                String.format("Dimension mismatch: expected [%d, %d], got [%d, %d]", 
                    expectedDims[1], expectedDims[2],
                    first.getRowDimension(), first.getColumnDimension()));
        }
    }
    
    /**
     * Compute L2 norm of time series
     */
    public static double norm(RealMatrix ts) {
        return MatrixUtils.norm(ts);
    }
    
    /**
     * Generate random indices for initialization
     */
    public static int[] randomChoice(int populationSize, int sampleSize, long seed) {
        return MatrixUtils.randomChoice(populationSize, sampleSize, seed);
    }
    
    /**
     * Check if any cluster is empty
     */
    public static void checkNoEmptyCluster(int[] labels, int nClusters) {
        MatrixUtils.checkNoEmptyCluster(labels, nClusters);
    }
    
    /**
     * Compute inertia from distances and labels
     */
    public static double computeInertia(double[][] distances, int[] labels) {
        return MatrixUtils.computeInertia(distances, labels);
    }
}
