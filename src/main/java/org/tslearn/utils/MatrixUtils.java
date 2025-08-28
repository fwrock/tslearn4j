package org.tslearn.utils;

import java.util.HashSet;
import java.util.Set;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Matrix utilities for time series operations using Apache Commons Math
 */
public class MatrixUtils {
    
    private static final Logger logger = LoggerFactory.getLogger(MatrixUtils.class);
    
    /**
     * Convert 3D time series array to RealMatrix list
     * Each time series becomes a RealMatrix with shape [sz, d]
     */
    public static RealMatrix[] toTimeSeriesDataset(double[][][] data) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("Data cannot be null or empty");
        }
        
        int nTs = data.length;
        int sz = data[0].length;
        int d = data[0][0].length;
        
        RealMatrix[] result = new RealMatrix[nTs];
        
        for (int i = 0; i < nTs; i++) {
            result[i] = new Array2DRowRealMatrix(sz, d);
            for (int j = 0; j < sz; j++) {
                for (int k = 0; k < d; k++) {
                    result[i].setEntry(j, k, data[i][j][k]);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Convert 2D double array to RealMatrix array (assuming d=1)
     */
    public static RealMatrix[] toTimeSeriesDataset(double[][] data) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("Data cannot be null or empty");
        }
        
        int nTs = data.length;
        int sz = data[0].length;
        
        RealMatrix[] result = new RealMatrix[nTs];
        
        for (int i = 0; i < nTs; i++) {
            result[i] = new Array2DRowRealMatrix(sz, 1);
            for (int j = 0; j < sz; j++) {
                result[i].setEntry(j, 0, data[i][j]);
            }
        }
        
        return result;
    }
    
    /**
     * Compute L2 norm of a time series matrix
     */
    public static double norm(RealMatrix ts) {
        double sum = 0.0;
        for (int i = 0; i < ts.getRowDimension(); i++) {
            for (int j = 0; j < ts.getColumnDimension(); j++) {
                double val = ts.getEntry(i, j);
                sum += val * val;
            }
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Compute norms for array of time series
     */
    public static double[] computeNorms(RealMatrix[] timeSeries) {
        double[] norms = new double[timeSeries.length];
        for (int i = 0; i < timeSeries.length; i++) {
            norms[i] = norm(timeSeries[i]);
        }
        return norms;
    }
    
    /**
     * Generate random indices for initialization
     */
    public static int[] randomChoice(int populationSize, int sampleSize, long seed) {
        if (sampleSize > populationSize) {
            throw new IllegalArgumentException("Sample size cannot be larger than population size");
        }
        
        RandomGenerator rng = new Well19937c(seed);
        int[] indices = new int[sampleSize];
        boolean[] used = new boolean[populationSize];
        
        for (int i = 0; i < sampleSize; i++) {
            int index;
            do {
                index = rng.nextInt(populationSize);
            } while (used[index]);
            
            indices[i] = index;
            used[index] = true;
        }
        
        return indices;
    }
    
    /**
     * Check if any cluster is empty
     */
    public static void checkNoEmptyCluster(int[] labels, int nClusters) {
        Set<Integer> presentClusters = new HashSet<>();
        for (int label : labels) {
            if (label >= 0 && label < nClusters) {
                presentClusters.add(label);
            }
        }
        
        for (int i = 0; i < nClusters; i++) {
            if (!presentClusters.contains(i)) {
                throw new EmptyClusterException("Cluster " + i + " is empty");
            }
        }
    }
    
    /**
     * Compute inertia from distances and labels
     */
    public static double computeInertia(double[][] distances, int[] labels) {
        double inertia = 0.0;
        for (int i = 0; i < labels.length; i++) {
            inertia += distances[i][labels[i]];
        }
        return inertia;
    }
    
    /**
     * Create identity matrix
     */
    public static RealMatrix eye(int size) {
        RealMatrix identity = new Array2DRowRealMatrix(size, size);
        for (int i = 0; i < size; i++) {
            identity.setEntry(i, i, 1.0);
        }
        return identity;
    }
    
    /**
     * Create matrix of ones
     */
    public static RealMatrix ones(int rows, int cols) {
        RealMatrix ones = new Array2DRowRealMatrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                ones.setEntry(i, j, 1.0);
            }
        }
        return ones;
    }
    
    /**
     * Element-wise matrix subtraction
     */
    public static RealMatrix subtract(RealMatrix a, RealMatrix b) {
        return a.subtract(b);
    }
    
    /**
     * Scalar matrix multiplication
     */
    public static RealMatrix scalarMultiply(RealMatrix matrix, double scalar) {
        return matrix.scalarMultiply(scalar);
    }
    
    /**
     * Extract time series belonging to a specific cluster
     */
    public static RealMatrix[] extractClusterData(RealMatrix[] data, int[] labels, int cluster) {
        // Count samples in cluster
        int count = 0;
        for (int label : labels) {
            if (label == cluster) count++;
        }
        
        if (count == 0) {
            throw new EmptyClusterException("Cluster " + cluster + " is empty");
        }
        
        // Extract data
        RealMatrix[] clusterData = new RealMatrix[count];
        int idx = 0;
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] == cluster) {
                clusterData[idx++] = data[i];
            }
        }
        
        return clusterData;
    }
    
    /**
     * Extract norms for samples belonging to a specific cluster
     */
    public static double[] extractClusterNorms(double[] norms, int[] labels, int cluster) {
        // Count samples in cluster
        int count = 0;
        for (int label : labels) {
            if (label == cluster) count++;
        }
        
        // Extract norms
        double[] clusterNorms = new double[count];
        int idx = 0;
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] == cluster) {
                clusterNorms[idx++] = norms[i];
            }
        }
        
        return clusterNorms;
    }
    
    /**
     * Find argmin of a row in distance matrix
     */
    public static int argMin(double[] row) {
        int minIdx = 0;
        double minVal = row[0];
        for (int i = 1; i < row.length; i++) {
            if (row[i] < minVal) {
                minVal = row[i];
                minIdx = i;
            }
        }
        return minIdx;
    }
    
    /**
     * Convert RealMatrix to 1D array (for univariate time series)
     */
    public static double[] matrixToArray(RealMatrix matrix) {
        if (matrix.getColumnDimension() != 1) {
            throw new IllegalArgumentException("Matrix must have single column for univariate conversion");
        }
        
        double[] result = new double[matrix.getRowDimension()];
        for (int i = 0; i < result.length; i++) {
            result[i] = matrix.getEntry(i, 0);
        }
        return result;
    }
    
    /**
     * Convert 1D array to RealMatrix column vector
     */
    public static RealMatrix arrayToMatrix(double[] array) {
        RealMatrix result = new Array2DRowRealMatrix(array.length, 1);
        for (int i = 0; i < array.length; i++) {
            result.setEntry(i, 0, array[i]);
        }
        return result;
    }
    
    /**
     * Copy a RealMatrix
     */
    public static RealMatrix copy(RealMatrix matrix) {
        return matrix.copy();
    }
}
