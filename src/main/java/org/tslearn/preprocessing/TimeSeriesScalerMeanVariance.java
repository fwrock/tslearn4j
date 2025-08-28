package org.tslearn.preprocessing;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Time Series Scaler for mean and variance normalization
 * Equivalent to sklearn.preprocessing.StandardScaler but for time series
 */
public class TimeSeriesScalerMeanVariance {
    
    private static final Logger logger = LoggerFactory.getLogger(TimeSeriesScalerMeanVariance.class);
    
    private final double mu;
    private final double std;
    private boolean fitted = false;
    private double[] meanPerSeries;
    private double[] stdPerSeries;
    
    public TimeSeriesScalerMeanVariance(double mu, double std) {
        this.mu = mu;
        this.std = std;
    }
    
    public TimeSeriesScalerMeanVariance() {
        this(0.0, 1.0);
    }
    
    /**
     * Fit the scaler to the data
     */
    public TimeSeriesScalerMeanVariance fit(RealMatrix[] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input cannot be null or empty");
        }
        
        int nTs = X.length;
        int sz = X[0].getRowDimension();
        int d = X[0].getColumnDimension();
        
        // Compute mean and std for each time series
        this.meanPerSeries = new double[nTs];
        this.stdPerSeries = new double[nTs];
        
        Mean meanCalc = new Mean();
        StandardDeviation stdCalc = new StandardDeviation();
        
        for (int i = 0; i < nTs; i++) {
            // Flatten time series to 1D for statistics calculation
            double[] tsFlat = new double[sz * d];
            int idx = 0;
            for (int j = 0; j < sz; j++) {
                for (int k = 0; k < d; k++) {
                    tsFlat[idx++] = X[i].getEntry(j, k);
                }
            }
            
            meanPerSeries[i] = meanCalc.evaluate(tsFlat);
            stdPerSeries[i] = stdCalc.evaluate(tsFlat);
            
            // Avoid division by zero
            if (stdPerSeries[i] == 0.0) {
                stdPerSeries[i] = 1e-8;
            }
        }
        
        this.fitted = true;
        return this;
    }
    
    /**
     * Transform the data using fitted parameters
     */
    public RealMatrix[] transform(RealMatrix[] X) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before transform");
        }
        
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input cannot be null or empty");
        }
        
        int nTs = X.length;
        RealMatrix[] result = new RealMatrix[nTs];
        
        for (int i = 0; i < nTs; i++) {
            int sz = X[i].getRowDimension();
            int d = X[i].getColumnDimension();
            
            result[i] = new Array2DRowRealMatrix(sz, d);
            
            // Apply transformation: (x - mean) / std * target_std + target_mean
            for (int j = 0; j < sz; j++) {
                for (int k = 0; k < d; k++) {
                    double val = X[i].getEntry(j, k);
                    double normalized = (val - meanPerSeries[i]) / stdPerSeries[i];
                    double scaled = normalized * std + mu;
                    result[i].setEntry(j, k, scaled);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Fit and transform in one step
     */
    public RealMatrix[] fitTransform(RealMatrix[] X) {
        return fit(X).transform(X);
    }
    
    /**
     * Transform using global parameters only (for centroids)
     */
    public static RealMatrix[] globalTransform(RealMatrix[] X, double mu, double std) {
        if (X == null || X.length == 0) {
            return X;
        }
        
        // Compute global statistics
        int totalElements = 0;
        for (RealMatrix matrix : X) {
            totalElements += matrix.getRowDimension() * matrix.getColumnDimension();
        }
        
        double[] allValues = new double[totalElements];
        int idx = 0;
        for (RealMatrix matrix : X) {
            for (int i = 0; i < matrix.getRowDimension(); i++) {
                for (int j = 0; j < matrix.getColumnDimension(); j++) {
                    allValues[idx++] = matrix.getEntry(i, j);
                }
            }
        }
        
        Mean meanCalc = new Mean();
        StandardDeviation stdCalc = new StandardDeviation();
        
        double globalMean = meanCalc.evaluate(allValues);
        double globalStd = stdCalc.evaluate(allValues);
        
        if (globalStd == 0.0) {
            globalStd = 1e-8;
        }
        
        // Apply transformation
        RealMatrix[] result = new RealMatrix[X.length];
        for (int i = 0; i < X.length; i++) {
            int sz = X[i].getRowDimension();
            int d = X[i].getColumnDimension();
            
            result[i] = new Array2DRowRealMatrix(sz, d);
            
            for (int j = 0; j < sz; j++) {
                for (int k = 0; k < d; k++) {
                    double val = X[i].getEntry(j, k);
                    double normalized = (val - globalMean) / globalStd;
                    double scaled = normalized * std + mu;
                    result[i].setEntry(j, k, scaled);
                }
            }
        }
        
        return result;
    }
    
    public boolean isFitted() {
        return fitted;
    }
    
    public double getMu() {
        return mu;
    }
    
    public double getStd() {
        return std;
    }
}
