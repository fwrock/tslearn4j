package org.tslearn.clustering;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.metrics.CrossCorrelation;
import org.tslearn.preprocessing.TimeSeriesScalerMeanVariance;
import org.tslearn.utils.EmptyClusterException;
import org.tslearn.utils.MatrixUtils;

/**
 * KShape clustering for time series.
 * 
 * KShape was originally presented in:
 * J. Paparrizos & L. Gravano. k-Shape: Efficient and Accurate
 * Clustering of Time Series. SIGMOD 2015. pp. 1855-1870.
 * 
 * This implementation is equivalent to the Python tslearn KShape algorithm.
 */
public class KShape {
    
    private static final Logger logger = LoggerFactory.getLogger(KShape.class);
    
    // Parameters
    private final int nClusters;
    private final int maxIter;
    private final double tol;
    private final int nInit;
    private final boolean verbose;
    private final long randomState;
    private final Object init; // Can be "random" or RealMatrix[]
    
    // Fitted attributes
    private RealMatrix[] clusterCenters;
    private int[] labels;
    private double inertia;
    private int nIter;
    private boolean fitted = false;
    
    // Working variables
    private double[] norms;
    private double[] normsCentroids;
    private RealMatrix[] XFit;
    
    /**
     * Constructor with default parameters
     */
    public KShape() {
        this(3, 100, 1e-6, 1, false, 0L, "random");
    }
    
    /**
     * Full constructor
     * 
     * @param nClusters Number of clusters to form
     * @param maxIter Maximum number of iterations 
     * @param tol Inertia variation threshold for convergence
     * @param nInit Number of initialization attempts
     * @param verbose Whether to print progress information
     * @param randomState Random seed
     * @param init Initialization method ("random" or initial centroids)
     */
    public KShape(int nClusters, int maxIter, double tol, int nInit, 
                  boolean verbose, long randomState, Object init) {
        this.nClusters = nClusters;
        this.maxIter = maxIter;
        this.tol = tol;
        this.nInit = nInit;
        this.verbose = verbose;
        this.randomState = randomState;
        this.init = init;
    }
    
    /**
     * Check if the model has been fitted
     */
    public boolean isFitted() {
        return fitted;
    }
    
    /**
     * Shape extraction step - finds optimal centroid for cluster k
     */
    private RealMatrix shapeExtraction(RealMatrix[] X, int k) {
        int sz = X[0].getRowDimension();
        
        // Extract data for cluster k
        RealMatrix[] clusterData = MatrixUtils.extractClusterData(X, labels, k);
        double[] clusterNorms = MatrixUtils.extractClusterNorms(norms, labels, k);
        
        if (clusterData.length == 0) {
            throw new EmptyClusterException("Cluster " + k + " is empty");
        }
        
        // Compute y-shifted SBD vectors
        RealMatrix[] Xp = CrossCorrelation.yShiftedSbdVec(
            clusterCenters[k], 
            clusterData, 
            -1, // norm_ref = -1 (auto-compute)
            clusterNorms
        );
        
        // Convert to 2D matrix for S = X^T * X computation
        RealMatrix XpMatrix = new Array2DRowRealMatrix(clusterData.length, sz);
        for (int i = 0; i < clusterData.length; i++) {
            double[] row = MatrixUtils.matrixToArray(Xp[i]);
            for (int j = 0; j < sz; j++) {
                XpMatrix.setEntry(i, j, row[j]);
            }
        }
        
        // Compute S = X^T * X
        RealMatrix S = XpMatrix.transpose().multiply(XpMatrix);
        
        // Compute Q = I - 1/sz * 1*1^T
        RealMatrix Q = MatrixUtils.eye(sz).subtract(
            MatrixUtils.ones(sz, sz).scalarMultiply(1.0 / sz));
        
        // Compute M = Q^T * S * Q
        RealMatrix M = Q.transpose().multiply(S).multiply(Q);
        
        // Find principal eigenvector using Apache Commons Math
        EigenDecomposition eigenDecomp;
        try {
            eigenDecomp = new EigenDecomposition(M);
        } catch (Exception e) {
            // If eigendecomposition fails, fall back to simple centroid
            logger.warn("Eigendecomposition failed, using simple centroid: " + e.getMessage());
            RealMatrix simpleCentroid = computeSimpleCentroid(clusterData);
            return simpleCentroid;
        }
        
        // Get eigenvector corresponding to largest eigenvalue
        int maxEigenIdx = 0;
        double maxEigenVal = eigenDecomp.getRealEigenvalue(0);
        for (int i = 1; i < eigenDecomp.getRealEigenvalues().length; i++) {
            if (eigenDecomp.getRealEigenvalue(i) > maxEigenVal) {
                maxEigenVal = eigenDecomp.getRealEigenvalue(i);
                maxEigenIdx = i;
            }
        }
        
        RealVector eigenVector = eigenDecomp.getEigenvector(maxEigenIdx);
        RealMatrix muK = new Array2DRowRealMatrix(sz, 1);
        for (int i = 0; i < sz; i++) {
            muK.setEntry(i, 0, eigenVector.getEntry(i));
        }
        
        // Choose sign based on distance minimization
        double distPlusMu = 0.0;
        double distMinusMu = 0.0;
        
        for (int i = 0; i < clusterData.length; i++) {
            RealMatrix xi = Xp[i];
            distPlusMu += computeEuclideanDistance(xi, muK);
            distMinusMu += computeEuclideanDistance(xi, muK.scalarMultiply(-1));
        }
        
        if (distMinusMu < distPlusMu) {
            muK = muK.scalarMultiply(-1);
        }
        
        return muK;
    }
    
    /**
     * Compute simple centroid (mean) as fallback when eigendecomposition fails
     */
    private RealMatrix computeSimpleCentroid(RealMatrix[] clusterData) {
        if (clusterData.length == 0) {
            throw new EmptyClusterException("Cannot compute centroid for empty cluster");
        }
        
        int sz = clusterData[0].getRowDimension();
        int d = clusterData[0].getColumnDimension();
        
        RealMatrix centroid = new Array2DRowRealMatrix(sz, d);
        
        // Compute mean across all series in cluster
        for (int i = 0; i < sz; i++) {
            for (int j = 0; j < d; j++) {
                double sum = 0.0;
                for (RealMatrix ts : clusterData) {
                    sum += ts.getEntry(i, j);
                }
                centroid.setEntry(i, j, sum / clusterData.length);
            }
        }
        
        return centroid;
    }
    
    /**
     * Compute Euclidean distance between two matrices
     */
    private double computeEuclideanDistance(RealMatrix a, RealMatrix b) {
        RealMatrix diff = a.subtract(b);
        double sum = 0.0;
        for (int i = 0; i < diff.getRowDimension(); i++) {
            for (int j = 0; j < diff.getColumnDimension(); j++) {
                double val = diff.getEntry(i, j);
                sum += val * val;
            }
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Update all centroids
     */
    private void updateCentroids(RealMatrix[] X) {
        for (int k = 0; k < nClusters; k++) {
            RealMatrix newCentroid = shapeExtraction(X, k);
            clusterCenters[k] = newCentroid;
        }
        
        // Normalize centroids
        clusterCenters = TimeSeriesScalerMeanVariance.globalTransform(
            clusterCenters, 0.0, 1.0);
        
        // Update centroid norms
        normsCentroids = MatrixUtils.computeNorms(clusterCenters);
    }
    
    /**
     * Compute cross-distances between data and centroids
     */
    private double[][] crossDists(RealMatrix[] X) {
        double[][] ccDists = CrossCorrelation.cdistNormalizedCC(
            X, clusterCenters, norms, normsCentroids, false);
        
        // Convert to distance (1 - correlation)
        for (int i = 0; i < ccDists.length; i++) {
            for (int j = 0; j < ccDists[i].length; j++) {
                ccDists[i][j] = 1.0 - ccDists[i][j];
            }
        }
        
        return ccDists;
    }
    
    /**
     * Assign samples to closest centroids
     */
    private void assign(RealMatrix[] X) {
        double[][] dists = crossDists(X);
        
        // Find closest centroid for each sample
        labels = new int[X.length];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = MatrixUtils.argMin(dists[i]);
        }
        
        // Check for empty clusters
        MatrixUtils.checkNoEmptyCluster(labels, nClusters);
        
        // Compute inertia
        inertia = MatrixUtils.computeInertia(dists, labels);
    }
    
    /**
     * Fit one initialization
     */
    private void fitOneInit(RealMatrix[] X, long seed) {
        RandomGenerator rng = new Well19937c(seed);
        
        // Initialize centroids
        if (init instanceof RealMatrix[]) {
            clusterCenters = new RealMatrix[nClusters];
            RealMatrix[] initCenters = (RealMatrix[]) init;
            for (int i = 0; i < nClusters; i++) {
                clusterCenters[i] = MatrixUtils.copy(initCenters[i]);
            }
        } else if ("random".equals(init)) {
            int[] indices = MatrixUtils.randomChoice(X.length, nClusters, seed);
            clusterCenters = new RealMatrix[nClusters];
            for (int i = 0; i < nClusters; i++) {
                clusterCenters[i] = MatrixUtils.copy(X[indices[i]]);
            }
        } else {
            throw new IllegalArgumentException("Invalid init parameter: " + init);
        }
        
        normsCentroids = MatrixUtils.computeNorms(clusterCenters);
        assign(X);
        
        double oldInertia = Double.POSITIVE_INFINITY;
        
        for (int iter = 0; iter < maxIter; iter++) {
            RealMatrix[] oldClusterCenters = new RealMatrix[nClusters];
            for (int i = 0; i < nClusters; i++) {
                oldClusterCenters[i] = MatrixUtils.copy(clusterCenters[i]);
            }
            
            updateCentroids(X);
            assign(X);
            
            if (verbose) {
                System.out.printf("%.3f --> ", inertia);
            }
            
            if (Math.abs(oldInertia - inertia) < tol || (oldInertia - inertia) < 0) {
                clusterCenters = oldClusterCenters;
                assign(X);
                break;
            }
            
            oldInertia = inertia;
            nIter = iter + 1;
        }
        
        if (verbose) {
            System.out.println();
        }
    }
    
    /**
     * Fit the KShape model
     * 
     * @param X Time series dataset 
     * @return This fitted estimator
     */
    public KShape fit(RealMatrix[] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input cannot be null or empty");
        }
        
        // Verify all time series have same dimensions
        int sz = X[0].getRowDimension();
        int d = X[0].getColumnDimension();
        for (RealMatrix ts : X) {
            if (ts.getRowDimension() != sz || ts.getColumnDimension() != d) {
                throw new IllegalArgumentException("All time series must have same dimensions");
            }
        }
        
        XFit = new RealMatrix[X.length];
        for (int i = 0; i < X.length; i++) {
            XFit[i] = MatrixUtils.copy(X[i]);
        }
        
        norms = MatrixUtils.computeNorms(X);
        
        int maxAttempts = init instanceof RealMatrix[] ? 1 : Math.max(nInit, 10);
        
        labels = null;
        inertia = Double.POSITIVE_INFINITY;
        clusterCenters = null;
        nIter = 0;
        
        RealMatrix[] bestCentroids = null;
        double minInertia = Double.POSITIVE_INFINITY;
        int nSuccessful = 0;
        int nAttempts = 0;
        
        while (nSuccessful < nInit && nAttempts < maxAttempts) {
            try {
                if (verbose && nInit > 1) {
                    System.out.println("Init " + (nSuccessful + 1));
                }
                nAttempts++;
                fitOneInit(X, randomState + nAttempts);
                
                if (inertia < minInertia) {
                    bestCentroids = new RealMatrix[nClusters];
                    for (int i = 0; i < nClusters; i++) {
                        bestCentroids[i] = MatrixUtils.copy(clusterCenters[i]);
                    }
                    minInertia = inertia;
                }
                nSuccessful++;
                
            } catch (EmptyClusterException e) {
                if (verbose) {
                    System.out.println("Resumed because of empty cluster");
                }
            }
        }
        
        if (bestCentroids != null) {
            clusterCenters = bestCentroids;
            inertia = minInertia;
            normsCentroids = MatrixUtils.computeNorms(clusterCenters);
            assign(X);
        }
        
        fitted = true;
        return this;
    }
    
    /**
     * Fit the model and predict cluster labels
     */
    public int[] fitPredict(RealMatrix[] X) {
        return fit(X).getLabels();
    }
    
    /**
     * Predict cluster labels for new data
     */
    public int[] predict(RealMatrix[] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input cannot be null or empty");
        }
        
        // Verify dimensions compatibility
        int sz = clusterCenters[0].getRowDimension();
        int d = clusterCenters[0].getColumnDimension();
        for (RealMatrix ts : X) {
            if (ts.getRowDimension() != sz || ts.getColumnDimension() != d) {
                throw new IllegalArgumentException("Input dimensions must match fitted data");
            }
        }
        
        // Normalize input data
        RealMatrix[] XNormalized = TimeSeriesScalerMeanVariance.globalTransform(X, 0.0, 1.0);
        double[] inputNorms = MatrixUtils.computeNorms(XNormalized);
        
        // Compute distances
        double[][] dists = CrossCorrelation.cdistNormalizedCC(
            XNormalized, clusterCenters, inputNorms, normsCentroids, false);
        
        // Convert to distances (1 - correlation)
        for (int i = 0; i < dists.length; i++) {
            for (int j = 0; j < dists[i].length; j++) {
                dists[i][j] = 1.0 - dists[i][j];
            }
        }
        
        // Assign to closest centroids
        int[] predictedLabels = new int[X.length];
        for (int i = 0; i < predictedLabels.length; i++) {
            predictedLabels[i] = MatrixUtils.argMin(dists[i]);
        }
        
        return predictedLabels;
    }
    
    /**
     * Convenience method to fit with double array input
     */
    public KShape fit(double[][][] X) {
        return fit(MatrixUtils.toTimeSeriesDataset(X));
    }
    
    /**
     * Convenience method to fit with 2D double array input (univariate)
     */
    public KShape fit(double[][] X) {
        return fit(MatrixUtils.toTimeSeriesDataset(X));
    }
    
    /**
     * Convenience method to predict with double array input
     */
    public int[] predict(double[][][] X) {
        return predict(MatrixUtils.toTimeSeriesDataset(X));
    }
    
    /**
     * Convenience method to predict with 2D double array input (univariate)
     */
    public int[] predict(double[][] X) {
        return predict(MatrixUtils.toTimeSeriesDataset(X));
    }
    
    // Getters
    public RealMatrix[] getClusterCenters() { return clusterCenters; }
    public int[] getLabels() { return labels; }
    public double getInertia() { return inertia; }
    public int getNIter() { return nIter; }
    public int getNClusters() { return nClusters; }
    public int getMaxIter() { return maxIter; }
    public double getTol() { return tol; }
    public int getNInit() { return nInit; }
    public boolean isVerbose() { return verbose; }
    public long getRandomState() { return randomState; }
}
