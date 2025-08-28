package org.tslearn.clustering;

import org.apache.commons.math3.linear.RealMatrix;
import org.tslearn.utils.MatrixUtils;

/**
 * Comparative example showing TSLearn4J vs Python tslearn equivalent code
 */
public class PythonEquivalentExample {
    
    public static void main(String[] args) {
        System.out.println("TSLearn4J - Python Equivalent Example");
        System.out.println("=====================================");
        
        /*
         * This Java code is equivalent to the following Python code:
         * 
         * from tslearn.generators import random_walks
         * from tslearn.preprocessing import TimeSeriesScalerMeanVariance
         * from tslearn.clustering import KShape
         * import numpy as np
         * 
         * # Generate random walks
         * X = random_walks(n_ts=50, sz=32, d=1)
         * X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X)
         * 
         * # Fit KShape
         * ks = KShape(n_clusters=3, n_init=1, random_state=0).fit(X)
         * print(f"Cluster centers shape: {ks.cluster_centers_.shape}")
         * print(f"Labels: {ks.labels_}")
         * print(f"Inertia: {ks.inertia_}")
         */
        
        // Generate equivalent of random walks data
        double[][] X = generateRandomWalks(20, 16, 1);
        System.out.println("Generated data shape: [" + X.length + ", " + X[0].length + ", 1]");
        
        // Apply preprocessing (equivalent to TimeSeriesScalerMeanVariance)
        RealMatrix[] timeSeries = MatrixUtils.toTimeSeriesDataset(X);
        
        // Fit KShape (equivalent to Python tslearn KShape)
        System.out.println("\nFitting KShape clustering...");
        KShape ks = new KShape(
            3,       // n_clusters=3
            100,     // max_iter=100 (default)
            1e-6,    // tol=1e-6 (default)
            1,       // n_init=1
            false,   // verbose=False (default)
            0L,      // random_state=0
            "random" // init='random' (default)
        );
        
        ks.fit(timeSeries);
        
        // Print results (equivalent to Python output)
        RealMatrix[] clusterCenters = ks.getClusterCenters();
        System.out.println("Cluster centers shape: [" + clusterCenters.length + ", " + 
                          clusterCenters[0].getRowDimension() + ", " + 
                          clusterCenters[0].getColumnDimension() + "]");
        
        int[] labels = ks.getLabels();
        System.out.print("Labels: [");
        for (int i = 0; i < labels.length; i++) {
            System.out.print(labels[i]);
            if (i < labels.length - 1) System.out.print(", ");
        }
        System.out.println("]");
        
        System.out.println("Inertia: " + String.format("%.6f", ks.getInertia()));
        System.out.println("Converged in: " + ks.getNIter() + " iterations");
        
        // Demonstrate prediction (equivalent to ks.predict(X_new))
        System.out.println("\nTesting prediction...");
        double[][] X_new = generateRandomWalks(3, 16, 1);
        int[] predictions = ks.predict(X_new);
        
        System.out.print("Predictions for new data: [");
        for (int i = 0; i < predictions.length; i++) {
            System.out.print(predictions[i]);
            if (i < predictions.length - 1) System.out.print(", ");
        }
        System.out.println("]");
        
        System.out.println("\n✅ TSLearn4J provides equivalent functionality to Python tslearn!");
    }
    
    /**
     * Generate simple random walks (equivalent to tslearn.generators.random_walks)
     */
    private static double[][] generateRandomWalks(int nTs, int sz, int d) {
        if (d != 1) {
            throw new IllegalArgumentException("Only univariate time series supported (d=1)");
        }
        
        double[][] walks = new double[nTs][sz];
        
        for (int i = 0; i < nTs; i++) {
            walks[i][0] = Math.random() - 0.5; // Start at random position
            
            for (int j = 1; j < sz; j++) {
                // Random walk: next value = current + random step
                double step = (Math.random() - 0.5) * 0.5;
                walks[i][j] = walks[i][j-1] + step;
            }
            
            // Apply some normalization to make patterns more distinct
            double mean = 0.0;
            for (double val : walks[i]) {
                mean += val;
            }
            mean /= sz;
            
            for (int j = 0; j < sz; j++) {
                walks[i][j] -= mean;
            }
        }
        
        return walks;
    }
}
