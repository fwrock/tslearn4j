package org.tslearn.clustering;

import org.apache.commons.math3.linear.RealMatrix;
import org.tslearn.utils.MatrixUtils;

/**
 * Example demonstrating KShape clustering usage
 */
public class KShapeExample {
    
    public static void main(String[] args) {
        System.out.println("KShape Clustering Example");
        System.out.println("========================");
        
        // Create synthetic time series data
        double[][] data = generateSyntheticData();
        
        System.out.println("Generated " + data.length + " time series of length " + data[0].length);
        
        // Convert to RealMatrix format
        RealMatrix[] timeSeries = MatrixUtils.toTimeSeriesDataset(data);
        
        // Create and fit KShape model
        System.out.println("\nFitting KShape with 3 clusters...");
        KShape kshape = new KShape(
            3,      // n_clusters
            50,     // max_iter  
            1e-4,   // tol
            3,      // n_init
            true,   // verbose
            42L,    // random_state
            "random" // init
        );
        
        kshape.fit(timeSeries);
        
        // Print results
        System.out.println("\nResults:");
        System.out.println("-------");
        System.out.println("Converged in " + kshape.getNIter() + " iterations");
        System.out.println("Final inertia: " + String.format("%.6f", kshape.getInertia()));
        
        int[] labels = kshape.getLabels();
        System.out.println("Cluster assignments: ");
        for (int i = 0; i < labels.length; i++) {
            System.out.println("  Time series " + i + " -> Cluster " + labels[i]);
        }
        
        // Show cluster sizes
        int[] clusterSizes = new int[kshape.getNClusters()];
        for (int label : labels) {
            clusterSizes[label]++;
        }
        
        System.out.println("\nCluster sizes:");
        for (int i = 0; i < clusterSizes.length; i++) {
            System.out.println("  Cluster " + i + ": " + clusterSizes[i] + " time series");
        }
        
        // Test prediction on new data
        System.out.println("\nTesting prediction on new data...");
        double[][] newData = {
            {1.5, 2.5, 3.5, 2.5, 1.5, 0.5, 1.5, 2.5, 3.5, 2.5},
            {-1.0, -0.5, 0.0, -0.5, -1.0, -1.5, -1.0, -0.5, 0.0, -0.5}
        };
        
        int[] predictions = kshape.predict(newData);
        System.out.println("Predictions for new data:");
        for (int i = 0; i < predictions.length; i++) {
            System.out.println("  New series " + i + " -> Cluster " + predictions[i]);
        }
        
        // Print centroids info
        System.out.println("\nCluster centroids shape:");
        RealMatrix[] centroids = kshape.getClusterCenters();
        for (int i = 0; i < centroids.length; i++) {
            System.out.println("  Centroid " + i + ": [" + 
                centroids[i].getRowDimension() + ", " + 
                centroids[i].getColumnDimension() + "]");
        }
    }
    
    /**
     * Generate synthetic time series data with 3 different patterns
     */
    private static double[][] generateSyntheticData() {
        int nSeries = 9;
        int length = 10;
        double[][] data = new double[nSeries][length];
        
        // Pattern 1: Increasing trend (3 series)
        for (int s = 0; s < 3; s++) {
            for (int i = 0; i < length; i++) {
                data[s][i] = i + 0.5 * Math.sin(i) + 0.1 * s * Math.random();
            }
        }
        
        // Pattern 2: Decreasing trend (3 series) 
        for (int s = 3; s < 6; s++) {
            for (int i = 0; i < length; i++) {
                data[s][i] = (length - i) + 0.5 * Math.cos(i) + 0.1 * (s-3) * Math.random();
            }
        }
        
        // Pattern 3: Oscillating pattern (3 series)
        for (int s = 6; s < 9; s++) {
            for (int i = 0; i < length; i++) {
                data[s][i] = 2.0 * Math.sin(2 * Math.PI * i / length) + 0.1 * (s-6) * Math.random();
            }
        }
        
        return data;
    }
}
