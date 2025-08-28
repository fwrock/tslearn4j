package org.tslearn.examples;

import org.tslearn.matrix_profile.MatrixProfile;

/**
 * Basic Matrix Profile demonstration.
 */
public class BasicMatrixProfileDemo {
    
    public static void main(String[] args) {
        System.out.println("=== Basic Matrix Profile Demo ===");
        
        // Create simple test data
        double[] timeSeries = new double[50];
        for (int i = 0; i < timeSeries.length; i++) {
            timeSeries[i] = Math.sin(2 * Math.PI * i / 10.0);
        }
        
        System.out.printf("Time series length: %d\n", timeSeries.length);
        
        // Create Matrix Profile
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(10)
                .verbose(true)
                .build();
        
        // Compute Matrix Profile
        System.out.println("Computing Matrix Profile...");
        MatrixProfile.MatrixProfileResult result = mp.stamp(timeSeries);
        
        System.out.printf("Matrix Profile computed successfully!\n");
        System.out.printf("Profile length: %d\n", result.getMatrixProfile().length);
        
        // Find top motif
        MatrixProfile.MotifResult motifs = mp.findMotifs(result, 1);
        if (!motifs.getMotifs().isEmpty()) {
            MatrixProfile.MotifResult.MotifPair motif = motifs.getMotifs().get(0);
            System.out.printf("Top motif: indices (%d, %d), distance %.4f\n", 
                            motif.getIndex1(), motif.getIndex2(), motif.getDistance());
        }
        
        // Find top discord
        MatrixProfile.DiscordResult discords = mp.findDiscords(result, 1);
        if (!discords.getDiscords().isEmpty()) {
            MatrixProfile.DiscordResult.Discord discord = discords.getDiscords().get(0);
            System.out.printf("Top discord: index %d, distance %.4f\n", 
                            discord.getIndex(), discord.getDistance());
        }
        
        System.out.println("Demo completed successfully!");
    }
}
