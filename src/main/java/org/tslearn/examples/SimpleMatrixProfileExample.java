package org.tslearn.examples;

import org.tslearn.matrix_profile.MatrixProfile;

import java.util.Arrays;
import java.util.Random;

/**
 * Simple Matrix Profile example demonstrating basic functionality.
 * 
 * This example shows:
 * - Basic Matrix Profile computation using STAMP algorithm
 * - Motif discovery (repeated patterns)
 * - Discord detection (anomalies)
 * - Performance characteristics
 * 
 * @author TSLearn4J
 */
public class SimpleMatrixProfileExample {
    
    public static void main(String[] args) {
        System.out.println("=== Matrix Profile Example ===\n");
        
        // Demo 1: Basic Matrix Profile computation
        basicMatrixProfileDemo();
        
        // Demo 2: Motif discovery
        motifDiscoveryDemo();
        
        // Demo 3: Discord detection
        discordDetectionDemo();
        
        // Demo 4: Performance test
        performanceTest();
        
        System.out.println("\n=== Matrix Profile Analysis Complete ===");
    }
    
    /**
     * Demonstrates basic Matrix Profile computation.
     */
    private static void basicMatrixProfileDemo() {
        System.out.println("1. Basic Matrix Profile Computation");
        System.out.println("===================================");
        
        // Create synthetic time series
        double[] timeSeries = createSyntheticTimeSeries();
        
        System.out.printf("Time series length: %d\n", timeSeries.length);
        System.out.printf("Sample values: [%.2f, %.2f, %.2f, ..., %.2f]\n",
                         timeSeries[0], timeSeries[1], timeSeries[2], timeSeries[timeSeries.length-1]);
        
        // Configure Matrix Profile
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(20)
                .verbose(true)
                .build();
        
        // Compute Matrix Profile
        long startTime = System.currentTimeMillis();
        MatrixProfile.MatrixProfileResult result = mp.stamp(timeSeries);
        long elapsed = System.currentTimeMillis() - startTime;
        
        System.out.printf("Computation time: %d ms\n", elapsed);
        System.out.printf("Profile length: %d\n", result.getMatrixProfile().length);
        
        // Show some statistics
        double[] mp_values = result.getMatrixProfile();
        double minDist = Arrays.stream(mp_values)
                              .filter(d -> d != Double.POSITIVE_INFINITY)
                              .min().orElse(0.0);
        double maxDist = Arrays.stream(mp_values)
                              .filter(d -> d != Double.POSITIVE_INFINITY)
                              .max().orElse(0.0);
        double avgDist = Arrays.stream(mp_values)
                              .filter(d -> d != Double.POSITIVE_INFINITY)
                              .average().orElse(0.0);
        
        System.out.printf("Distance statistics: min=%.4f, max=%.4f, avg=%.4f\n", 
                         minDist, maxDist, avgDist);
        System.out.println();
    }
    
    /**
     * Demonstrates motif discovery.
     */
    private static void motifDiscoveryDemo() {
        System.out.println("2. Motif Discovery");
        System.out.println("==================");
        
        // Create time series with known repeated patterns
        double[] timeSeries = createTimeSeriesWithMotifs();
        
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(15)
                .verbose(true)
                .build();
        
        MatrixProfile.MatrixProfileResult result = mp.stamp(timeSeries);
        MatrixProfile.MotifResult motifs = mp.findMotifs(result, 3);
        
        System.out.printf("Found %d motifs:\n", motifs.getMotifs().size());
        for (int i = 0; i < motifs.getMotifs().size(); i++) {
            MatrixProfile.MotifResult.MotifPair motif = motifs.getMotifs().get(i);
            System.out.printf("  Motif %d: indices (%d, %d), distance %.4f\n", 
                            i + 1, motif.getIndex1(), motif.getIndex2(), motif.getDistance());
            
            // Extract and show the actual subsequences
            double[] subseq1 = mp.extractSubsequence(timeSeries, motif.getIndex1());
            double[] subseq2 = mp.extractSubsequence(timeSeries, motif.getIndex2());
            
            System.out.printf("    Pattern 1: [%.2f, %.2f, %.2f, ..., %.2f]\n",
                            subseq1[0], subseq1[1], subseq1[2], subseq1[subseq1.length-1]);
            System.out.printf("    Pattern 2: [%.2f, %.2f, %.2f, ..., %.2f]\n",
                            subseq2[0], subseq2[1], subseq2[2], subseq2[subseq2.length-1]);
        }
        System.out.println();
    }
    
    /**
     * Demonstrates discord (anomaly) detection.
     */
    private static void discordDetectionDemo() {
        System.out.println("3. Discord Detection");
        System.out.println("===================");
        
        // Create time series with anomalies
        double[] timeSeries = createTimeSeriesWithAnomalies();
        
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(12)
                .verbose(true)
                .build();
        
        MatrixProfile.MatrixProfileResult result = mp.stamp(timeSeries);
        MatrixProfile.DiscordResult discords = mp.findDiscords(result, 3);
        
        System.out.printf("Found %d discords:\n", discords.getDiscords().size());
        for (int i = 0; i < discords.getDiscords().size(); i++) {
            MatrixProfile.DiscordResult.Discord discord = discords.getDiscords().get(i);
            System.out.printf("  Discord %d: index %d, distance %.4f\n", 
                            i + 1, discord.getIndex(), discord.getDistance());
            
            // Extract and show the anomalous subsequence
            double[] anomaly = mp.extractSubsequence(timeSeries, discord.getIndex());
            System.out.printf("    Anomaly: [%.2f, %.2f, %.2f, ..., %.2f]\n",
                            anomaly[0], anomaly[1], anomaly[2], anomaly[anomaly.length-1]);
        }
        System.out.println();
    }
    
    /**
     * Demonstrates performance characteristics.
     */
    private static void performanceTest() {
        System.out.println("4. Performance Test");
        System.out.println("===================");
        
        int[] seriesLengths = {100, 200, 500, 1000};
        int subsequenceLength = 20;
        
        System.out.printf("%-12s %-15s %-15s\n", "Series Len", "Time (ms)", "Rate (op/s)");
        System.out.println("-------------------------------------------");
        
        for (int seriesLen : seriesLengths) {
            // Generate test data
            double[] testSeries = generateRandomTimeSeries(seriesLen);
            
            // Configure Matrix Profile
            MatrixProfile mp = new MatrixProfile.Builder()
                    .subsequenceLength(subsequenceLength)
                    .verbose(false)
                    .build();
            
            // Benchmark computation
            long startTime = System.nanoTime();
            MatrixProfile.MatrixProfileResult result = mp.stamp(testSeries);
            long elapsed = System.nanoTime() - startTime;
            
            double timeMs = elapsed / 1_000_000.0;
            double rate = 1000.0 / timeMs;
            
            System.out.printf("%-12d %-15.1f %-15.1f\n", seriesLen, timeMs, rate);
        }
        System.out.println();
    }
    
    // Helper methods for generating test data
    
    private static double[] createSyntheticTimeSeries() {
        double[] series = new double[100];
        Random rand = new Random(42);
        
        for (int i = 0; i < series.length; i++) {
            // Combine sine wave with noise
            series[i] = Math.sin(2 * Math.PI * i / 25.0) + rand.nextGaussian() * 0.1;
        }
        
        return series;
    }
    
    private static double[] createTimeSeriesWithMotifs() {
        double[] series = new double[120];
        Random rand = new Random(42);
        
        // Base noise
        for (int i = 0; i < series.length; i++) {
            series[i] = rand.nextGaussian() * 0.1;
        }
        
        // Create a motif pattern (triangle wave)
        double[] motifPattern = new double[15];
        for (int i = 0; i < motifPattern.length; i++) {
            if (i < motifPattern.length / 2) {
                motifPattern[i] = 2.0 * i / (motifPattern.length / 2);
            } else {
                motifPattern[i] = 2.0 - 2.0 * (i - motifPattern.length / 2) / (motifPattern.length / 2);
            }
        }
        
        // Insert motif at multiple locations
        int[] motifLocations = {20, 60, 90};
        for (int loc : motifLocations) {
            for (int i = 0; i < motifPattern.length && loc + i < series.length; i++) {
                series[loc + i] += motifPattern[i];
            }
        }
        
        return series;
    }
    
    private static double[] createTimeSeriesWithAnomalies() {
        double[] series = new double[100];
        Random rand = new Random(42);
        
        // Normal pattern (sine wave with noise)
        for (int i = 0; i < series.length; i++) {
            series[i] = Math.sin(2 * Math.PI * i / 20.0) + rand.nextGaussian() * 0.1;
        }
        
        // Inject anomalies (sudden spikes)
        int[] anomalyLocations = {25, 55, 80};
        for (int loc : anomalyLocations) {
            for (int i = 0; i < 8 && loc + i < series.length; i++) {
                series[loc + i] += 2.0 + rand.nextGaussian() * 0.2; // Add large spike
            }
        }
        
        return series;
    }
    
    private static double[] generateRandomTimeSeries(int length) {
        double[] series = new double[length];
        Random rand = new Random(42);
        
        for (int i = 0; i < length; i++) {
            series[i] = Math.sin(2 * Math.PI * i / 30.0) + 
                       0.5 * Math.sin(2 * Math.PI * i / 7.0) + 
                       rand.nextGaussian() * 0.15;
        }
        
        return series;
    }
}
