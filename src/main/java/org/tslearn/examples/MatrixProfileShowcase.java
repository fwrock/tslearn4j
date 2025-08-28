package org.tslearn.examples;

import org.tslearn.matrix_profile.MatrixProfile;

/**
 * Comprehensive Matrix Profile demonstration showing motifs and discords.
 */
public class MatrixProfileShowcase {
    
    public static void main(String[] args) {
        System.out.println("=== Matrix Profile Showcase ===");
        System.out.println("Demonstrating motifs and discords discovery\n");
        
        // Demo 1: Time series with repeated patterns (motifs)
        motifDiscoveryDemo();
        
        // Demo 2: Time series with anomalies (discords)
        discordDetectionDemo();
        
        // Demo 3: Performance test
        performanceTest();
        
        System.out.println("\n=== Matrix Profile Analysis Complete ===");
    }
    
    private static void motifDiscoveryDemo() {
        System.out.println("1. MOTIF DISCOVERY");
        System.out.println("==================");
        
        // Create time series with repeated patterns
        double[] timeSeries = createTimeSeriesWithMotifs();
        
        System.out.printf("Time series length: %d\n", timeSeries.length);
        System.out.printf("Sample values: [%.2f, %.2f, %.2f, ..., %.2f]\n",
                         timeSeries[0], timeSeries[1], timeSeries[2], timeSeries[timeSeries.length-1]);
        
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(15)
                .verbose(true)
                .build();
        
        long startTime = System.currentTimeMillis();
        MatrixProfile.MatrixProfileResult result = mp.stamp(timeSeries);
        long elapsed = System.currentTimeMillis() - startTime;
        
        System.out.printf("Computation time: %d ms\n", elapsed);
        
        // Show Matrix Profile statistics
        double[] profile = result.getMatrixProfile();
        double minDist = Double.POSITIVE_INFINITY;
        double maxDist = -1;
        double sumDist = 0;
        int validCount = 0;
        
        for (double dist : profile) {
            if (dist != Double.POSITIVE_INFINITY) {
                minDist = Math.min(minDist, dist);
                maxDist = Math.max(maxDist, dist);
                sumDist += dist;
                validCount++;
            }
        }
        
        double avgDist = sumDist / validCount;
        System.out.printf("Distance stats: min=%.4f, max=%.4f, avg=%.4f\n", 
                         minDist, maxDist, avgDist);
        
        // Find motifs
        MatrixProfile.MotifResult motifs = mp.findMotifs(result, 3);
        System.out.printf("\nFound %d potential motifs:\n", motifs.getMotifs().size());
        
        for (int i = 0; i < motifs.getMotifs().size(); i++) {
            MatrixProfile.MotifResult.MotifPair motif = motifs.getMotifs().get(i);
            System.out.printf("  Motif %d: indices (%d, %d), distance %.4f\n", 
                            i + 1, motif.getIndex1(), motif.getIndex2(), motif.getDistance());
        }
        
        System.out.println();
    }
    
    private static void discordDetectionDemo() {
        System.out.println("2. DISCORD DETECTION");
        System.out.println("===================");
        
        // Create time series with anomalies
        double[] timeSeries = createTimeSeriesWithAnomalies();
        
        System.out.printf("Time series length: %d\n", timeSeries.length);
        
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(12)
                .verbose(true)
                .build();
        
        MatrixProfile.MatrixProfileResult result = mp.stamp(timeSeries);
        
        // Find discords
        MatrixProfile.DiscordResult discords = mp.findDiscords(result, 3);
        System.out.printf("\nFound %d potential discords:\n", discords.getDiscords().size());
        
        for (int i = 0; i < discords.getDiscords().size(); i++) {
            MatrixProfile.DiscordResult.Discord discord = discords.getDiscords().get(i);
            System.out.printf("  Discord %d: index %d, distance %.4f\n", 
                            i + 1, discord.getIndex(), discord.getDistance());
        }
        
        System.out.println();
    }
    
    private static void performanceTest() {
        System.out.println("3. PERFORMANCE TEST");
        System.out.println("==================");
        
        int[] seriesLengths = {100, 200, 500};
        int subsequenceLength = 20;
        
        System.out.printf("%-12s %-15s %-15s\n", "Series Len", "Time (ms)", "Rate (ops/s)");
        System.out.println("-------------------------------------------");
        
        for (int seriesLen : seriesLengths) {
            double[] testSeries = generateRandomTimeSeries(seriesLen);
            
            MatrixProfile mp = new MatrixProfile.Builder()
                    .subsequenceLength(subsequenceLength)
                    .verbose(false)
                    .build();
            
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
    
    private static double[] createTimeSeriesWithMotifs() {
        double[] series = new double[120];
        java.util.Random rand = new java.util.Random(42);
        
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
        java.util.Random rand = new java.util.Random(42);
        
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
        java.util.Random rand = new java.util.Random(42);
        
        for (int i = 0; i < length; i++) {
            series[i] = Math.sin(2 * Math.PI * i / 30.0) + 
                       0.5 * Math.sin(2 * Math.PI * i / 7.0) + 
                       rand.nextGaussian() * 0.15;
        }
        
        return series;
    }
}
