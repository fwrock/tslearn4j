package org.tslearn.examples;

import org.tslearn.matrix_profile.*;

import java.util.*;

/**
 * Comprehensive examples demonstrating Matrix Profile functionality.
 * 
 * This example shows:
 * - Basic Matrix Profile computation
 * - Motif discovery
 * - Discord detection
 * - Multidimensional analysis
 * - Streaming anomaly detection
 * - Performance benchmarking
 * 
 * @author TSLearn4J
 */
public class MatrixProfileExample {
    
    public static void main(String[] args) {
        System.out.println("=== Matrix Profile Example ===\n");
        
        // Demo 1: Basic Matrix Profile and Motif Discovery
        basicMatrixProfileDemo();
        
        // Demo 2: Discord Discovery
        discordDiscoveryDemo();
        
        // Demo 3: Multivariate Analysis
        multivariateAnalysisDemo();
        
        // Demo 4: Performance Comparison
        performanceDemo();
        
        // Demo 5: Advanced Features
        advancedFeaturesDemo();
    }
    
    /**
     * Demonstrates basic Matrix Profile computation and motif discovery.
     */
    private static void basicMatrixProfileDemo() {
        System.out.println("1. Basic Matrix Profile and Motif Discovery");
        System.out.println("==========================================");
        
        // Create synthetic time series with repeated patterns
        double[] timeSeries = createSyntheticTimeSeriesWithMotifs();
        
        System.out.printf("Time series length: %d\n", timeSeries.length);
        System.out.printf("Sample values: [%.2f, %.2f, %.2f, ..., %.2f, %.2f]\n\n",
                         timeSeries[0], timeSeries[1], timeSeries[2], 
                         timeSeries[timeSeries.length-2], timeSeries[timeSeries.length-1]);
        
        // Configure Matrix Profile
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(20)
                .verbose(true)
                .build();
        
        // Compute Matrix Profile
        System.out.println("Computing Matrix Profile...");
        long startTime = System.currentTimeMillis();
        MatrixProfile.MatrixProfileResult result = mp.stamp(timeSeries);
        long elapsed = System.currentTimeMillis() - startTime;
        
        System.out.printf("Matrix Profile computed in %d ms\n", elapsed);
        System.out.printf("Profile length: %d\n", result.getMatrixProfile().length);
        
        // Find motifs
        System.out.println("\nDiscovering motifs...");
        MatrixProfile.MotifResult motifs = mp.findMotifs(result, 3);
        
        System.out.printf("Found %d motifs:\n", motifs.getMotifs().size());
        for (int i = 0; i < motifs.getMotifs().size(); i++) {
            MatrixProfile.MotifResult.MotifPair motif = motifs.getMotifs().get(i);
            System.out.printf("  Motif %d: indices (%d, %d), distance %.4f\n", 
                            i + 1, motif.getIndex1(), motif.getIndex2(), motif.getDistance());
            
            // Print actual subsequences
            double[] subseq1 = mp.extractSubsequence(timeSeries, motif.getIndex1());
            double[] subseq2 = mp.extractSubsequence(timeSeries, motif.getIndex2());
            System.out.printf("    Subsequence 1: [%.2f, %.2f, %.2f, ..., %.2f]\n",
                            subseq1[0], subseq1[1], subseq1[2], subseq1[subseq1.length-1]);
            System.out.printf("    Subsequence 2: [%.2f, %.2f, %.2f, ..., %.2f]\n",
                            subseq2[0], subseq2[1], subseq2[2], subseq2[subseq2.length-1]);
        }
        
        System.out.println();
    }
    
    /**
     * Demonstrates discord (anomaly) discovery.
     */
    private static void discordDiscoveryDemo() {
        System.out.println("2. Discord Discovery");
        System.out.println("===================");
        
        // Create time series with anomalies
        double[] timeSeries = createTimeSeriesWithAnomalies();
        
        System.out.printf("Time series length: %d\n", timeSeries.length);
        
        // Configure Matrix Profile for discord detection
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(15)
                .verbose(true)
                .build();
        
        // Compute Matrix Profile
        MatrixProfile.MatrixProfileResult result = mp.stamp(timeSeries);
        
        // Find discords
        System.out.println("Discovering discords...");
        MatrixProfile.DiscordResult discords = mp.findDiscords(result, 3);
        
        System.out.printf("Found %d discords:\n", discords.getDiscords().size());
        for (int i = 0; i < discords.getDiscords().size(); i++) {
            MatrixProfile.DiscordResult.Discord discord = discords.getDiscords().get(i);
            System.out.printf("  Discord %d: index %d, distance %.4f\n", 
                            i + 1, discord.getIndex(), discord.getDistance());
            
            // Print anomalous subsequence
            double[] anomaly = mp.extractSubsequence(timeSeries, discord.getIndex());
            System.out.printf("    Anomaly: [%.2f, %.2f, %.2f, ..., %.2f]\n",
                            anomaly[0], anomaly[1], anomaly[2], anomaly[anomaly.length-1]);
        }
        
        // Analyze discord distribution
        DiscordDiscovery discordDiscovery = new DiscordDiscovery(mp, true);
        DiscordDiscovery.DiscordAnalysis analysis = discordDiscovery.analyzeDiscordDistribution(result);
        
        System.out.printf("\nDiscord Distribution Analysis:\n");
        System.out.printf("  Mean distance: %.4f\n", analysis.getMean());
        System.out.printf("  Std deviation: %.4f\n", analysis.getStd());
        System.out.printf("  Median: %.4f\n", analysis.getMedian());
        System.out.printf("  95th percentile: %.4f\n", analysis.getQ95());
        System.out.printf("  Suggested thresholds:\n");
        System.out.printf("    Conservative: %.4f\n", analysis.getConservativeThreshold());
        System.out.printf("    Moderate: %.4f\n", analysis.getModerateThreshold());
        System.out.printf("    Aggressive: %.4f\n", analysis.getAggressiveThreshold());
        
        System.out.println();
    }
    
    /**
     * Demonstrates multivariate Matrix Profile analysis.
     */
    private static void multivariateAnalysisDemo() {
        System.out.println("3. Multivariate Analysis");
        System.out.println("========================");
        
        // Create multivariate time series
        double[][] multivariateSeries = createMultivariateTimeSeries();
        
        System.out.printf("Multivariate series: %d timesteps x %d dimensions\n", 
                         multivariateSeries.length, multivariateSeries[0].length);
        
        // Analyze each dimension separately
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(10)
                .verbose(false)
                .build();
        
        for (int dim = 0; dim < multivariateSeries[0].length; dim++) {
            System.out.printf("\nDimension %d analysis:\n", dim + 1);
            
            // Extract dimension
            double[] dimSeries = new double[multivariateSeries.length];
            for (int t = 0; t < multivariateSeries.length; t++) {
                dimSeries[t] = multivariateSeries[t][dim];
            }
            
            // Compute Matrix Profile for this dimension
            MatrixProfile.MatrixProfileResult result = mp.stamp(dimSeries);
            
            // Find top motif and discord for this dimension
            MatrixProfile.MotifResult motifs = mp.findMotifs(result, 1);
            MatrixProfile.DiscordResult discords = mp.findDiscords(result, 1);
            
            if (!motifs.getMotifs().isEmpty()) {
                MatrixProfile.MotifResult.MotifPair motif = motifs.getMotifs().get(0);
                System.out.printf("  Top motif: indices (%d, %d), distance %.4f\n",
                                motif.getIndex1(), motif.getIndex2(), motif.getDistance());
            }
            
            if (!discords.getDiscords().isEmpty()) {
                MatrixProfile.DiscordResult.Discord discord = discords.getDiscords().get(0);
                System.out.printf("  Top discord: index %d, distance %.4f\n",
                                discord.getIndex(), discord.getDistance());
            }
        }
        
        // Multi-dimensional motif discovery
        MotifDiscovery motifDiscovery = new MotifDiscovery(mp, true);
        System.out.println("\nMulti-dimensional motif discovery:");
        List<MotifDiscovery.MultiDimMotif> multiMotifs = 
                motifDiscovery.findMultiDimensionalMotifs(multivariateSeries, 10, 2);
        
        for (int i = 0; i < multiMotifs.size(); i++) {
            MotifDiscovery.MultiDimMotif motif = multiMotifs.get(i);
            System.out.printf("  Multi-motif %d: indices (%d, %d), distance %.4f, dimensions %s\n",
                            i + 1, motif.getIndex1(), motif.getIndex2(), motif.getDistance(),
                            Arrays.toString(motif.getDimensions()));
        }
        
        System.out.println();
    }
    
    /**
     * Demonstrates performance characteristics.
     */
    private static void performanceDemo() {
        System.out.println("4. Performance Benchmarking");
        System.out.println("===========================");
        
        int[] seriesLengths = {100, 500, 1000, 2000};
        int[] subsequenceLengths = {10, 20, 30};
        
        System.out.printf("%-12s %-15s %-15s %-15s\n", "Series Len", "Subseq Len", "Time (ms)", "Rate (ops/s)");
        System.out.println("--------------------------------------------------------");
        
        for (int seriesLen : seriesLengths) {
            for (int subseqLen : subsequenceLengths) {
                if (subseqLen >= seriesLen / 4) continue; // Skip if too large
                
                // Generate test data
                double[] testSeries = generateRandomTimeSeries(seriesLen);
                
                // Configure Matrix Profile
                MatrixProfile mp = new MatrixProfile.Builder()
                        .subsequenceLength(subseqLen)
                        .verbose(false)
                        .build();
                
                // Benchmark computation
                long startTime = System.nanoTime();
                MatrixProfile.MatrixProfileResult result = mp.stamp(testSeries);
                long elapsed = System.nanoTime() - startTime;
                
                double timeMs = elapsed / 1_000_000.0;
                double rate = 1000.0 / timeMs;
                
                System.out.printf("%-12d %-15d %-15.1f %-15.1f\n", 
                                seriesLen, subseqLen, timeMs, rate);
            }
        }
        
        System.out.println();
    }
    
    /**
     * Demonstrates advanced features like streaming detection and contextual analysis.
     */
    private static void advancedFeaturesDemo() {
        System.out.println("5. Advanced Features");
        System.out.println("===================");
        
        // Streaming discord detection
        System.out.println("Streaming discord detection:");
        
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(15)
                .verbose(false)
                .build();
        
        DiscordDiscovery discordDiscovery = new DiscordDiscovery(mp, false);
        DiscordDiscovery.StreamingDiscordDetector streamDetector = 
                discordDiscovery.createStreamingDetector(100, 2.0);
        
        // Simulate streaming data
        double[] streamData = createTimeSeriesWithAnomalies();
        int discordCount = 0;
        
        for (int i = 0; i < streamData.length; i++) {
            boolean isDiscord = streamDetector.addPoint(streamData[i]);
            if (isDiscord) {
                discordCount++;
                System.out.printf("  Discord detected at position %d (value: %.3f)\n", i, streamData[i]);
            }
        }
        
        System.out.printf("Total discords detected: %d\n", discordCount);
        
        // Variable-length motif discovery
        System.out.println("\nVariable-length motif discovery:");
        double[] varSeries = createSyntheticTimeSeriesWithMotifs();
        
        MotifDiscovery motifDiscovery = new MotifDiscovery(mp, false);
        Map<Integer, List<MatrixProfile.MotifResult.MotifPair>> varMotifs = 
                motifDiscovery.findVariableLengthMotifs(varSeries, 10, 25, 2);
        
        for (Map.Entry<Integer, List<MatrixProfile.MotifResult.MotifPair>> entry : varMotifs.entrySet()) {
            int length = entry.getKey();
            List<MatrixProfile.MotifResult.MotifPair> motifs = entry.getValue();
            
            System.out.printf("  Length %d: %d motifs found\n", length, motifs.size());
            for (int i = 0; i < Math.min(2, motifs.size()); i++) {
                MatrixProfile.MotifResult.MotifPair motif = motifs.get(i);
                System.out.printf("    Motif %d: indices (%d, %d), distance %.4f\n",
                                i + 1, motif.getIndex1(), motif.getIndex2(), motif.getDistance());
            }
        }
        
        System.out.println("\n=== Matrix Profile Analysis Complete ===");
    }
    
    // Helper methods for generating test data
    
    private static double[] createSyntheticTimeSeriesWithMotifs() {
        double[] series = new double[200];
        Random rand = new Random(42);
        
        // Generate base noise
        for (int i = 0; i < series.length; i++) {
            series[i] = rand.nextGaussian() * 0.1;
        }
        
        // Add repeated motif pattern (sine wave)
        double[] motifPattern = new double[20];
        for (int i = 0; i < motifPattern.length; i++) {
            motifPattern[i] = Math.sin(2 * Math.PI * i / motifPattern.length);
        }
        
        // Insert motif at multiple locations
        int[] motifLocations = {30, 80, 150};
        for (int loc : motifLocations) {
            for (int i = 0; i < motifPattern.length && loc + i < series.length; i++) {
                series[loc + i] += motifPattern[i];
            }
        }
        
        return series;
    }
    
    private static double[] createTimeSeriesWithAnomalies() {
        double[] series = new double[150];
        Random rand = new Random(42);
        
        // Generate normal pattern (sin wave with noise)
        for (int i = 0; i < series.length; i++) {
            series[i] = Math.sin(2 * Math.PI * i / 30.0) + rand.nextGaussian() * 0.1;
        }
        
        // Inject anomalies
        int[] anomalyLocations = {40, 90, 120};
        for (int loc : anomalyLocations) {
            for (int i = 0; i < 10 && loc + i < series.length; i++) {
                series[loc + i] += 3.0 * (rand.nextDouble() - 0.5); // Large random spike
            }
        }
        
        return series;
    }
    
    private static double[][] createMultivariateTimeSeries() {
        int length = 100;
        int dimensions = 3;
        double[][] series = new double[length][dimensions];
        Random rand = new Random(42);
        
        for (int t = 0; t < length; t++) {
            // Dimension 1: sine wave
            series[t][0] = Math.sin(2 * Math.PI * t / 20.0) + rand.nextGaussian() * 0.1;
            
            // Dimension 2: cosine wave (90 degrees out of phase)
            series[t][1] = Math.cos(2 * Math.PI * t / 20.0) + rand.nextGaussian() * 0.1;
            
            // Dimension 3: linear trend with noise
            series[t][2] = 0.02 * t + rand.nextGaussian() * 0.2;
        }
        
        return series;
    }
    
    private static double[] generateRandomTimeSeries(int length) {
        double[] series = new double[length];
        Random rand = new Random(42);
        
        for (int i = 0; i < length; i++) {
            series[i] = Math.sin(2 * Math.PI * i / 25.0) + rand.nextGaussian() * 0.2;
        }
        
        return series;
    }
}
