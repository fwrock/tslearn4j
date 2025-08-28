package org.tslearn.matrix_profile;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/**
 * Comprehensive test suite for Matrix Profile functionality.
 * 
 * Tests cover:
 * - Basic Matrix Profile computation
 * - Motif discovery algorithms
 * - Discord detection
 * - AB-join operations
 * - Edge cases and error handling
 * - Performance characteristics
 * 
 * @author TSLearn4J
 */
public class MatrixProfileTest {
    
    private MatrixProfile matrixProfile;
    private double[] testSeries;
    private double[] seriesWithMotifs;
    private double[] seriesWithAnomalies;
    
    @BeforeEach
    void setUp() {
        matrixProfile = new MatrixProfile.Builder()
                .subsequenceLength(10)
                .verbose(false)
                .build();
        
        // Create test time series
        testSeries = createTestTimeSeries();
        seriesWithMotifs = createSeriesWithKnownMotifs();
        seriesWithAnomalies = createSeriesWithKnownAnomalies();
    }
    
    @Test
    void testBasicMatrixProfileComputation() {
        MatrixProfile.MatrixProfileResult result = matrixProfile.stamp(testSeries);
        
        assertNotNull(result);
        assertNotNull(result.getMatrixProfile());
        assertNotNull(result.getProfileIndex());
        assertEquals(testSeries.length - 10 + 1, result.getMatrixProfile().length);
        assertEquals(testSeries.length - 10 + 1, result.getProfileIndex().length);
        
        // Check that all distances are non-negative
        for (double distance : result.getMatrixProfile()) {
            if (distance != Double.POSITIVE_INFINITY) {
                assertTrue(distance >= 0, "All distances should be non-negative");
            }
        }
        
        // Check that profile indices are valid
        for (int index : result.getProfileIndex()) {
            if (index != -1) {
                assertTrue(index >= 0 && index < result.getMatrixProfile().length,
                         "Profile indices should be valid");
            }
        }
    }
    
    @Test
    void testMotifDiscovery() {
        MatrixProfile.MatrixProfileResult result = matrixProfile.stamp(seriesWithMotifs);
        MatrixProfile.MotifResult motifs = matrixProfile.findMotifs(result, 3);
        
        assertNotNull(motifs);
        assertNotNull(motifs.getMotifs());
        assertTrue(motifs.getMotifs().size() <= 3, "Should find at most 3 motifs");
        
        // Check motif properties
        for (MatrixProfile.MotifResult.MotifPair motif : motifs.getMotifs()) {
            assertTrue(motif.getIndex1() >= 0);
            assertTrue(motif.getIndex2() >= 0);
            assertTrue(motif.getDistance() >= 0);
            assertNotEquals(motif.getIndex1(), motif.getIndex2());
        }
        
        // Motifs should be sorted by distance (ascending)
        List<MatrixProfile.MotifResult.MotifPair> motifList = motifs.getMotifs();
        for (int i = 1; i < motifList.size(); i++) {
            assertTrue(motifList.get(i).getDistance() >= motifList.get(i-1).getDistance(),
                     "Motifs should be sorted by distance");
        }
    }
    
    @Test
    void testDiscordDiscovery() {
        MatrixProfile.MatrixProfileResult result = matrixProfile.stamp(seriesWithAnomalies);
        MatrixProfile.DiscordResult discords = matrixProfile.findDiscords(result, 2);
        
        assertNotNull(discords);
        assertNotNull(discords.getDiscords());
        assertTrue(discords.getDiscords().size() <= 2, "Should find at most 2 discords");
        
        // Check discord properties
        for (MatrixProfile.DiscordResult.Discord discord : discords.getDiscords()) {
            assertTrue(discord.getIndex() >= 0);
            assertTrue(discord.getDistance() >= 0);
        }
        
        // Discords should be sorted by distance (descending)
        List<MatrixProfile.DiscordResult.Discord> discordList = discords.getDiscords();
        for (int i = 1; i < discordList.size(); i++) {
            assertTrue(discordList.get(i).getDistance() <= discordList.get(i-1).getDistance(),
                     "Discords should be sorted by distance (descending)");
        }
    }
    
    @Test
    void testABJoin() {
        double[] seriesA = Arrays.copyOfRange(testSeries, 0, testSeries.length / 2);
        double[] seriesB = Arrays.copyOfRange(testSeries, testSeries.length / 4, testSeries.length);
        
        MatrixProfile.MatrixProfileResult result = matrixProfile.abJoin(seriesA, seriesB);
        
        assertNotNull(result);
        assertEquals(seriesA.length - 10 + 1, result.getMatrixProfile().length);
        
        // AB-join should have no left/right profiles (they're null)
        assertNull(result.getLeftProfile());
        assertNull(result.getRightProfile());
        
        // Check distances are valid
        for (double distance : result.getMatrixProfile()) {
            if (distance != Double.POSITIVE_INFINITY) {
                assertTrue(distance >= 0);
            }
        }
    }
    
    @Test
    void testSubsequenceExtraction() {
        double[] extracted = matrixProfile.extractSubsequence(testSeries, 5);
        
        assertEquals(10, extracted.length);
        for (int i = 0; i < extracted.length; i++) {
            assertEquals(testSeries[5 + i], extracted[i], 1e-10);
        }
    }
    
    @Test
    void testBuilderConfiguration() {
        MatrixProfile mp1 = new MatrixProfile.Builder()
                .subsequenceLength(15)
                .normalize(false)
                .exclusionZone(20)
                .verbose(true)
                .build();
        
        // Test with different configuration
        MatrixProfile.MatrixProfileResult result = mp1.stamp(testSeries);
        assertNotNull(result);
        assertEquals(testSeries.length - 15 + 1, result.getMatrixProfile().length);
    }
    
    @Test
    void testEdgeCases() {
        // Test with minimum series length
        double[] shortSeries = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
        
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(10)
                .build();
        
        MatrixProfile.MatrixProfileResult result = mp.stamp(shortSeries);
        assertNotNull(result);
        assertEquals(11, result.getMatrixProfile().length);
        
        // Test error cases
        assertThrows(IllegalArgumentException.class, () -> {
            new MatrixProfile.Builder().subsequenceLength(3).build(); // Too small
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            mp.stamp(new double[]{1, 2, 3}); // Series too short
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            mp.extractSubsequence(testSeries, -1); // Invalid index
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            mp.extractSubsequence(testSeries, testSeries.length); // Index out of bounds
        });
    }
    
    @Test
    void testMotifDiscoveryEdgeCases() {
        // Test with series that has no clear motifs
        double[] randomSeries = new double[50];
        Random rand = new Random(42);
        for (int i = 0; i < randomSeries.length; i++) {
            randomSeries[i] = rand.nextGaussian();
        }
        
        MatrixProfile.MatrixProfileResult result = matrixProfile.stamp(randomSeries);
        MatrixProfile.MotifResult motifs = matrixProfile.findMotifs(result, 5);
        
        assertNotNull(motifs);
        // Should still find some motifs, even in random data
        assertTrue(motifs.getMotifs().size() <= 5);
    }
    
    @Test
    void testDiscordDiscoveryEdgeCases() {
        // Test with constant series (should have low or zero distances)
        double[] constantSeries = new double[30];
        Arrays.fill(constantSeries, 5.0);
        
        MatrixProfile.MatrixProfileResult result = matrixProfile.stamp(constantSeries);
        MatrixProfile.DiscordResult discords = matrixProfile.findDiscords(result, 2);
        
        assertNotNull(discords);
        // Constant series should have very few or no meaningful discords
        for (MatrixProfile.DiscordResult.Discord discord : discords.getDiscords()) {
            assertTrue(discord.getDistance() < 1e-10, "Constant series should have near-zero distances");
        }
    }
    
    @Test
    void testZNormalization() {
        // Test that z-normalization is working correctly
        // Create series with different scales
        double[] series1 = {1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1};
        double[] series2 = new double[series1.length];
        
        // Scale and shift series2
        for (int i = 0; i < series1.length; i++) {
            series2[i] = series1[i] * 10 + 100;
        }
        
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(8)
                .normalize(true)
                .build();
        
        MatrixProfile.MatrixProfileResult result1 = mp.stamp(series1);
        MatrixProfile.MatrixProfileResult result2 = mp.stamp(series2);
        
        // With z-normalization, the matrix profiles should be similar
        for (int i = 0; i < result1.getMatrixProfile().length; i++) {
            double diff = Math.abs(result1.getMatrixProfile()[i] - result2.getMatrixProfile()[i]);
            assertTrue(diff < 0.1, "Z-normalized results should be similar regardless of scale");
        }
    }
    
    @Test
    void testExclusionZone() {
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(10)
                .exclusionZone(5) // Custom exclusion zone
                .build();
        
        MatrixProfile.MatrixProfileResult result = mp.stamp(testSeries);
        
        // Check that no subsequence is its own nearest neighbor within exclusion zone
        int[] profileIndex = result.getProfileIndex();
        for (int i = 0; i < profileIndex.length; i++) {
            if (profileIndex[i] != -1) {
                int distance = Math.abs(i - profileIndex[i]);
                assertTrue(distance > 5, "Exclusion zone should prevent trivial matches");
            }
        }
    }
    
    @Test
    void testPerformanceCharacteristics() {
        // Test that computation time scales reasonably
        int[] sizes = {50, 100, 200};
        long[] times = new long[sizes.length];
        
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(10)
                .verbose(false)
                .build();
        
        for (int i = 0; i < sizes.length; i++) {
            double[] series = createTestSeriesOfSize(sizes[i]);
            
            long startTime = System.nanoTime();
            mp.stamp(series);
            long elapsed = System.nanoTime() - startTime;
            
            times[i] = elapsed;
        }
        
        // Time should increase with size, but not too dramatically for small sizes
        for (int i = 1; i < times.length; i++) {
            assertTrue(times[i] >= times[i-1], "Time should increase with input size");
            // For reasonable small sizes, shouldn't be more than 10x slower
            assertTrue(times[i] < 10 * times[i-1], "Performance should scale reasonably");
        }
    }
    
    // Helper methods for creating test data
    
    private double[] createTestTimeSeries() {
        double[] series = new double[50];
        for (int i = 0; i < series.length; i++) {
            series[i] = Math.sin(2 * Math.PI * i / 10.0) + Math.random() * 0.1;
        }
        return series;
    }
    
    private double[] createSeriesWithKnownMotifs() {
        double[] series = new double[60];
        
        // Background noise
        Random rand = new Random(42);
        for (int i = 0; i < series.length; i++) {
            series[i] = rand.nextGaussian() * 0.1;
        }
        
        // Insert known motif pattern
        double[] motif = {1, 2, 3, 2, 1, 0, -1, 0, 1, 2};
        
        // Insert at positions 10 and 35
        System.arraycopy(motif, 0, series, 10, motif.length);
        System.arraycopy(motif, 0, series, 35, motif.length);
        
        return series;
    }
    
    private double[] createSeriesWithKnownAnomalies() {
        double[] series = new double[50];
        
        // Normal pattern
        for (int i = 0; i < series.length; i++) {
            series[i] = Math.sin(2 * Math.PI * i / 15.0);
        }
        
        // Insert anomalies
        series[20] = 5.0; // Spike
        series[21] = 5.0;
        series[22] = 5.0;
        
        return series;
    }
    
    private double[] createTestSeriesOfSize(int size) {
        double[] series = new double[size];
        for (int i = 0; i < size; i++) {
            series[i] = Math.sin(2 * Math.PI * i / 20.0) + Math.random() * 0.1;
        }
        return series;
    }
}
