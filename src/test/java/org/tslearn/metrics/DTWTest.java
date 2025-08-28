package org.tslearn.metrics;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Comprehensive tests for DTW implementation with optimizations
 */
public class DTWTest {
    
    private DTW dtw;
    private double[] ts1;
    private double[] ts2;
    private double[] ts3;
    
    @BeforeEach
    public void setUp() {
        dtw = new DTW();
        
        // Simple test time series
        ts1 = new double[]{1.0, 2.0, 3.0, 2.0, 1.0};
        ts2 = new double[]{1.0, 2.0, 3.0, 2.0, 1.0}; // Identical to ts1
        ts3 = new double[]{2.0, 3.0, 4.0, 3.0, 2.0}; // Shifted version
    }
    
    @Test
    public void testIdenticalSeriesDistance() {
        double distance = dtw.distance(ts1, ts2);
        assertEquals(0.0, distance, 1e-10, "Distance between identical series should be 0");
    }
    
    @Test
    public void testSelfDistance() {
        double distance = dtw.distance(ts1, ts1);
        assertEquals(0.0, distance, 1e-10, "Self distance should be 0");
    }
    
    @Test
    public void testSymmetry() {
        double dist1 = dtw.distance(ts1, ts3);
        double dist2 = dtw.distance(ts3, ts1);
        assertEquals(dist1, dist2, 1e-10, "DTW distance should be symmetric");
    }
    
    @Test
    public void testTriangleInequality() {
        double[] ts4 = {0.0, 1.0, 2.0, 1.0, 0.0};
        
        double d12 = dtw.distance(ts1, ts2);
        double d23 = dtw.distance(ts2, ts4);
        double d13 = dtw.distance(ts1, ts4);
        
        // DTW doesn't satisfy triangle inequality in general, but this is a sanity check
        assertTrue(d13 >= 0, "Distance should be non-negative");
    }
    
    @Test
    public void testEmptySeriesThrowsException() {
        double[] empty = {};
        assertThrows(IllegalArgumentException.class, () -> {
            dtw.distance(ts1, empty);
        });
    }
    
    @Test
    public void testNullSeriesThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            dtw.distance(ts1, null);
        });
    }
    
    @Test
    public void testSakoeChibaConstraint() {
        DTW constrainedDTW = new DTW(2); // Sakoe-Chiba band width of 2
        
        double unconstrainedDistance = dtw.distance(ts1, ts3);
        double constrainedDistance = constrainedDTW.distance(ts1, ts3);
        
        // Constrained DTW should give same or higher distance
        assertTrue(constrainedDistance >= unconstrainedDistance - 1e-10, 
                  "Constrained DTW should not give lower distance");
    }
    
    @Test
    public void testItakuraConstraint() {
        DTW itakuraDTW = new DTW(DTW.GlobalConstraint.ITAKURA, 0.0, false, Double.POSITIVE_INFINITY);
        
        double distance = itakuraDTW.distance(ts1, ts3);
        assertTrue(distance >= 0, "Itakura constrained DTW should give non-negative distance");
    }
    
    @Test
    public void testEarlyTermination() {
        DTW earlyTermDTW = new DTW(DTW.GlobalConstraint.NONE, 0.0, true, 1.0);
        
        double[] largeDiff1 = {0.0, 0.0, 0.0, 0.0, 0.0};
        double[] largeDiff2 = {10.0, 10.0, 10.0, 10.0, 10.0};
        
        double distance = earlyTermDTW.distance(largeDiff1, largeDiff2);
        assertEquals(Double.POSITIVE_INFINITY, distance, 
                    "Early termination should return infinity for large distances");
    }
    
    @Test
    public void testDifferentLengthSeries() {
        double[] short1 = {1.0, 2.0, 3.0};
        double[] long1 = {1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0};
        
        double distance = dtw.distance(short1, long1);
        assertTrue(distance >= 0, "DTW should handle different length series");
        assertTrue(Double.isFinite(distance), "DTW distance should be finite");
    }
    
    @Test
    public void testPathGeneration() {
        DTW.DTWResult result = dtw.distanceWithPath(ts1, ts3);
        
        assertNotNull(result, "DTW result should not be null");
        assertTrue(result.getDistance() >= 0, "DTW distance should be non-negative");
        assertTrue(result.getPathLength() > 0, "DTW path should not be empty");
        
        int[][] path = result.getPath();
        assertNotNull(path, "DTW path should not be null");
        
        // Check path validity
        for (int[] step : path) {
            assertEquals(2, step.length, "Each path step should have 2 indices");
            assertTrue(step[0] >= 0 && step[0] < ts1.length, "Path i-index should be valid");
            assertTrue(step[1] >= 0 && step[1] < ts3.length, "Path j-index should be valid");
        }
        
        // Path should start at (0,0) and end at (n-1,m-1)
        assertEquals(0, path[0][0], "Path should start at i=0");
        assertEquals(0, path[0][1], "Path should start at j=0");
        assertEquals(ts1.length - 1, path[path.length - 1][0], "Path should end at i=n-1");
        assertEquals(ts3.length - 1, path[path.length - 1][1], "Path should end at j=m-1");
    }
    
    @Test
    public void testPerformanceWithLongSeries() {
        // Create longer time series for performance testing
        double[] longTs1 = new double[100];
        double[] longTs2 = new double[100];
        
        for (int i = 0; i < 100; i++) {
            longTs1[i] = Math.sin(2 * Math.PI * i / 10.0);
            longTs2[i] = Math.sin(2 * Math.PI * i / 10.0 + Math.PI / 4); // Phase shifted
        }
        
        long startTime = System.currentTimeMillis();
        double distance = dtw.distance(longTs1, longTs2);
        long endTime = System.currentTimeMillis();
        
        assertTrue(distance >= 0, "DTW distance should be non-negative");
        assertTrue(endTime - startTime < 1000, "DTW should complete within reasonable time");
    }
    
    @Test
    public void testConstraintParameters() {
        DTW dtwSakoeChiba = new DTW(5);
        assertEquals(DTW.GlobalConstraint.SAKOE_CHIBA, dtwSakoeChiba.getGlobalConstraint());
        assertEquals(5.0, dtwSakoeChiba.getGlobalConstraintParam(), 1e-10);
        
        DTW dtwWithEarlyTerm = new DTW(DTW.GlobalConstraint.NONE, 0.0, true, 10.0);
        assertTrue(dtwWithEarlyTerm.isEarlyTerminationEnabled());
        assertEquals(10.0, dtwWithEarlyTerm.getEarlyTerminationThreshold(), 1e-10);
    }
    
    @Test
    public void testStepPattern() {
        // Test that DTW follows the correct step pattern (diagonal, horizontal, vertical)
        double[] simple1 = {1.0, 2.0};
        double[] simple2 = {1.0, 2.0};
        
        DTW.DTWResult result = dtw.distanceWithPath(simple1, simple2);
        int[][] path = result.getPath();
        
        // For identical 2-element series, optimal path should be diagonal
        assertEquals(2, path.length, "Path length should be 2 for 2x2 alignment");
        assertArrayEquals(new int[]{0, 0}, path[0], "First step should be (0,0)");
        assertArrayEquals(new int[]{1, 1}, path[1], "Second step should be (1,1)");
    }
}
