package org.tslearn.metrics;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Tests for DTW Lower Bound functions
 */
public class DTWLowerBoundTest {
    
    private double[] ts1;
    private double[] ts2;
    private double[] ts3;
    private DTW dtw;
    
    @BeforeEach
    public void setUp() {
        ts1 = new double[]{1.0, 2.0, 3.0, 2.0, 1.0};
        ts2 = new double[]{1.0, 2.0, 3.0, 2.0, 1.0}; // Identical
        ts3 = new double[]{2.0, 3.0, 4.0, 3.0, 2.0}; // Shifted
        dtw = new DTW();
    }
    
    @Test
    public void testLBKeoghIsLowerBound() {
        int bandWidth = 2;
        double lbDistance = DTWLowerBound.lbKeogh(ts1, ts3, bandWidth);
        double dtwDistance = new DTW(bandWidth).distance(ts1, ts3);
        
        assertTrue(lbDistance <= dtwDistance + 1e-10, 
                  "LB_Keogh should be a lower bound for DTW distance");
    }
    
    @Test
    public void testLBKeoghIdenticalSeries() {
        double lbDistance = DTWLowerBound.lbKeogh(ts1, ts2, 1);
        assertEquals(0.0, lbDistance, 1e-10, 
                    "LB_Keogh should be 0 for identical series");
    }
    
    @Test
    public void testLBImprovedTighterThanKeogh() {
        int bandWidth = 2;
        double lbKeogh = DTWLowerBound.lbKeogh(ts1, ts3, bandWidth);
        double lbImproved = DTWLowerBound.lbImproved(ts1, ts3, bandWidth);
        
        assertTrue(lbImproved >= lbKeogh - 1e-10, 
                  "LB_Improved should be at least as tight as LB_Keogh");
    }
    
    @Test
    public void testLBYi() {
        double lbYi = DTWLowerBound.lbYi(ts1, ts3);
        assertTrue(lbYi >= 0, "LB_Yi should be non-negative");
        
        // For identical series, LB_Yi should be 0
        double lbYiIdentical = DTWLowerBound.lbYi(ts1, ts2);
        assertEquals(0.0, lbYiIdentical, 1e-10, "LB_Yi should be 0 for identical series");
    }
    
    @Test
    public void testLBPAA() {
        double lbPAA = DTWLowerBound.lbPAA(ts1, ts3, 2);
        assertTrue(lbPAA >= 0, "LB_PAA should be non-negative");
        
        // For identical series, LB_PAA should be 0
        double lbPAAIdentical = DTWLowerBound.lbPAA(ts1, ts2, 2);
        assertEquals(0.0, lbPAAIdentical, 1e-10, "LB_PAA should be 0 for identical series");
    }
    
    @Test
    public void testLBCascade() {
        double threshold = 10.0;
        double lbCascade = DTWLowerBound.lbCascade(ts1, ts3, 2, threshold);
        
        assertTrue(lbCascade >= 0 || lbCascade == Double.POSITIVE_INFINITY, 
                  "LB_Cascade should be non-negative or infinity");
    }
    
    @Test
    public void testLBCascadePruning() {
        double lowThreshold = 0.1;
        double lbCascade = DTWLowerBound.lbCascade(ts1, ts3, 1, lowThreshold);
        
        // Should prune if lower bound exceeds threshold
        if (lbCascade == Double.POSITIVE_INFINITY) {
            // Verify that actual DTW distance would indeed exceed threshold
            double actualDTW = new DTW(1).distance(ts1, ts3);
            assertTrue(actualDTW > lowThreshold, 
                      "Pruning should only occur when actual DTW exceeds threshold");
        }
    }
    
    @Test
    public void testLBStats() {
        DTWLowerBound.LBStats stats = new DTWLowerBound.LBStats();
        
        assertEquals(0, stats.getPruningRate(), 1e-10, "Initial pruning rate should be 0");
        
        stats.incrementTotal();
        stats.incrementLBYiPrunes();
        
        assertNotNull(stats.toString(), "Stats toString should not be null");
    }
    
    @Test
    public void testDifferentLengthSeriesThrowsException() {
        double[] short1 = {1.0, 2.0};
        double[] long1 = {1.0, 2.0, 3.0, 4.0, 5.0};
        
        assertThrows(IllegalArgumentException.class, () -> {
            DTWLowerBound.lbKeogh(short1, long1, 1);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            DTWLowerBound.lbPAA(short1, long1, 2);
        });
    }
    
    @Test
    public void testEmptySeries() {
        double[] empty1 = {};
        double[] empty2 = {};
        
        double lbYi = DTWLowerBound.lbYi(empty1, empty2);
        assertEquals(0.0, lbYi, 1e-10, "LB_Yi should be 0 for empty series");
    }
    
    @Test
    public void testLowerBoundPerformance() {
        // Create longer series for performance testing
        double[] long1 = new double[100];
        double[] long2 = new double[100];
        
        for (int i = 0; i < 100; i++) {
            long1[i] = Math.sin(2 * Math.PI * i / 10.0);
            long2[i] = Math.cos(2 * Math.PI * i / 10.0);
        }
        
        long startTime = System.nanoTime();
        double lbKeogh = DTWLowerBound.lbKeogh(long1, long2, 10);
        long lbTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        double dtwDistance = new DTW(10).distance(long1, long2);
        long dtwTime = System.nanoTime() - startTime;
        
        assertTrue(lbKeogh <= dtwDistance + 1e-10, "LB_Keogh should be lower bound");
        assertTrue(lbTime < dtwTime, "Lower bound should be faster than full DTW");
    }
}
