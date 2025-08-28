package org.tslearn.metrics;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Lower Bound functions for DTW to speed up time series similarity search.
 * 
 * These functions provide fast lower bounds that can be used to prune
 * unnecessary DTW calculations in large datasets.
 * 
 * Based on Keogh's research and the Python tslearn implementation.
 */
public class DTWLowerBound {
    
    private static final Logger logger = LoggerFactory.getLogger(DTWLowerBound.class);
    
    /**
     * LB_Keogh lower bound for DTW with Sakoe-Chiba constraint
     * 
     * This provides a fast lower bound estimation that can be used to
     * prune DTW calculations. If LB_Keogh(q,c) > best_so_far, then
     * DTW(q,c) > best_so_far and we can skip the full DTW calculation.
     * 
     * @param query Query time series
     * @param candidate Candidate time series  
     * @param bandWidth Sakoe-Chiba band width
     * @return Lower bound estimate for DTW distance
     */
    public static double lbKeogh(double[] query, double[] candidate, int bandWidth) {
        if (query.length != candidate.length) {
            throw new IllegalArgumentException("Time series must have the same length for LB_Keogh");
        }
        
        int n = query.length;
        double sum = 0.0;
        
        // Create envelope for the candidate series
        double[] upper = new double[n];
        double[] lower = new double[n];
        
        createEnvelope(candidate, bandWidth, upper, lower);
        
        // Calculate lower bound
        for (int i = 0; i < n; i++) {
            if (query[i] > upper[i]) {
                double diff = query[i] - upper[i];
                sum += diff * diff;
            } else if (query[i] < lower[i]) {
                double diff = lower[i] - query[i];
                sum += diff * diff;
            }
            // If lower[i] <= query[i] <= upper[i], no contribution to lower bound
        }
        
        return Math.sqrt(sum);
    }
    
    /**
     * Create envelope (upper and lower bounds) for a time series
     * given a Sakoe-Chiba band constraint
     */
    private static void createEnvelope(double[] series, int bandWidth, double[] upper, double[] lower) {
        int n = series.length;
        
        for (int i = 0; i < n; i++) {
            int start = Math.max(0, i - bandWidth);
            int end = Math.min(n - 1, i + bandWidth);
            
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            
            for (int j = start; j <= end; j++) {
                min = Math.min(min, series[j]);
                max = Math.max(max, series[j]);
            }
            
            lower[i] = min;
            upper[i] = max;
        }
    }
    
    /**
     * LB_Improved lower bound - tighter bound than LB_Keogh
     * 
     * This uses both query and candidate envelopes for a tighter bound
     */
    public static double lbImproved(double[] query, double[] candidate, int bandWidth) {
        double lb1 = lbKeogh(query, candidate, bandWidth);
        double lb2 = lbKeogh(candidate, query, bandWidth);
        return Math.max(lb1, lb2);
    }
    
    /**
     * LB_Yi lower bound - another fast lower bound estimation
     * 
     * Based on the first and last elements of the time series
     */
    public static double lbYi(double[] query, double[] candidate) {
        if (query.length == 0 || candidate.length == 0) {
            return 0.0;
        }
        
        double diffFirst = query[0] - candidate[0];
        double diffLast = query[query.length - 1] - candidate[candidate.length - 1];
        
        return Math.sqrt(diffFirst * diffFirst + diffLast * diffLast);
    }
    
    /**
     * LB_New lower bound using PAA (Piecewise Aggregate Approximation)
     * 
     * @param query Query time series
     * @param candidate Candidate time series
     * @param segmentSize Size of PAA segments
     * @return Lower bound estimate
     */
    public static double lbPAA(double[] query, double[] candidate, int segmentSize) {
        if (query.length != candidate.length) {
            throw new IllegalArgumentException("Time series must have the same length for LB_PAA");
        }
        
        int n = query.length;
        int numSegments = (n + segmentSize - 1) / segmentSize; // Ceiling division
        
        double sum = 0.0;
        
        for (int i = 0; i < numSegments; i++) {
            int start = i * segmentSize;
            int end = Math.min((i + 1) * segmentSize, n);
            
            // Calculate PAA values for this segment
            double queryPAA = 0.0;
            double candidatePAA = 0.0;
            
            for (int j = start; j < end; j++) {
                queryPAA += query[j];
                candidatePAA += candidate[j];
            }
            
            queryPAA /= (end - start);
            candidatePAA /= (end - start);
            
            double diff = queryPAA - candidatePAA;
            sum += diff * diff * (end - start); // Weight by segment length
        }
        
        return Math.sqrt(sum);
    }
    
    /**
     * Fast lower bound cascade - combines multiple lower bounds
     * 
     * Uses increasingly expensive but tighter bounds to prune as early as possible
     * 
     * @param query Query time series
     * @param candidate Candidate time series
     * @param bandWidth Sakoe-Chiba band width
     * @param threshold Current best distance (for pruning)
     * @return Lower bound estimate, or Double.POSITIVE_INFINITY if can be pruned
     */
    public static double lbCascade(double[] query, double[] candidate, int bandWidth, double threshold) {
        // Level 1: Very fast LB_Yi
        double lb = lbYi(query, candidate);
        if (lb >= threshold) {
            return Double.POSITIVE_INFINITY; // Can prune
        }
        
        // Level 2: Fast LB_PAA with large segments
        lb = Math.max(lb, lbPAA(query, candidate, Math.max(1, query.length / 10)));
        if (lb >= threshold) {
            return Double.POSITIVE_INFINITY; // Can prune
        }
        
        // Level 3: LB_Keogh
        lb = Math.max(lb, lbKeogh(query, candidate, bandWidth));
        if (lb >= threshold) {
            return Double.POSITIVE_INFINITY; // Can prune
        }
        
        // Level 4: LB_Improved (most expensive but tightest)
        lb = Math.max(lb, lbImproved(query, candidate, bandWidth));
        
        return lb;
    }
    
    /**
     * Statistics class to track lower bound performance
     */
    public static class LBStats {
        private int totalComparisons = 0;
        private int lbYiPrunes = 0;
        private int lbPAAPrunes = 0;
        private int lbKeoghPrunes = 0;
        private int lbImprovedPrunes = 0;
        private int dtwCalculations = 0;
        
        public void incrementTotal() { totalComparisons++; }
        public void incrementLBYiPrunes() { lbYiPrunes++; }
        public void incrementLBPAAPrunes() { lbPAAPrunes++; }
        public void incrementLBKeoghPrunes() { lbKeoghPrunes++; }
        public void incrementLBImprovedPrunes() { lbImprovedPrunes++; }
        public void incrementDTWCalculations() { dtwCalculations++; }
        
        public double getPruningRate() {
            if (totalComparisons == 0) return 0.0;
            return (double)(totalComparisons - dtwCalculations) / totalComparisons;
        }
        
        @Override
        public String toString() {
            return String.format(
                "LBStats{total=%d, LB_Yi=%d, LB_PAA=%d, LB_Keogh=%d, LB_Improved=%d, DTW=%d, pruningRate=%.2f%%}",
                totalComparisons, lbYiPrunes, lbPAAPrunes, lbKeoghPrunes, lbImprovedPrunes, 
                dtwCalculations, getPruningRate() * 100
            );
        }
    }
}
