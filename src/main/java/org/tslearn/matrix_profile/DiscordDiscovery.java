package org.tslearn.matrix_profile;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Advanced discord (anomaly) discovery utilities for Matrix Profile.
 */
public class DiscordDiscovery {

    private final MatrixProfile mp;
    private final boolean verbose;

    public DiscordDiscovery(MatrixProfile mp, boolean verbose) {
        this.mp = mp;
        this.verbose = verbose;
    }

    /**
     * Analyzes the distribution of discord distances to suggest thresholds.
     */
    public DiscordAnalysis analyzeDiscordDistribution(MatrixProfile.MatrixProfileResult result) {
        double[] profile = result.getMatrixProfile();
        double[] finite = Arrays.stream(profile)
                .filter(v -> !Double.isInfinite(v) && !Double.isNaN(v))
                .sorted()
                .toArray();

        if (finite.length == 0) {
            return new DiscordAnalysis(0, 0, 0, 0, 0, 0, 0);
        }

        double mean = Arrays.stream(finite).average().orElse(0);
        double variance = Arrays.stream(finite).map(v -> (v - mean) * (v - mean)).average().orElse(0);
        double std = Math.sqrt(variance);
        double median = finite[finite.length / 2];
        double q95 = finite[(int) (finite.length * 0.95)];

        double conservative = mean + 3 * std;
        double moderate = mean + 2 * std;
        double aggressive = mean + std;

        return new DiscordAnalysis(mean, std, median, q95, conservative, moderate, aggressive);
    }

    /**
     * Creates a streaming discord detector.
     */
    public StreamingDiscordDetector createStreamingDetector(int windowSize, double threshold) {
        return new StreamingDiscordDetector(mp, windowSize, threshold);
    }

    /**
     * Analysis results for discord distribution.
     */
    public static class DiscordAnalysis {
        private final double mean;
        private final double std;
        private final double median;
        private final double q95;
        private final double conservativeThreshold;
        private final double moderateThreshold;
        private final double aggressiveThreshold;

        public DiscordAnalysis(double mean, double std, double median, double q95,
                               double conservativeThreshold, double moderateThreshold,
                               double aggressiveThreshold) {
            this.mean = mean;
            this.std = std;
            this.median = median;
            this.q95 = q95;
            this.conservativeThreshold = conservativeThreshold;
            this.moderateThreshold = moderateThreshold;
            this.aggressiveThreshold = aggressiveThreshold;
        }

        public double getMean() { return mean; }
        public double getStd() { return std; }
        public double getMedian() { return median; }
        public double getQ95() { return q95; }
        public double getConservativeThreshold() { return conservativeThreshold; }
        public double getModerateThreshold() { return moderateThreshold; }
        public double getAggressiveThreshold() { return aggressiveThreshold; }
    }

    /**
     * Sliding-window streaming discord detector.
     */
    public static class StreamingDiscordDetector {
        private final MatrixProfile mp;
        private final int windowSize;
        private final double threshold;
        private final List<Double> buffer = new ArrayList<>();

        public StreamingDiscordDetector(MatrixProfile mp, int windowSize, double threshold) {
            this.mp = mp;
            this.windowSize = windowSize;
            this.threshold = threshold;
        }

        /**
         * Adds a new data point and returns true if it is considered a discord.
         */
        public boolean addPoint(double value) {
            buffer.add(value);
            if (buffer.size() < windowSize + mp.getSubsequenceLength()) {
                return false;
            }

            // Keep only the most recent windowSize + subsequenceLength points
            while (buffer.size() > windowSize + mp.getSubsequenceLength()) {
                buffer.remove(0);
            }

            double[] series = buffer.stream().mapToDouble(Double::doubleValue).toArray();
            MatrixProfile.MatrixProfileResult result = mp.stamp(series);
            double[] profile = result.getMatrixProfile();

            // Check if the last subsequence distance exceeds threshold
            if (profile.length == 0) return false;
            double lastDist = profile[profile.length - 1];
            return lastDist > threshold;
        }
    }
}
