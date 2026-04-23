package org.tslearn.matrix_profile;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Advanced motif discovery utilities for Matrix Profile.
 */
public class MotifDiscovery {

    private final MatrixProfile mp;
    private final boolean verbose;

    public MotifDiscovery(MatrixProfile mp, boolean verbose) {
        this.mp = mp;
        this.verbose = verbose;
    }

    /**
     * Finds multi-dimensional motifs across all dimensions of a multivariate time series.
     *
     * @param multivariateSeries timesteps x dimensions array
     * @param subsequenceLength  length of subsequences to compare
     * @param numMotifs          number of top motifs to return
     * @return list of multi-dimensional motif pairs
     */
    public List<MultiDimMotif> findMultiDimensionalMotifs(double[][] multivariateSeries,
                                                          int subsequenceLength, int numMotifs) {
        int timesteps = multivariateSeries.length;
        int dims = multivariateSeries[0].length;

        // Build a per-dimension matrix profile and sum them
        MatrixProfile dimMP = new MatrixProfile.Builder()
                .subsequenceLength(subsequenceLength)
                .verbose(false)
                .build();

        int profileLength = timesteps - subsequenceLength + 1;
        double[] combined = new double[profileLength];
        int[] combinedIdx = new int[profileLength];
        Arrays.fill(combined, Double.POSITIVE_INFINITY);

        for (int d = 0; d < dims; d++) {
            double[] dimSeries = new double[timesteps];
            for (int t = 0; t < timesteps; t++) {
                dimSeries[t] = multivariateSeries[t][d];
            }
            MatrixProfile.MatrixProfileResult res = dimMP.stamp(dimSeries);
            double[] profile = res.getMatrixProfile();
            int[] idx = res.getProfileIndex();

            for (int i = 0; i < profileLength && i < profile.length; i++) {
                if (profile[i] < combined[i]) {
                    combined[i] = profile[i];
                    combinedIdx[i] = idx[i];
                }
            }
        }

        // Extract top motifs
        List<MultiDimMotif> result = new ArrayList<>();
        boolean[] used = new boolean[profileLength];
        int exclusionZone = subsequenceLength / 2;

        for (int count = 0; count < numMotifs; count++) {
            double minVal = Double.POSITIVE_INFINITY;
            int minIdx = -1;
            for (int i = 0; i < profileLength; i++) {
                if (!used[i] && combined[i] < minVal) {
                    minVal = combined[i];
                    minIdx = i;
                }
            }
            if (minIdx < 0) break;

            int matchIdx = combinedIdx[minIdx];
            int[] allDims = new int[dims];
            for (int d = 0; d < dims; d++) allDims[d] = d;

            result.add(new MultiDimMotif(minIdx, matchIdx, minVal, allDims));

            for (int i = 0; i < profileLength; i++) {
                if (Math.abs(i - minIdx) < exclusionZone || Math.abs(i - matchIdx) < exclusionZone) {
                    used[i] = true;
                }
            }
        }

        return result;
    }

    /**
     * Finds motifs for variable subsequence lengths in [minLen, maxLen].
     */
    public Map<Integer, List<MatrixProfile.MotifResult.MotifPair>> findVariableLengthMotifs(
            double[] timeSeries, int minLen, int maxLen, int numMotifs) {

        Map<Integer, List<MatrixProfile.MotifResult.MotifPair>> result = new HashMap<>();

        for (int len = minLen; len <= maxLen; len++) {
            if (timeSeries.length < 2 * len) continue;

            MatrixProfile lenMP = new MatrixProfile.Builder()
                    .subsequenceLength(len)
                    .verbose(false)
                    .build();

            MatrixProfile.MotifResult motifResult = lenMP.findMotifs(timeSeries, numMotifs);
            result.put(len, motifResult.getMotifs());
        }

        return result;
    }

    /**
     * Represents a motif pair found across multiple dimensions.
     */
    public static class MultiDimMotif {
        private final int index1;
        private final int index2;
        private final double distance;
        private final int[] dimensions;

        public MultiDimMotif(int index1, int index2, double distance, int[] dimensions) {
            this.index1 = index1;
            this.index2 = index2;
            this.distance = distance;
            this.dimensions = dimensions;
        }

        public int getIndex1() { return index1; }
        public int getIndex2() { return index2; }
        public double getDistance() { return distance; }
        public int[] getDimensions() { return dimensions; }
    }
}
