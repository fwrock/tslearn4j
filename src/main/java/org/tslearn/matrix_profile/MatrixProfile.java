package org.tslearn.matrix_profile;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Basic Matrix Profile implementation.
 */
public class MatrixProfile {
    
    private final int subsequenceLength;
    private final boolean verbose;
    private final boolean normalize;
    private final int exclusionZone;
    
    protected MatrixProfile() {
        this(new Builder());
    }
    
    protected MatrixProfile(Builder builder) {
        this.subsequenceLength = builder.subsequenceLength;
        this.verbose = builder.verbose;
        this.normalize = builder.normalize;
        this.exclusionZone = builder.exclusionZone >= 0 ? builder.exclusionZone : builder.subsequenceLength / 2;
    }
    
    public static class Builder {
        private int subsequenceLength = 10;
        private boolean verbose = false;
        private boolean normalize = false;
        private boolean useFFT = false;
        private int exclusionZone = -1; // -1 means auto (m/2)
        
        public Builder() {}
        
        public Builder(double[] series, int subsequenceLength) {
            this.subsequenceLength = subsequenceLength;
        }
        
        public Builder subsequenceLength(int length) {
            this.subsequenceLength = length;
            return this;
        }
        
        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }
        
        public Builder normalize(boolean normalize) {
            this.normalize = normalize;
            return this;
        }
        
        public Builder useFFT(boolean useFFT) {
            this.useFFT = useFFT;
            return this;
        }
        
        public Builder exclusionZone(int exclusionZone) {
            this.exclusionZone = exclusionZone;
            return this;
        }
        
        public MatrixProfile build() {
            if (subsequenceLength < 4) {
                throw new IllegalArgumentException("Subsequence length must be at least 4");
            }
            return new MatrixProfile(this);
        }
    }
    
    protected int getSubsequenceLength() {
        return subsequenceLength;
    }
    
    protected boolean isVerbose() {
        return verbose;
    }
    
    public double[] extractSubsequence(double[] timeSeries, int index) {
        if (index < 0 || index + subsequenceLength > timeSeries.length) {
            throw new IllegalArgumentException("Invalid subsequence index: " + index);
        }
        return Arrays.copyOfRange(timeSeries, index, index + subsequenceLength);
    }
    
    public MatrixProfileResult stamp(double[] timeSeries) {
        if (timeSeries == null || timeSeries.length == 0) {
            throw new IllegalArgumentException("Time series cannot be null or empty");
        }
        
        int n = timeSeries.length;
        int profileLength = n - subsequenceLength + 1;
        
        if (profileLength < 1) {
            throw new IllegalArgumentException(
                "Time series too short for subsequence length " + subsequenceLength);
        }
        
        if (verbose) {
            System.out.printf("Computing Matrix Profile: series length=%d\n", n);
        }
        
        double[] matrixProfile = new double[profileLength];
        int[] profileIndex = new int[profileLength];
        
        // Precompute per-subsequence mean and std when normalizing
        double[] means = null;
        double[] stds = null;
        if (normalize) {
            means = new double[profileLength];
            stds = new double[profileLength];
            for (int i = 0; i < profileLength; i++) {
                double sum = 0;
                for (int k = 0; k < subsequenceLength; k++) sum += timeSeries[i + k];
                means[i] = sum / subsequenceLength;
                double var = 0;
                for (int k = 0; k < subsequenceLength; k++) {
                    double d = timeSeries[i + k] - means[i];
                    var += d * d;
                }
                stds[i] = Math.sqrt(var / subsequenceLength);
            }
        }
        
        // Compute distances
        for (int i = 0; i < profileLength; i++) {
            double minDistance = Double.POSITIVE_INFINITY;
            int minIndex = -1;
            
            for (int j = 0; j < profileLength; j++) {
                if (Math.abs(i - j) < exclusionZone) continue;
                
                double distance = 0;
                if (normalize && stds[i] > 0 && stds[j] > 0) {
                    for (int k = 0; k < subsequenceLength; k++) {
                        double zi = (timeSeries[i + k] - means[i]) / stds[i];
                        double zj = (timeSeries[j + k] - means[j]) / stds[j];
                        double diff = zi - zj;
                        distance += diff * diff;
                    }
                } else {
                    for (int k = 0; k < subsequenceLength; k++) {
                        double diff = timeSeries[i + k] - timeSeries[j + k];
                        distance += diff * diff;
                    }
                }
                distance = Math.sqrt(distance);
                
                if (distance < minDistance) {
                    minDistance = distance;
                    minIndex = j;
                }
            }
            
            matrixProfile[i] = minDistance;
            profileIndex[i] = minIndex;
        }
        
        return new MatrixProfileResult(matrixProfile, profileIndex);
    }
    
    public MotifResult findMotifs(double[] timeSeries, int numMotifs) {
        MatrixProfileResult result = stamp(timeSeries);
        return findMotifs(result, numMotifs);
    }
    
    public MotifResult findMotifs(MatrixProfileResult result, int numMotifs) {
        double[] matrixProfile = result.getMatrixProfile();
        int[] profileIndex = result.getProfileIndex();
        List<MotifResult.MotifPair> motifs = new ArrayList<>();
        boolean[] used = new boolean[matrixProfile.length];
        int ez = this.exclusionZone;

        for (int count = 0; count < numMotifs; count++) {
            double minVal = Double.POSITIVE_INFINITY;
            int minIdx = -1;
            for (int i = 0; i < matrixProfile.length; i++) {
                if (!used[i] && matrixProfile[i] < minVal) {
                    minVal = matrixProfile[i];
                    minIdx = i;
                }
            }
            if (minIdx < 0) break;
            int matchIdx = profileIndex[minIdx];
            motifs.add(new MotifResult.MotifPair(minIdx, matchIdx, minVal));
            for (int i = 0; i < matrixProfile.length; i++) {
                if (Math.abs(i - minIdx) < ez || Math.abs(i - matchIdx) < ez) {
                    used[i] = true;
                }
            }
        }
        return new MotifResult(motifs, matrixProfile);
    }
    
    public DiscordResult findDiscords(double[] timeSeries, int numDiscords) {
        MatrixProfileResult result = stamp(timeSeries);
        return findDiscords(result, numDiscords);
    }
    
    public DiscordResult findDiscords(MatrixProfileResult result, int numDiscords) {
        double[] matrixProfile = result.getMatrixProfile();
        List<DiscordResult.Discord> discords = new ArrayList<>();
        boolean[] used = new boolean[matrixProfile.length];
        int ez = this.exclusionZone;

        for (int count = 0; count < numDiscords; count++) {
            double maxVal = Double.NEGATIVE_INFINITY;
            int maxIdx = -1;
            for (int i = 0; i < matrixProfile.length; i++) {
                if (!used[i] && !Double.isInfinite(matrixProfile[i]) && matrixProfile[i] > maxVal) {
                    maxVal = matrixProfile[i];
                    maxIdx = i;
                }
            }
            if (maxIdx < 0) break;
            discords.add(new DiscordResult.Discord(maxIdx, maxVal));
            for (int i = 0; i < matrixProfile.length; i++) {
                if (Math.abs(i - maxIdx) < ez) {
                    used[i] = true;
                }
            }
        }
        return new DiscordResult(discords, matrixProfile);
    }
    
    public MatrixProfileResult abJoin(double[] seriesA, double[] seriesB) {
        if (seriesA == null || seriesA.length < subsequenceLength ||
            seriesB == null || seriesB.length < subsequenceLength) {
            throw new IllegalArgumentException("Series too short for given subsequence length");
        }
        int m = subsequenceLength;
        int profileLengthA = seriesA.length - m + 1;
        int profileLengthB = seriesB.length - m + 1;
        double[] mp = new double[profileLengthA];
        int[] pi = new int[profileLengthA];
        Arrays.fill(mp, Double.POSITIVE_INFINITY);
        for (int i = 0; i < profileLengthA; i++) {
            for (int j = 0; j < profileLengthB; j++) {
                double dist = 0;
                for (int k = 0; k < m; k++) {
                    double diff = seriesA[i + k] - seriesB[j + k];
                    dist += diff * diff;
                }
                dist = Math.sqrt(dist);
                if (dist < mp[i]) {
                    mp[i] = dist;
                    pi[i] = j;
                }
            }
        }
        return new MatrixProfileResult(mp, pi, null, null, null, null, m, seriesA);
    }
    
    // Result classes
    public static class MatrixProfileResult {
        public final double[] matrixProfile;
        private final int[] profileIndex;
        private final double[] leftProfile;
        private final int[] leftProfileIndex;
        private final double[] rightProfile;
        private final int[] rightProfileIndex;
        private final int subsequenceLength;
        private final double[] timeSeries;
        
        public MatrixProfileResult(double[] mp, int[] pi) {
            this(mp, pi, null, null, null, null, 0, null);
        }
        
        public MatrixProfileResult(double[] mp, int[] pi,
                                   double[] leftProfile, int[] leftProfileIndex,
                                   double[] rightProfile, int[] rightProfileIndex,
                                   int subsequenceLength, double[] timeSeries) {
            this.matrixProfile = mp;
            this.profileIndex = pi;
            this.leftProfile = leftProfile;
            this.leftProfileIndex = leftProfileIndex;
            this.rightProfile = rightProfile;
            this.rightProfileIndex = rightProfileIndex;
            this.subsequenceLength = subsequenceLength;
            this.timeSeries = timeSeries;
        }
        
        public double[] getMatrixProfile() { return matrixProfile; }
        public int[] getProfileIndex() { return profileIndex; }
        public double[] getLeftProfile() { return leftProfile; }
        public int[] getLeftProfileIndex() { return leftProfileIndex; }
        public double[] getRightProfile() { return rightProfile; }
        public int[] getRightProfileIndex() { return rightProfileIndex; }
        public int getSubsequenceLength() { return subsequenceLength; }
        public double[] getTimeSeries() { return timeSeries; }
    }
    
    public static class MotifResult {
        private final List<MotifPair> motifs;
        private final double[] matrixProfile;
        
        public MotifResult() {
            this(new ArrayList<>(), new double[0]);
        }
        
        public MotifResult(List<MotifPair> motifs, double[] matrixProfile) {
            this.motifs = motifs;
            this.matrixProfile = matrixProfile;
        }
        
        public List<MotifPair> getMotifs() { return motifs; }
        public double[] getMatrixProfile() { return matrixProfile; }
        
        public static class MotifPair {
            private final int index1;
            private final int index2;
            private final double distance;
            
            public MotifPair() { this(0, 0, 0.0); }
            
            public MotifPair(int index1, int index2, double distance) {
                this.index1 = index1;
                this.index2 = index2;
                this.distance = distance;
            }
            
            public int getIndex1() { return index1; }
            public int getIndex2() { return index2; }
            public double getDistance() { return distance; }
        }
    }
    
    public static class DiscordResult {
        private final List<Discord> discords;
        private final double[] matrixProfile;
        
        public DiscordResult() {
            this(new ArrayList<>(), new double[0]);
        }
        
        public DiscordResult(List<Discord> discords, double[] matrixProfile) {
            this.discords = discords;
            this.matrixProfile = matrixProfile;
        }
        
        public List<Discord> getDiscords() { return discords; }
        public double[] getMatrixProfile() { return matrixProfile; }
        
        public static class Discord {
            private final int index;
            private final double distance;
            
            public Discord() { this(0, 0.0); }
            
            public Discord(int index, double distance) {
                this.index = index;
                this.distance = distance;
            }
            
            public int getIndex() { return index; }
            public double getDistance() { return distance; }
        }
    }
}
