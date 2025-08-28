package org.tslearn.metrics;

import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Dynamic Time Warping (DTW) distance implementation with various optimizations.
 * 
 * This implementation provides:
 * - Standard DTW distance calculation
 * - Sakoe-Chiba band constraint for efficiency
 * - Itakura parallelogram constraint
 * - Early termination optimization
 * - Memory-optimized computation using only two rows
 * 
 * Based on the Python tslearn implementation but optimized for Java performance.
 */
public class DTW {
    
    private static final Logger logger = LoggerFactory.getLogger(DTW.class);
    
    // Global constraint types
    public enum GlobalConstraint {
        NONE,           // No constraint
        SAKOE_CHIBA,    // Sakoe-Chiba band
        ITAKURA         // Itakura parallelogram
    }
    
    private final GlobalConstraint globalConstraint;
    private final double globalConstraintParam;
    private final boolean enableEarlyTermination;
    private final double earlyTerminationThreshold;
    
    /**
     * Constructor with default parameters (no constraints)
     */
    public DTW() {
        this(GlobalConstraint.NONE, 0.0, false, Double.POSITIVE_INFINITY);
    }
    
    /**
     * Constructor with Sakoe-Chiba band constraint
     * 
     * @param sakoeChiba Sakoe-Chiba band width (number of diagonals)
     */
    public DTW(int sakoeChiba) {
        this(GlobalConstraint.SAKOE_CHIBA, sakoeChiba, false, Double.POSITIVE_INFINITY);
    }
    
    /**
     * Full constructor with all optimization options
     * 
     * @param globalConstraint Type of global constraint
     * @param globalConstraintParam Parameter for the constraint (band width, etc.)
     * @param enableEarlyTermination Whether to use early termination
     * @param earlyTerminationThreshold Threshold for early termination
     */
    public DTW(GlobalConstraint globalConstraint, 
               double globalConstraintParam,
               boolean enableEarlyTermination,
               double earlyTerminationThreshold) {
        this.globalConstraint = globalConstraint;
        this.globalConstraintParam = globalConstraintParam;
        this.enableEarlyTermination = enableEarlyTermination;
        this.earlyTerminationThreshold = earlyTerminationThreshold;
    }
    
    /**
     * Calculate DTW distance between two time series
     * 
     * @param ts1 First time series
     * @param ts2 Second time series
     * @return DTW distance
     */
    public double distance(double[] ts1, double[] ts2) {
        if (ts1 == null || ts2 == null) {
            throw new IllegalArgumentException("Time series cannot be null");
        }
        
        if (ts1.length == 0 || ts2.length == 0) {
            throw new IllegalArgumentException("Time series cannot be empty");
        }
        
        int n = ts1.length;
        int m = ts2.length;
        
        // Choose algorithm based on constraint type and series length
        switch (globalConstraint) {
            case SAKOE_CHIBA:
                return dtwSakoeChiba(ts1, ts2, (int) globalConstraintParam);
            case ITAKURA:
                return dtwItakura(ts1, ts2);
            default:
                return dtwStandard(ts1, ts2);
        }
    }
    
    /**
     * Calculate DTW distance between two time series (matrix format)
     * 
     * @param ts1 First time series as 1x n matrix
     * @param ts2 Second time series as 1x n matrix  
     * @return DTW distance
     */
    public double distance(RealMatrix ts1, RealMatrix ts2) {
        return distance(ts1.getData()[0], ts2.getData()[0]);
    }
    
    /**
     * Standard DTW implementation without constraints
     * Uses memory optimization (only 2 rows instead of full matrix)
     */
    private double dtwStandard(double[] ts1, double[] ts2) {
        int n = ts1.length;
        int m = ts2.length;
        
        // Use only two rows for memory efficiency
        double[] prevRow = new double[m + 1];
        double[] currRow = new double[m + 1];
        
        // Initialize first row
        prevRow[0] = 0.0;
        for (int j = 1; j <= m; j++) {
            prevRow[j] = Double.POSITIVE_INFINITY;
        }
        
        // Fill DTW matrix row by row
        for (int i = 1; i <= n; i++) {
            currRow[0] = Double.POSITIVE_INFINITY;
            
            double minInRow = Double.POSITIVE_INFINITY;
            
            for (int j = 1; j <= m; j++) {
                double cost = euclideanDistance(ts1[i-1], ts2[j-1]);
                
                double dtw = Math.min(Math.min(
                    prevRow[j],     // insertion
                    currRow[j-1]),  // deletion
                    prevRow[j-1]    // match
                ) + cost;
                
                currRow[j] = dtw;
                minInRow = Math.min(minInRow, dtw);
            }
            
            // Early termination check
            if (enableEarlyTermination && minInRow > earlyTerminationThreshold) {
                logger.debug("Early termination at row {}", i);
                return Double.POSITIVE_INFINITY;
            }
            
            // Swap rows
            double[] temp = prevRow;
            prevRow = currRow;
            currRow = temp;
        }
        
        return prevRow[m];
    }
    
    /**
     * DTW with Sakoe-Chiba band constraint for efficiency
     * Limits the warping path to a band around the diagonal
     */
    private double dtwSakoeChiba(double[] ts1, double[] ts2, int bandWidth) {
        int n = ts1.length;
        int m = ts2.length;
        
        if (bandWidth < 0) {
            throw new IllegalArgumentException("Band width must be non-negative");
        }
        
        // Use only two rows for memory efficiency
        double[] prevRow = new double[m + 1];
        double[] currRow = new double[m + 1];
        
        // Initialize
        for (int j = 0; j <= m; j++) {
            prevRow[j] = Double.POSITIVE_INFINITY;
            currRow[j] = Double.POSITIVE_INFINITY;
        }
        prevRow[0] = 0.0;
        
        for (int i = 1; i <= n; i++) {
            currRow[0] = Double.POSITIVE_INFINITY;
            
            // Calculate band boundaries for this row
            int jStart = Math.max(1, i - bandWidth);
            int jEnd = Math.min(m, i + bandWidth);
            
            double minInRow = Double.POSITIVE_INFINITY;
            
            for (int j = jStart; j <= jEnd; j++) {
                double cost = euclideanDistance(ts1[i-1], ts2[j-1]);
                
                double dtw = Math.min(Math.min(
                    prevRow[j],     // insertion
                    currRow[j-1]),  // deletion
                    prevRow[j-1]    // match
                ) + cost;
                
                currRow[j] = dtw;
                minInRow = Math.min(minInRow, dtw);
            }
            
            // Early termination check
            if (enableEarlyTermination && minInRow > earlyTerminationThreshold) {
                logger.debug("Early termination at row {} with Sakoe-Chiba band {}", i, bandWidth);
                return Double.POSITIVE_INFINITY;
            }
            
            // Swap rows
            double[] temp = prevRow;
            prevRow = currRow;
            currRow = temp;
        }
        
        return prevRow[m];
    }
    
    /**
     * DTW with Itakura parallelogram constraint
     * More restrictive than Sakoe-Chiba, follows a parallelogram shape
     */
    private double dtwItakura(double[] ts1, double[] ts2) {
        int n = ts1.length;
        int m = ts2.length;
        
        // Use only two rows for memory efficiency
        double[] prevRow = new double[m + 1];
        double[] currRow = new double[m + 1];
        
        // Initialize
        for (int j = 0; j <= m; j++) {
            prevRow[j] = Double.POSITIVE_INFINITY;
            currRow[j] = Double.POSITIVE_INFINITY;
        }
        prevRow[0] = 0.0;
        
        for (int i = 1; i <= n; i++) {
            currRow[0] = Double.POSITIVE_INFINITY;
            
            // Itakura parallelogram boundaries
            double slope1 = 0.5; // Lower slope
            double slope2 = 2.0; // Upper slope
            
            int jStart = Math.max(1, (int) Math.floor(i * slope1));
            int jEnd = Math.min(m, (int) Math.ceil(i * slope2));
            
            double minInRow = Double.POSITIVE_INFINITY;
            
            for (int j = jStart; j <= jEnd; j++) {
                double cost = euclideanDistance(ts1[i-1], ts2[j-1]);
                
                double dtw = Math.min(Math.min(
                    prevRow[j],     // insertion
                    currRow[j-1]),  // deletion
                    prevRow[j-1]    // match
                ) + cost;
                
                currRow[j] = dtw;
                minInRow = Math.min(minInRow, dtw);
            }
            
            // Early termination check
            if (enableEarlyTermination && minInRow > earlyTerminationThreshold) {
                logger.debug("Early termination at row {} with Itakura constraint", i);
                return Double.POSITIVE_INFINITY;
            }
            
            // Swap rows
            double[] temp = prevRow;
            prevRow = currRow;
            currRow = temp;
        }
        
        return prevRow[m];
    }
    
    /**
     * Calculate DTW path (alignment) between two time series
     * Returns the optimal warping path as pairs of indices
     */
    public DTWResult distanceWithPath(double[] ts1, double[] ts2) {
        int n = ts1.length;
        int m = ts2.length;
        
        // Full DTW matrix for path backtracking
        double[][] dtwMatrix = new double[n + 1][m + 1];
        
        // Initialize matrix
        for (int i = 0; i <= n; i++) {
            dtwMatrix[i][0] = Double.POSITIVE_INFINITY;
        }
        for (int j = 0; j <= m; j++) {
            dtwMatrix[0][j] = Double.POSITIVE_INFINITY;
        }
        dtwMatrix[0][0] = 0.0;
        
        // Fill DTW matrix
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (isValidCell(i, j, n, m)) {
                    double cost = euclideanDistance(ts1[i-1], ts2[j-1]);
                    
                    dtwMatrix[i][j] = Math.min(Math.min(
                        dtwMatrix[i-1][j],     // insertion
                        dtwMatrix[i][j-1]),    // deletion
                        dtwMatrix[i-1][j-1]    // match
                    ) + cost;
                } else {
                    dtwMatrix[i][j] = Double.POSITIVE_INFINITY;
                }
            }
        }
        
        // Backtrack to find optimal path
        int[][] path = backtrackPath(dtwMatrix, n, m);
        
        return new DTWResult(dtwMatrix[n][m], path);
    }
    
    /**
     * Check if a cell (i,j) is valid under the current global constraint
     */
    private boolean isValidCell(int i, int j, int n, int m) {
        switch (globalConstraint) {
            case SAKOE_CHIBA:
                return Math.abs(i - j) <= globalConstraintParam;
            case ITAKURA:
                double ratio = (double) j / i;
                return ratio >= 0.5 && ratio <= 2.0;
            default:
                return true;
        }
    }
    
    /**
     * Backtrack through DTW matrix to find optimal alignment path
     */
    private int[][] backtrackPath(double[][] dtwMatrix, int n, int m) {
        java.util.List<int[]> pathList = new java.util.ArrayList<>();
        
        int i = n, j = m;
        while (i > 0 && j > 0) {
            pathList.add(new int[]{i-1, j-1}); // Convert to 0-based indexing
            
            // Find which direction gave minimum cost
            double match = dtwMatrix[i-1][j-1];
            double deletion = dtwMatrix[i][j-1];
            double insertion = dtwMatrix[i-1][j];
            
            if (match <= deletion && match <= insertion) {
                i--; j--;
            } else if (deletion <= insertion) {
                j--;
            } else {
                i--;
            }
        }
        
        // Handle remaining path to (0,0)
        while (i > 0) {
            pathList.add(new int[]{i-1, j-1});
            i--;
        }
        while (j > 0) {
            pathList.add(new int[]{i-1, j-1});
            j--;
        }
        
        // Reverse path (was built backwards)
        java.util.Collections.reverse(pathList);
        
        return pathList.toArray(new int[pathList.size()][]);
    }
    
    /**
     * Euclidean distance between two points
     */
    private double euclideanDistance(double x1, double x2) {
        double diff = x1 - x2;
        return diff * diff; // Squared distance (standard for DTW)
    }
    
    /**
     * Get the global constraint type being used
     */
    public GlobalConstraint getGlobalConstraint() {
        return globalConstraint;
    }
    
    /**
     * Get the global constraint parameter
     */
    public double getGlobalConstraintParam() {
        return globalConstraintParam;
    }
    
    /**
     * Check if early termination is enabled
     */
    public boolean isEarlyTerminationEnabled() {
        return enableEarlyTermination;
    }
    
    /**
     * Get the early termination threshold
     */
    public double getEarlyTerminationThreshold() {
        return earlyTerminationThreshold;
    }
    
    /**
     * Result class that contains both distance and alignment path
     */
    public static class DTWResult {
        private final double distance;
        private final int[][] path;
        
        public DTWResult(double distance, int[][] path) {
            this.distance = distance;
            this.path = path;
        }
        
        public double getDistance() {
            return distance;
        }
        
        public int[][] getPath() {
            return path;
        }
        
        public int getPathLength() {
            return path.length;
        }
        
        @Override
        public String toString() {
            return String.format("DTWResult{distance=%.6f, pathLength=%d}", distance, path.length);
        }
    }
    
    /**
     * Calcula a distância DTW entre duas séries temporais multivariadas.
     * 
     * @param ts1 Primeira série temporal [time_length][n_features]
     * @param ts2 Segunda série temporal [time_length][n_features]
     * @return Distância DTW
     */
    public double distance(double[][] ts1, double[][] ts2) {
        if (ts1 == null || ts2 == null) {
            throw new IllegalArgumentException("Séries temporais não podem ser nulas");
        }
        
        if (ts1[0].length != ts2[0].length) {
            throw new IllegalArgumentException("Séries devem ter o mesmo número de features");
        }
        
        int n = ts1.length;
        int m = ts2.length;
        int nFeatures = ts1[0].length;
        
        // Matriz de custos DTW
        double[][] cost = new double[n + 1][m + 1];
        
        // Inicialização
        for (int i = 0; i <= n; i++) {
            cost[i][0] = Double.POSITIVE_INFINITY;
        }
        for (int j = 0; j <= m; j++) {
            cost[0][j] = Double.POSITIVE_INFINITY;
        }
        cost[0][0] = 0.0;
        
        // Preenchimento da matriz
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (isValidCell(i - 1, j - 1, n, m)) {
                    double euclideanDist = euclideanDistance(ts1[i - 1], ts2[j - 1]);
                    cost[i][j] = euclideanDist + Math.min(Math.min(
                        cost[i - 1][j],     // Inserção
                        cost[i][j - 1]),    // Deleção
                        cost[i - 1][j - 1]  // Match
                    );
                } else {
                    cost[i][j] = Double.POSITIVE_INFINITY;
                }
            }
        }
        
        return cost[n][m];
    }
    
    /**
     * Calcula a distância DTW com o caminho de alinhamento para séries multivariadas.
     */
    public DTWPathResult distanceWithPath(double[][] ts1, double[][] ts2) {
        if (ts1 == null || ts2 == null) {
            throw new IllegalArgumentException("Séries temporais não podem ser nulas");
        }
        
        int n = ts1.length;
        int m = ts2.length;
        
        // Matriz de custos DTW
        double[][] cost = new double[n + 1][m + 1];
        
        // Inicialização
        for (int i = 0; i <= n; i++) {
            cost[i][0] = Double.POSITIVE_INFINITY;
        }
        for (int j = 0; j <= m; j++) {
            cost[0][j] = Double.POSITIVE_INFINITY;
        }
        cost[0][0] = 0.0;
        
        // Preenchimento da matriz
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (isValidCell(i - 1, j - 1, n, m)) {
                    double euclideanDist = euclideanDistance(ts1[i - 1], ts2[j - 1]);
                    cost[i][j] = euclideanDist + Math.min(Math.min(
                        cost[i - 1][j],     // Inserção
                        cost[i][j - 1]),    // Deleção
                        cost[i - 1][j - 1]  // Match
                    );
                } else {
                    cost[i][j] = Double.POSITIVE_INFINITY;
                }
            }
        }
        
        // Reconstruir caminho
        java.util.List<int[]> path = new java.util.ArrayList<>();
        int i = n, j = m;
        
        while (i > 0 && j > 0) {
            path.add(new int[]{i - 1, j - 1});
            
            double diag = cost[i - 1][j - 1];
            double left = cost[i][j - 1];
            double up = cost[i - 1][j];
            
            if (diag <= left && diag <= up) {
                i--; j--;
            } else if (left <= up) {
                j--;
            } else {
                i--;
            }
        }
        
        // Reverter o caminho
        java.util.Collections.reverse(path);
        
        return new DTWPathResult(cost[n][m], path);
    }
    
    /**
     * Calcula a distância euclidiana entre dois pontos multivariados.
     */
    private double euclideanDistance(double[] point1, double[] point2) {
        double sum = 0.0;
        for (int d = 0; d < point1.length; d++) {
            double diff = point1[d] - point2[d];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Resultado DTW com caminho para séries multivariadas.
     */
    public static class DTWPathResult {
        private final double distance;
        private final java.util.List<int[]> path;
        
        public DTWPathResult(double distance, java.util.List<int[]> path) {
            this.distance = distance;
            this.path = path;
        }
        
        public double getDistance() {
            return distance;
        }
        
        public java.util.List<int[]> getPath() {
            return path;
        }
        
        public String toString() {
            return String.format("DTWPathResult{distance=%.6f, pathLength=%d}", distance, path.size());
        }
    }
    
    /**
     * Builder pattern para construção da DTW.
     */
    public static class Builder {
        private GlobalConstraint constraint = GlobalConstraint.NONE;
        private double constraintParam = 0.0;
        private boolean earlyTermination = false;
        private double earlyThreshold = Double.POSITIVE_INFINITY;
        
        public Builder constraintType(GlobalConstraint constraint) {
            this.constraint = constraint;
            return this;
        }
        
        public Builder sakoeChibaRadius(int radius) {
            this.constraint = GlobalConstraint.SAKOE_CHIBA;
            this.constraintParam = radius;
            return this;
        }
        
        public Builder itakuraParallelogram() {
            this.constraint = GlobalConstraint.ITAKURA;
            return this;
        }
        
        public Builder enableEarlyTermination(double threshold) {
            this.earlyTermination = true;
            this.earlyThreshold = threshold;
            return this;
        }
        
        public DTW build() {
            return new DTW(constraint, constraintParam, earlyTermination, earlyThreshold);
        }
    }
}
