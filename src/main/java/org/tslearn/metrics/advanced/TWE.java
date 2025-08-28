package org.tslearn.metrics.advanced;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Time Warp Edit (TWE) distance for time series.
 * 
 * TWE combina as vantagens do Dynamic Time Warping (DTW) com operações
 * de edição (inserção/deleção), permitindo tanto alinhamento temporal
 * quanto modificações estruturais na série temporal.
 * 
 * A métrica considera:
 * - Operações de edição: inserção e deleção de pontos
 * - Warping temporal: alinhamento flexível como DTW
 * - Penalty por stiffness: controla quanto warping é permitido
 * 
 * Características:
 * - Combina DTW com edit distance
 * - Controle de rigidez temporal via parâmetro λ (lambda)
 * - Adequada para séries com diferentes estruturas temporais
 * - Robusta a outliers e variações locais
 * 
 * Parâmetros:
 * - nu (ν): penalty para operações de edição
 * - lambda (λ): penalty para operações de warping (stiffness)
 * 
 * Referência:
 * Marteau, P. F. (2009).
 * "Time warp edit distance with stiffness adjustment for time series matching"
 * 
 * @author TSLearn4J
 */
public class TWE {
    
    private static final Logger logger = LoggerFactory.getLogger(TWE.class);
    
    private final double nu;        // Penalty para edit operations
    private final double lambda;    // Penalty para warping (stiffness)
    private final boolean verbose;
    
    /**
     * Construtor com parâmetros padrão.
     * nu = 0.001, lambda = 1.0
     */
    public TWE() {
        this(0.001, 1.0, false);
    }
    
    /**
     * Construtor com nu personalizado.
     * 
     * @param nu Penalty para operações de edição
     */
    public TWE(double nu) {
        this(nu, 1.0, false);
    }
    
    /**
     * Construtor completo.
     * 
     * @param nu Penalty para operações de edição (geralmente 0.001)
     * @param lambda Penalty para warping/stiffness (geralmente 1.0)
     * @param verbose Se deve imprimir informações de debug
     */
    public TWE(double nu, double lambda, boolean verbose) {
        if (nu < 0 || lambda < 0) {
            throw new IllegalArgumentException("Nu e lambda devem ser não-negativos");
        }
        
        this.nu = nu;
        this.lambda = lambda;
        this.verbose = verbose;
        
        if (verbose) {
            logger.info("TWE configurada: nu={}, lambda={}", nu, lambda);
        }
    }
    
    /**
     * Calcula a distância TWE entre duas séries temporais univariadas.
     * 
     * @param ts1 Primeira série temporal
     * @param ts2 Segunda série temporal
     * @return Distância TWE
     */
    public double distance(double[] ts1, double[] ts2) {
        if (ts1 == null || ts2 == null) {
            throw new IllegalArgumentException("Séries temporais não podem ser null");
        }
        
        if (ts1.length == 0 && ts2.length == 0) {
            return 0.0;
        }
        
        if (ts1.length == 0) {
            return calculateInsertionCost(ts2);
        }
        
        if (ts2.length == 0) {
            return calculateDeletionCost(ts1);
        }
        
        double distance = calculateTWE(ts1, ts2);
        
        if (verbose) {
            logger.debug("TWE distance calculada: {:.6f} entre séries de tamanhos {} e {}", 
                        distance, ts1.length, ts2.length);
        }
        
        return distance;
    }
    
    /**
     * Calcula a distância TWE entre duas séries temporais multivariadas.
     * 
     * @param ts1 Primeira série temporal [time][features]
     * @param ts2 Segunda série temporal [time][features]
     * @return Distância TWE
     */
    public double distance(double[][] ts1, double[][] ts2) {
        if (ts1 == null || ts2 == null) {
            throw new IllegalArgumentException("Séries temporais não podem ser null");
        }
        
        if (ts1.length == 0 && ts2.length == 0) {
            return 0.0;
        }
        
        if (ts1.length == 0 || ts2.length == 0) {
            double[][] nonEmpty = ts1.length > 0 ? ts1 : ts2;
            return calculateMultivariateInsertionDeletionCost(nonEmpty);
        }
        
        if (ts1[0].length != ts2[0].length) {
            throw new IllegalArgumentException("Séries devem ter o mesmo número de features");
        }
        
        double distance = calculateMultivariateTWE(ts1, ts2);
        
        if (verbose) {
            logger.debug("TWE multivariada: distância={:.6f} para séries {}x{} e {}x{}", 
                        distance, ts1.length, ts1[0].length, ts2.length, ts2[0].length);
        }
        
        return distance;
    }
    
    /**
     * Implementação do algoritmo TWE usando programação dinâmica.
     */
    private double calculateTWE(double[] ts1, double[] ts2) {
        int m = ts1.length;
        int n = ts2.length;
        
        // Matriz de programação dinâmica
        double[][] dp = new double[m + 1][n + 1];
        
        // Inicialização
        dp[0][0] = 0.0;
        
        // Primeira linha: apenas inserções da série 2
        for (int j = 1; j <= n; j++) {
            dp[0][j] = dp[0][j - 1] + calculateInsertionPenalty(ts2[j - 1], j - 1);
        }
        
        // Primeira coluna: apenas deleções da série 1
        for (int i = 1; i <= m; i++) {
            dp[i][0] = dp[i - 1][0] + calculateDeletionPenalty(ts1[i - 1], i - 1);
        }
        
        // Preenchimento da matriz principal
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                // Custo do matching (com warping penalty)
                double matchCost = calculateMatchCost(ts1[i - 1], ts2[j - 1], i - 1, j - 1);
                double matchOption = dp[i - 1][j - 1] + matchCost;
                
                // Custo da inserção
                double insertCost = calculateInsertionPenalty(ts2[j - 1], j - 1);
                double insertOption = dp[i][j - 1] + insertCost;
                
                // Custo da deleção
                double deleteCost = calculateDeletionPenalty(ts1[i - 1], i - 1);
                double deleteOption = dp[i - 1][j] + deleteCost;
                
                // Escolher a opção de menor custo
                dp[i][j] = Math.min(Math.min(matchOption, insertOption), deleteOption);
            }
        }
        
        return dp[m][n];
    }
    
    /**
     * Implementação TWE para séries multivariadas.
     */
    private double calculateMultivariateTWE(double[][] ts1, double[][] ts2) {
        int m = ts1.length;
        int n = ts2.length;
        int d = ts1[0].length; // número de features
        
        double[][] dp = new double[m + 1][n + 1];
        
        // Inicialização
        dp[0][0] = 0.0;
        
        // Primeira linha e coluna
        for (int j = 1; j <= n; j++) {
            dp[0][j] = dp[0][j - 1] + calculateMultivariateInsertionPenalty(ts2[j - 1], j - 1);
        }
        
        for (int i = 1; i <= m; i++) {
            dp[i][0] = dp[i - 1][0] + calculateMultivariateDeletionPenalty(ts1[i - 1], i - 1);
        }
        
        // Preenchimento principal
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                // Match cost para todos os features
                double matchCost = calculateMultivariateMatchCost(ts1[i - 1], ts2[j - 1], i - 1, j - 1);
                double matchOption = dp[i - 1][j - 1] + matchCost;
                
                // Insert/delete costs
                double insertCost = calculateMultivariateInsertionPenalty(ts2[j - 1], j - 1);
                double insertOption = dp[i][j - 1] + insertCost;
                
                double deleteCost = calculateMultivariateDeletionPenalty(ts1[i - 1], i - 1);
                double deleteOption = dp[i - 1][j] + deleteCost;
                
                dp[i][j] = Math.min(Math.min(matchOption, insertOption), deleteOption);
            }
        }
        
        return dp[m][n];
    }
    
    /**
     * Calcula o custo de matching entre dois pontos com penalty de warping.
     */
    private double calculateMatchCost(double v1, double v2, int i, int j) {
        double valueDiff = Math.abs(v1 - v2);
        double timeDiff = Math.abs(i - j);
        double warpPenalty = lambda * timeDiff;
        
        return valueDiff + warpPenalty;
    }
    
    /**
     * Calcula o custo de matching multivariado.
     */
    private double calculateMultivariateMatchCost(double[] vec1, double[] vec2, int i, int j) {
        double valueDiff = 0.0;
        for (int k = 0; k < vec1.length; k++) {
            valueDiff += Math.abs(vec1[k] - vec2[k]);
        }
        
        double timeDiff = Math.abs(i - j);
        double warpPenalty = lambda * timeDiff * vec1.length; // Scale by dimensionality
        
        return valueDiff + warpPenalty;
    }
    
    /**
     * Calcula o penalty para inserção de um ponto.
     */
    private double calculateInsertionPenalty(double value, int position) {
        return nu + Math.abs(value) * 0.1; // Small penalty proportional to magnitude
    }
    
    /**
     * Calcula o penalty para deleção de um ponto.
     */
    private double calculateDeletionPenalty(double value, int position) {
        return nu + Math.abs(value) * 0.1;
    }
    
    /**
     * Calcula o penalty multivariado para inserção.
     */
    private double calculateMultivariateInsertionPenalty(double[] vector, int position) {
        double magnitude = 0.0;
        for (double v : vector) {
            magnitude += Math.abs(v);
        }
        return nu * vector.length + magnitude * 0.1;
    }
    
    /**
     * Calcula o penalty multivariado para deleção.
     */
    private double calculateMultivariateDeletionPenalty(double[] vector, int position) {
        return calculateMultivariateInsertionPenalty(vector, position);
    }
    
    /**
     * Calcula o custo de inserção de uma série completa.
     */
    private double calculateInsertionCost(double[] ts) {
        double cost = 0.0;
        for (int i = 0; i < ts.length; i++) {
            cost += calculateInsertionPenalty(ts[i], i);
        }
        return cost;
    }
    
    /**
     * Calcula o custo de deleção de uma série completa.
     */
    private double calculateDeletionCost(double[] ts) {
        double cost = 0.0;
        for (int i = 0; i < ts.length; i++) {
            cost += calculateDeletionPenalty(ts[i], i);
        }
        return cost;
    }
    
    /**
     * Calcula o custo de inserção/deleção multivariado.
     */
    private double calculateMultivariateInsertionDeletionCost(double[][] ts) {
        double cost = 0.0;
        for (int i = 0; i < ts.length; i++) {
            cost += calculateMultivariateInsertionPenalty(ts[i], i);
        }
        return cost;
    }
    
    /**
     * Calcula parâmetros automaticamente baseados nas características das séries.
     */
    public static TWE createAutoConfigured(double[] ts1, double[] ts2) {
        double std1 = calculateStandardDeviation(ts1);
        double std2 = calculateStandardDeviation(ts2);
        double avgStd = (std1 + std2) / 2.0;
        
        // Nu proporcional à escala dos dados
        double autoNu = Math.max(0.001, avgStd * 0.01);
        
        // Lambda baseado na diferença de comprimentos
        double lengthRatio = (double) Math.abs(ts1.length - ts2.length) / 
                           Math.max(ts1.length, ts2.length);
        double autoLambda = Math.max(0.1, lengthRatio * 2.0);
        
        return new TWE(autoNu, autoLambda, false);
    }
    
    /**
     * Retorna resultado detalhado da distância TWE.
     */
    public TWEResult distanceWithDetails(double[] ts1, double[] ts2) {
        double distance = distance(ts1, ts2);
        
        // Estimativas aproximadas dos componentes
        double lengthDiff = Math.abs(ts1.length - ts2.length);
        double editComponent = lengthDiff * nu;
        double warpComponent = distance - editComponent;
        
        return new TWEResult(distance, editComponent, warpComponent, lengthDiff);
    }
    
    /**
     * Calcula o desvio padrão de uma série temporal.
     */
    private static double calculateStandardDeviation(double[] ts) {
        if (ts.length <= 1) return 0.0;
        
        double mean = 0.0;
        for (double value : ts) {
            mean += value;
        }
        mean /= ts.length;
        
        double variance = 0.0;
        for (double value : ts) {
            variance += Math.pow(value - mean, 2);
        }
        variance /= (ts.length - 1);
        
        return Math.sqrt(variance);
    }
    
    /**
     * Builder pattern para configuração flexível.
     */
    public static class Builder {
        private double nu = 0.001;
        private double lambda = 1.0;
        private boolean verbose = false;
        
        public Builder nu(double nu) {
            this.nu = nu;
            return this;
        }
        
        public Builder lambda(double lambda) {
            this.lambda = lambda;
            return this;
        }
        
        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }
        
        public Builder autoConfigured(double[] ts1, double[] ts2) {
            TWE auto = createAutoConfigured(ts1, ts2);
            this.nu = auto.nu;
            this.lambda = auto.lambda;
            return this;
        }
        
        public TWE build() {
            return new TWE(nu, lambda, verbose);
        }
    }
    
    /**
     * Resultado detalhado da distância TWE.
     */
    public static class TWEResult {
        private final double distance;
        private final double editComponent;
        private final double warpComponent;
        private final double lengthDifference;
        
        public TWEResult(double distance, double editComponent, double warpComponent, double lengthDifference) {
            this.distance = distance;
            this.editComponent = editComponent;
            this.warpComponent = warpComponent;
            this.lengthDifference = lengthDifference;
        }
        
        public double getDistance() { return distance; }
        public double getEditComponent() { return editComponent; }
        public double getWarpComponent() { return warpComponent; }
        public double getLengthDifference() { return lengthDifference; }
        
        public double getEditRatio() {
            return distance > 0 ? editComponent / distance : 0.0;
        }
        
        public double getWarpRatio() {
            return distance > 0 ? warpComponent / distance : 0.0;
        }
        
        @Override
        public String toString() {
            return String.format("TWEResult{distance=%.6f, edit=%.6f(%.1f%%), warp=%.6f(%.1f%%), length_diff=%.0f}", 
                               distance, editComponent, getEditRatio() * 100, 
                               warpComponent, getWarpRatio() * 100, lengthDifference);
        }
    }
    
    // Getters
    public double getNu() { return nu; }
    public double getLambda() { return lambda; }
    public boolean isVerbose() { return verbose; }
}
