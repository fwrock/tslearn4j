package org.tslearn.metrics.advanced;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Longest Common Subsequence (LCSS) distance for time series.
 * 
 * LCSS é uma métrica robusta que mede a similaridade entre séries temporais
 * baseada na subsequência comum mais longa, permitindo algumas variações
 * nos valores e deslocamentos temporais.
 * 
 * A métrica é definida como:
 * LCSS(s1, s2) = 1 - (|LCS(s1, s2)| / min(|s1|, |s2|))
 * 
 * Onde LCS é a subsequência comum mais longa encontrada considerando:
 * - Um threshold de valor (epsilon) para considerar pontos como "iguais"
 * - Uma janela temporal (delta) para permitir deslocamentos temporais
 * 
 * Características:
 * - Robusta a outliers
 * - Permite variações temporais e de amplitude
 * - Adequada para séries com ruído
 * - Invariante a transformações monótonas
 * 
 * Referência:
 * Vlachos, M., Kollios, G., & Gunopulos, D. (2002). 
 * "Discovering similar multidimensional trajectories"
 * 
 * @author TSLearn4J
 */
public class LCSS {
    
    private static final Logger logger = LoggerFactory.getLogger(LCSS.class);
    
    private final double epsilon;
    private final int delta;
    private final boolean verbose;
    
    /**
     * Construtor com parâmetros padrão.
     * epsilon = 0.1, delta = 1
     */
    public LCSS() {
        this(0.1, 1, false);
    }
    
    /**
     * Construtor com epsilon personalizado.
     * 
     * @param epsilon Threshold para considerar valores como similares
     */
    public LCSS(double epsilon) {
        this(epsilon, 1, false);
    }
    
    /**
     * Construtor completo.
     * 
     * @param epsilon Threshold para considerar valores como similares (geralmente 0.1 * std)
     * @param delta Janela temporal máxima para alinhamento (geralmente length/10)
     * @param verbose Se deve imprimir informações de debug
     */
    public LCSS(double epsilon, int delta, boolean verbose) {
        if (epsilon < 0) {
            throw new IllegalArgumentException("Epsilon deve ser não-negativo");
        }
        if (delta < 0) {
            throw new IllegalArgumentException("Delta deve ser não-negativo");
        }
        
        this.epsilon = epsilon;
        this.delta = delta;
        this.verbose = verbose;
        
        if (verbose) {
            logger.info("LCSS configurada: epsilon={}, delta={}", epsilon, delta);
        }
    }
    
    /**
     * Calcula a distância LCSS entre duas séries temporais univariadas.
     * 
     * @param ts1 Primeira série temporal
     * @param ts2 Segunda série temporal
     * @return Distância LCSS normalizada [0, 1]
     */
    public double distance(double[] ts1, double[] ts2) {
        if (ts1 == null || ts2 == null) {
            throw new IllegalArgumentException("Séries temporais não podem ser null");
        }
        
        if (ts1.length == 0 || ts2.length == 0) {
            return 1.0; // Máxima dissimilaridade
        }
        
        int lcsLength = calculateLCS(ts1, ts2);
        int minLength = Math.min(ts1.length, ts2.length);
        
        double similarity = (double) lcsLength / minLength;
        double distance = 1.0 - similarity;
        
        if (verbose) {
            logger.debug("LCSS: LCS={}, min_length={}, similarity={:.4f}, distance={:.4f}", 
                        lcsLength, minLength, similarity, distance);
        }
        
        return distance;
    }
    
    /**
     * Calcula a distância LCSS entre duas séries temporais multivariadas.
     * 
     * @param ts1 Primeira série temporal [time][features]
     * @param ts2 Segunda série temporal [time][features]
     * @return Distância LCSS normalizada [0, 1]
     */
    public double distance(double[][] ts1, double[][] ts2) {
        if (ts1 == null || ts2 == null) {
            throw new IllegalArgumentException("Séries temporais não podem ser null");
        }
        
        if (ts1.length == 0 || ts2.length == 0) {
            return 1.0;
        }
        
        if (ts1[0].length != ts2[0].length) {
            throw new IllegalArgumentException("Séries devem ter o mesmo número de features");
        }
        
        int lcsLength = calculateMultivariateLCS(ts1, ts2);
        int minLength = Math.min(ts1.length, ts2.length);
        
        double similarity = (double) lcsLength / minLength;
        double distance = 1.0 - similarity;
        
        if (verbose) {
            logger.debug("LCSS Multivariada: LCS={}, min_length={}, distance={:.4f}", 
                        lcsLength, minLength, distance);
        }
        
        return distance;
    }
    
    /**
     * Calcula o comprimento da subsequência comum mais longa (LCS) univariada.
     */
    private int calculateLCS(double[] ts1, double[] ts2) {
        int m = ts1.length;
        int n = ts2.length;
        
        // Matriz DP para LCS
        int[][] dp = new int[m + 1][n + 1];
        
        // Preenchimento da matriz DP
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                // Verificar se os pontos são similares considerando delta temporal
                if (isWithinTemporalWindow(i - 1, j - 1) && 
                    isSimilar(ts1[i - 1], ts2[j - 1])) {
                    
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        return dp[m][n];
    }
    
    /**
     * Calcula o comprimento da LCS para séries multivariadas.
     */
    private int calculateMultivariateLCS(double[][] ts1, double[][] ts2) {
        int m = ts1.length;
        int n = ts2.length;
        
        int[][] dp = new int[m + 1][n + 1];
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (isWithinTemporalWindow(i - 1, j - 1) && 
                    isMultivariateSimilar(ts1[i - 1], ts2[j - 1])) {
                    
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        return dp[m][n];
    }
    
    /**
     * Verifica se dois índices estão dentro da janela temporal permitida.
     */
    private boolean isWithinTemporalWindow(int i, int j) {
        return Math.abs(i - j) <= delta;
    }
    
    /**
     * Verifica se dois valores são similares (dentro do threshold epsilon).
     */
    private boolean isSimilar(double v1, double v2) {
        return Math.abs(v1 - v2) <= epsilon;
    }
    
    /**
     * Verifica se dois vetores multivariados são similares.
     * Todos os features devem estar dentro do threshold.
     */
    private boolean isMultivariateSimilar(double[] vec1, double[] vec2) {
        if (vec1.length != vec2.length) {
            return false;
        }
        
        for (int k = 0; k < vec1.length; k++) {
            if (!isSimilar(vec1[k], vec2[k])) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Retorna a distância LCSS com informações adicionais.
     */
    public LCSSResult distanceWithDetails(double[] ts1, double[] ts2) {
        if (ts1 == null || ts2 == null) {
            throw new IllegalArgumentException("Séries temporais não podem ser null");
        }
        
        if (ts1.length == 0 || ts2.length == 0) {
            return new LCSSResult(1.0, 0, Math.min(ts1.length, ts2.length), 0.0);
        }
        
        int lcsLength = calculateLCS(ts1, ts2);
        int minLength = Math.min(ts1.length, ts2.length);
        double similarity = (double) lcsLength / minLength;
        double distance = 1.0 - similarity;
        
        return new LCSSResult(distance, lcsLength, minLength, similarity);
    }
    
    /**
     * Calcula epsilon automaticamente baseado no desvio padrão das séries.
     * 
     * @param ts1 Primeira série temporal
     * @param ts2 Segunda série temporal
     * @param factor Fator multiplicador do desvio padrão (padrão: 0.1)
     * @return Epsilon calculado
     */
    public static double calculateAutoEpsilon(double[] ts1, double[] ts2, double factor) {
        double std1 = calculateStandardDeviation(ts1);
        double std2 = calculateStandardDeviation(ts2);
        double avgStd = (std1 + std2) / 2.0;
        return avgStd * factor;
    }
    
    /**
     * Calcula epsilon automaticamente com fator padrão 0.1.
     */
    public static double calculateAutoEpsilon(double[] ts1, double[] ts2) {
        return calculateAutoEpsilon(ts1, ts2, 0.1);
    }
    
    /**
     * Calcula delta automaticamente baseado no comprimento das séries.
     * 
     * @param ts1 Primeira série temporal
     * @param ts2 Segunda série temporal
     * @param factor Fator do comprimento mínimo (padrão: 0.1)
     * @return Delta calculado
     */
    public static int calculateAutoDelta(double[] ts1, double[] ts2, double factor) {
        int minLength = Math.min(ts1.length, ts2.length);
        return Math.max(1, (int) (minLength * factor));
    }
    
    /**
     * Calcula delta automaticamente com fator padrão 0.1.
     */
    public static int calculateAutoDelta(double[] ts1, double[] ts2) {
        return calculateAutoDelta(ts1, ts2, 0.1);
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
        private double epsilon = 0.1;
        private int delta = 1;
        private boolean verbose = false;
        
        public Builder epsilon(double epsilon) {
            this.epsilon = epsilon;
            return this;
        }
        
        public Builder delta(int delta) {
            this.delta = delta;
            return this;
        }
        
        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }
        
        public Builder autoEpsilon(double[] ts1, double[] ts2, double factor) {
            this.epsilon = calculateAutoEpsilon(ts1, ts2, factor);
            return this;
        }
        
        public Builder autoEpsilon(double[] ts1, double[] ts2) {
            this.epsilon = calculateAutoEpsilon(ts1, ts2);
            return this;
        }
        
        public Builder autoDelta(double[] ts1, double[] ts2, double factor) {
            this.delta = calculateAutoDelta(ts1, ts2, factor);
            return this;
        }
        
        public Builder autoDelta(double[] ts1, double[] ts2) {
            this.delta = calculateAutoDelta(ts1, ts2);
            return this;
        }
        
        public LCSS build() {
            return new LCSS(epsilon, delta, verbose);
        }
    }
    
    /**
     * Resultado detalhado da distância LCSS.
     */
    public static class LCSSResult {
        private final double distance;
        private final int lcsLength;
        private final int minLength;
        private final double similarity;
        
        public LCSSResult(double distance, int lcsLength, int minLength, double similarity) {
            this.distance = distance;
            this.lcsLength = lcsLength;
            this.minLength = minLength;
            this.similarity = similarity;
        }
        
        public double getDistance() { return distance; }
        public int getLcsLength() { return lcsLength; }
        public int getMinLength() { return minLength; }
        public double getSimilarity() { return similarity; }
        
        @Override
        public String toString() {
            return String.format("LCSSResult{distance=%.4f, lcsLength=%d, minLength=%d, similarity=%.4f}", 
                               distance, lcsLength, minLength, similarity);
        }
    }
    
    // Getters
    public double getEpsilon() { return epsilon; }
    public int getDelta() { return delta; }
    public boolean isVerbose() { return verbose; }
}
