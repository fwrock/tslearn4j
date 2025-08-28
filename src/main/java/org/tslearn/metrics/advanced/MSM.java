package org.tslearn.metrics.advanced;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Move-Split-Merge (MSM) distance for time series.
 * 
 * MSM é uma métrica robusta que considera três operações básicas para
 * transformar uma série temporal em outra:
 * - Move: Alterar o valor de um ponto
 * - Split: Dividir um ponto em dois
 * - Merge: Combinar dois pontos adjacentes
 * 
 * A métrica é especialmente adequada para séries temporais com:
 * - Diferentes resoluções temporais
 * - Variações no número de pontos significativos
 * - Outliers ocasionais
 * 
 * Características:
 * - Métrica de edição especializada para séries temporais
 * - Lida bem com compressão/expansão temporal
 * - Robusta a ruído e outliers
 * - Considera tanto magnitude quanto estrutura temporal
 * 
 * Referência:
 * Stefan, A., Athitsos, V., & Das, G. (2013).
 * "The Move-Split-Merge metric for time series"
 * 
 * @author TSLearn4J
 */
public class MSM {
    
    private static final Logger logger = LoggerFactory.getLogger(MSM.class);
    
    private final double moveCost;
    private final double splitMergeCost;
    private final boolean verbose;
    
    /**
     * Construtor com custos padrão.
     * moveCost = 1.0, splitMergeCost = 1.0
     */
    public MSM() {
        this(1.0, 1.0, false);
    }
    
    /**
     * Construtor com custo de movimento personalizado.
     * 
     * @param moveCost Custo da operação de movimento
     */
    public MSM(double moveCost) {
        this(moveCost, 1.0, false);
    }
    
    /**
     * Construtor completo.
     * 
     * @param moveCost Custo da operação de movimento (geralmente 1.0)
     * @param splitMergeCost Custo das operações de split/merge (geralmente 1.0)
     * @param verbose Se deve imprimir informações de debug
     */
    public MSM(double moveCost, double splitMergeCost, boolean verbose) {
        if (moveCost < 0 || splitMergeCost < 0) {
            throw new IllegalArgumentException("Custos devem ser não-negativos");
        }
        
        this.moveCost = moveCost;
        this.splitMergeCost = splitMergeCost;
        this.verbose = verbose;
        
        if (verbose) {
            logger.info("MSM configurada: moveCost={}, splitMergeCost={}", 
                       moveCost, splitMergeCost);
        }
    }
    
    /**
     * Calcula a distância MSM entre duas séries temporais univariadas.
     * 
     * @param ts1 Primeira série temporal
     * @param ts2 Segunda série temporal
     * @return Distância MSM
     */
    public double distance(double[] ts1, double[] ts2) {
        if (ts1 == null || ts2 == null) {
            throw new IllegalArgumentException("Séries temporais não podem ser null");
        }
        
        if (ts1.length == 0 && ts2.length == 0) {
            return 0.0;
        }
        
        if (ts1.length == 0) {
            return ts2.length * splitMergeCost;
        }
        
        if (ts2.length == 0) {
            return ts1.length * splitMergeCost;
        }
        
        double distance = calculateMSM(ts1, ts2);
        
        if (verbose) {
            logger.debug("MSM distance calculada: {:.4f} entre séries de tamanhos {} e {}", 
                        distance, ts1.length, ts2.length);
        }
        
        return distance;
    }
    
    /**
     * Calcula a distância MSM entre duas séries temporais multivariadas.
     * Aplica MSM em cada dimensão e soma os resultados.
     * 
     * @param ts1 Primeira série temporal [time][features]
     * @param ts2 Segunda série temporal [time][features]
     * @return Distância MSM
     */
    public double distance(double[][] ts1, double[][] ts2) {
        if (ts1 == null || ts2 == null) {
            throw new IllegalArgumentException("Séries temporais não podem ser null");
        }
        
        if (ts1.length == 0 && ts2.length == 0) {
            return 0.0;
        }
        
        if (ts1.length == 0 || ts2.length == 0) {
            int maxLength = Math.max(ts1.length, ts2.length);
            int features = ts1.length > 0 ? ts1[0].length : ts2[0].length;
            return maxLength * features * splitMergeCost;
        }
        
        if (ts1[0].length != ts2[0].length) {
            throw new IllegalArgumentException("Séries devem ter o mesmo número de features");
        }
        
        double totalDistance = 0.0;
        int numFeatures = ts1[0].length;
        
        // Aplicar MSM para cada feature
        for (int f = 0; f < numFeatures; f++) {
            double[] feature1 = extractFeature(ts1, f);
            double[] feature2 = extractFeature(ts2, f);
            totalDistance += distance(feature1, feature2);
        }
        
        if (verbose) {
            logger.debug("MSM multivariada: distância total={:.4f} para {} features", 
                        totalDistance, numFeatures);
        }
        
        return totalDistance;
    }
    
    /**
     * Implementação do algoritmo MSM usando programação dinâmica.
     */
    private double calculateMSM(double[] ts1, double[] ts2) {
        int m = ts1.length;
        int n = ts2.length;
        
        // Matriz de programação dinâmica
        double[][] dp = new double[m + 1][n + 1];
        
        // Inicialização das bordas
        dp[0][0] = 0.0;
        
        // Primeira linha: apenas splits da série 2
        for (int j = 1; j <= n; j++) {
            dp[0][j] = dp[0][j - 1] + splitMergeCost;
        }
        
        // Primeira coluna: apenas splits da série 1
        for (int i = 1; i <= m; i++) {
            dp[i][0] = dp[i - 1][0] + splitMergeCost;
        }
        
        // Preenchimento da matriz principal
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                // Custo do movimento (substituição)
                double moveCostValue = calculateMoveCost(ts1[i - 1], ts2[j - 1]);
                double moveOption = dp[i - 1][j - 1] + moveCostValue;
                
                // Custo do split (inserção)
                double splitOption = dp[i][j - 1] + calculateSplitCost(ts1, ts2, i - 1, j - 1);
                
                // Custo do merge (deleção)
                double mergeOption = dp[i - 1][j] + calculateMergeCost(ts1, ts2, i - 1, j - 1);
                
                // Escolher a opção de menor custo
                dp[i][j] = Math.min(Math.min(moveOption, splitOption), mergeOption);
            }
        }
        
        return dp[m][n];
    }
    
    /**
     * Calcula o custo da operação de movimento.
     */
    private double calculateMoveCost(double v1, double v2) {
        if (Math.abs(v1 - v2) < 1e-10) { // Valores essencialmente iguais
            return 0.0;
        }
        return moveCost * Math.abs(v1 - v2);
    }
    
    /**
     * Calcula o custo da operação de split.
     */
    private double calculateSplitCost(double[] ts1, double[] ts2, int i, int j) {
        if (i >= 0 && i < ts1.length && j >= 0 && j < ts2.length) {
            // Custo baseado na diferença entre valores adjacentes
            double penalty = Math.abs(ts2[j] - (i > 0 ? ts1[i - 1] : ts1[i]));
            return splitMergeCost + 0.1 * penalty;
        }
        return splitMergeCost;
    }
    
    /**
     * Calcula o custo da operação de merge.
     */
    private double calculateMergeCost(double[] ts1, double[] ts2, int i, int j) {
        if (i >= 0 && i < ts1.length && j >= 0 && j < ts2.length) {
            // Custo baseado na diferença entre valores adjacentes
            double penalty = Math.abs(ts1[i] - (j > 0 ? ts2[j - 1] : ts2[j]));
            return splitMergeCost + 0.1 * penalty;
        }
        return splitMergeCost;
    }
    
    /**
     * Extrai uma feature específica de uma série multivariada.
     */
    private double[] extractFeature(double[][] ts, int featureIndex) {
        double[] feature = new double[ts.length];
        for (int i = 0; i < ts.length; i++) {
            feature[i] = ts[i][featureIndex];
        }
        return feature;
    }
    
    /**
     * Retorna a distância MSM com informações detalhadas sobre as operações.
     */
    public MSMResult distanceWithDetails(double[] ts1, double[] ts2) {
        if (ts1 == null || ts2 == null) {
            throw new IllegalArgumentException("Séries temporais não podem ser null");
        }
        
        double distance = distance(ts1, ts2);
        
        // Calcular estatísticas aproximadas das operações
        int totalOperations = Math.max(ts1.length, ts2.length);
        int estimatedMoves = Math.min(ts1.length, ts2.length);
        int estimatedSplitsMerges = Math.abs(ts1.length - ts2.length);
        
        return new MSMResult(distance, estimatedMoves, estimatedSplitsMerges, totalOperations);
    }
    
    /**
     * Calcula custos automaticamente baseados nas estatísticas das séries.
     */
    public static MSM createAutoConfigured(double[] ts1, double[] ts2) {
        double std1 = calculateStandardDeviation(ts1);
        double std2 = calculateStandardDeviation(ts2);
        double avgStd = (std1 + std2) / 2.0;
        
        // Custos proporcionais à variabilidade dos dados
        double moveCost = Math.max(0.1, avgStd * 0.5);
        double splitMergeCost = Math.max(0.1, avgStd * 0.3);
        
        return new MSM(moveCost, splitMergeCost, false);
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
        private double moveCost = 1.0;
        private double splitMergeCost = 1.0;
        private boolean verbose = false;
        
        public Builder moveCost(double moveCost) {
            this.moveCost = moveCost;
            return this;
        }
        
        public Builder splitMergeCost(double splitMergeCost) {
            this.splitMergeCost = splitMergeCost;
            return this;
        }
        
        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }
        
        public Builder autoConfigured(double[] ts1, double[] ts2) {
            MSM auto = createAutoConfigured(ts1, ts2);
            this.moveCost = auto.moveCost;
            this.splitMergeCost = auto.splitMergeCost;
            return this;
        }
        
        public MSM build() {
            return new MSM(moveCost, splitMergeCost, verbose);
        }
    }
    
    /**
     * Resultado detalhado da distância MSM.
     */
    public static class MSMResult {
        private final double distance;
        private final int estimatedMoves;
        private final int estimatedSplitsMerges;
        private final int totalOperations;
        
        public MSMResult(double distance, int estimatedMoves, int estimatedSplitsMerges, int totalOperations) {
            this.distance = distance;
            this.estimatedMoves = estimatedMoves;
            this.estimatedSplitsMerges = estimatedSplitsMerges;
            this.totalOperations = totalOperations;
        }
        
        public double getDistance() { return distance; }
        public int getEstimatedMoves() { return estimatedMoves; }
        public int getEstimatedSplitsMerges() { return estimatedSplitsMerges; }
        public int getTotalOperations() { return totalOperations; }
        
        public double getMoveRatio() {
            return totalOperations > 0 ? (double) estimatedMoves / totalOperations : 0.0;
        }
        
        public double getSplitMergeRatio() {
            return totalOperations > 0 ? (double) estimatedSplitsMerges / totalOperations : 0.0;
        }
        
        @Override
        public String toString() {
            return String.format("MSMResult{distance=%.4f, moves=%d, splits/merges=%d, total=%d, move_ratio=%.2f}", 
                               distance, estimatedMoves, estimatedSplitsMerges, totalOperations, getMoveRatio());
        }
    }
    
    // Getters
    public double getMoveCost() { return moveCost; }
    public double getSplitMergeCost() { return splitMergeCost; }
    public boolean isVerbose() { return verbose; }
}
