package org.tslearn.shapelets;

import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Representa um shapelet individual - uma subsequência discriminativa de série temporal.
 * 
 * Um shapelet é uma subsequência que é útil para distinguir entre diferentes
 * classes de séries temporais. Esta implementação inclui:
 * - Os valores do shapelet
 * - Informações sobre sua origem
 * - Métricas de qualidade discriminativa
 * 
 * @author TSLearn4J
 */
public class Shapelet {
    
    private static final Logger logger = LoggerFactory.getLogger(Shapelet.class);
    
    // Dados do shapelet
    private final double[][] values;  // [length][n_features]
    private final int length;
    private final int nFeatures;
    
    // Metadados
    private final int sourceTimeSeriesIndex;
    private final int startPosition;
    private final double qualityScore;
    private final String label;
    
    // Cache para otimização
    private Double norm = null;
    private double[][] normalizedValues = null;
    
    /**
     * Construtor principal do Shapelet.
     */
    public Shapelet(double[][] values, int sourceIndex, int startPos, 
                   double quality, String label) {
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("Valores do shapelet não podem ser vazios");
        }
        
        this.values = deepCopy2D(values);
        this.length = values.length;
        this.nFeatures = values[0].length;
        this.sourceTimeSeriesIndex = sourceIndex;
        this.startPosition = startPos;
        this.qualityScore = quality;
        this.label = label != null ? label : "unknown";
        
        validateShapelet();
    }
    
    /**
     * Construtor simplificado para shapelet univariado.
     */
    public Shapelet(double[] values, int sourceIndex, int startPos, double quality) {
        this(convertToMultivariate(values), sourceIndex, startPos, quality, null);
    }
    
    /**
     * Construtor para shapelet sem metadados.
     */
    public Shapelet(double[][] values) {
        this(values, -1, -1, 0.0, null);
    }
    
    /**
     * Calcula a distância entre este shapelet e uma subsequência de série temporal.
     * 
     * @param timeSeries Série temporal completa [time_length][n_features]
     * @param startPos Posição inicial da subsequência
     * @return Distância euclidiana normalizada
     */
    public double distance(double[][] timeSeries, int startPos) {
        if (startPos + length > timeSeries.length) {
            return Double.POSITIVE_INFINITY;
        }
        
        if (timeSeries[0].length != nFeatures) {
            throw new IllegalArgumentException("Número de features não coincide");
        }
        
        double sumSquaredDiff = 0.0;
        
        for (int t = 0; t < length; t++) {
            for (int d = 0; d < nFeatures; d++) {
                double diff = values[t][d] - timeSeries[startPos + t][d];
                sumSquaredDiff += diff * diff;
            }
        }
        
        return Math.sqrt(sumSquaredDiff);
    }
    
    /**
     * Calcula a distância mínima entre este shapelet e qualquer subsequência 
     * da série temporal (sliding window).
     */
    public ShapeletMatch findBestMatch(double[][] timeSeries) {
        double minDistance = Double.POSITIVE_INFINITY;
        int bestPosition = -1;
        
        int maxStartPos = timeSeries.length - length;
        
        for (int pos = 0; pos <= maxStartPos; pos++) {
            double dist = distance(timeSeries, pos);
            if (dist < minDistance) {
                minDistance = dist;
                bestPosition = pos;
            }
        }
        
        return new ShapeletMatch(minDistance, bestPosition, this);
    }
    
    /**
     * Calcula a transformação shapelet para um conjunto de séries temporais.
     * 
     * @param dataset Dataset de séries temporais [n_series][time_length][n_features]
     * @return Array de distâncias mínimas para cada série temporal
     */
    public double[] transform(double[][][] dataset) {
        double[] distances = new double[dataset.length];
        
        for (int i = 0; i < dataset.length; i++) {
            ShapeletMatch match = findBestMatch(dataset[i]);
            distances[i] = match.getDistance();
        }
        
        return distances;
    }
    
    /**
     * Normaliza o shapelet (z-score normalization).
     */
    public Shapelet normalize() {
        if (normalizedValues != null) {
            return new Shapelet(normalizedValues, sourceTimeSeriesIndex, 
                              startPosition, qualityScore, label);
        }
        
        double[][] normalized = new double[length][nFeatures];
        
        for (int d = 0; d < nFeatures; d++) {
            // Calcular média e desvio padrão para cada feature
            double mean = 0.0;
            for (int t = 0; t < length; t++) {
                mean += values[t][d];
            }
            mean /= length;
            
            double variance = 0.0;
            for (int t = 0; t < length; t++) {
                double diff = values[t][d] - mean;
                variance += diff * diff;
            }
            double std = Math.sqrt(variance / length);
            
            // Normalizar
            if (std > 1e-8) { // Evitar divisão por zero
                for (int t = 0; t < length; t++) {
                    normalized[t][d] = (values[t][d] - mean) / std;
                }
            } else {
                // Se desvio padrão é muito pequeno, manter valores originais
                for (int t = 0; t < length; t++) {
                    normalized[t][d] = values[t][d];
                }
            }
        }
        
        this.normalizedValues = normalized;
        return new Shapelet(normalized, sourceTimeSeriesIndex, 
                          startPosition, qualityScore, label);
    }
    
    /**
     * Calcula a norma euclidiana do shapelet.
     */
    public double norm() {
        if (norm != null) {
            return norm;
        }
        
        double sumSquares = 0.0;
        for (int t = 0; t < length; t++) {
            for (int d = 0; d < nFeatures; d++) {
                sumSquares += values[t][d] * values[t][d];
            }
        }
        
        norm = Math.sqrt(sumSquares);
        return norm;
    }
    
    /**
     * Verifica se dois shapelets são similares (distância < threshold).
     */
    public boolean isSimilarTo(Shapelet other, double threshold) {
        if (other.length != this.length || other.nFeatures != this.nFeatures) {
            return false;
        }
        
        double dist = 0.0;
        for (int t = 0; t < length; t++) {
            for (int d = 0; d < nFeatures; d++) {
                double diff = this.values[t][d] - other.values[t][d];
                dist += diff * diff;
            }
        }
        
        return Math.sqrt(dist) < threshold;
    }
    
    /**
     * Cria uma subsequência da série temporal como shapelet candidato.
     */
    public static Shapelet extractSubsequence(double[][] timeSeries, int start, int length) {
        if (start + length > timeSeries.length) {
            throw new IllegalArgumentException("Subsequência ultrapassa limite da série");
        }
        
        double[][] subsequence = new double[length][timeSeries[0].length];
        for (int t = 0; t < length; t++) {
            System.arraycopy(timeSeries[start + t], 0, subsequence[t], 0, 
                           timeSeries[0].length);
        }
        
        return new Shapelet(subsequence, -1, start, 0.0, null);
    }
    
    /**
     * Valida a estrutura do shapelet.
     */
    private void validateShapelet() {
        for (int t = 0; t < length; t++) {
            if (values[t].length != nFeatures) {
                throw new IllegalArgumentException("Inconsistência no número de features");
            }
            for (int d = 0; d < nFeatures; d++) {
                if (!Double.isFinite(values[t][d])) {
                    throw new IllegalArgumentException("Valores do shapelet devem ser finitos");
                }
            }
        }
    }
    
    // Métodos utilitários
    private static double[][] deepCopy2D(double[][] original) {
        double[][] copy = new double[original.length][];
        for (int i = 0; i < original.length; i++) {
            copy[i] = original[i].clone();
        }
        return copy;
    }
    
    private static double[][] convertToMultivariate(double[] univariate) {
        double[][] multivariate = new double[univariate.length][1];
        for (int i = 0; i < univariate.length; i++) {
            multivariate[i][0] = univariate[i];
        }
        return multivariate;
    }
    
    // Getters
    public double[][] getValues() {
        return deepCopy2D(values);
    }
    
    public int getLength() {
        return length;
    }
    
    public int getNumFeatures() {
        return nFeatures;
    }
    
    public int getSourceIndex() {
        return sourceTimeSeriesIndex;
    }
    
    public int getStartPosition() {
        return startPosition;
    }
    
    public double getQualityScore() {
        return qualityScore;
    }
    
    public String getLabel() {
        return label;
    }
    
    @Override
    public String toString() {
        return String.format("Shapelet{length=%d, features=%d, quality=%.4f, source=%d:%d}", 
                           length, nFeatures, qualityScore, sourceTimeSeriesIndex, startPosition);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        Shapelet shapelet = (Shapelet) obj;
        return length == shapelet.length &&
               nFeatures == shapelet.nFeatures &&
               sourceTimeSeriesIndex == shapelet.sourceTimeSeriesIndex &&
               startPosition == shapelet.startPosition &&
               Arrays.deepEquals(values, shapelet.values);
    }
    
    @Override
    public int hashCode() {
        int result = Arrays.deepHashCode(values);
        result = 31 * result + length;
        result = 31 * result + nFeatures;
        result = 31 * result + sourceTimeSeriesIndex;
        result = 31 * result + startPosition;
        return result;
    }
    
    /**
     * Classe para representar um match/alinhamento do shapelet.
     */
    public static class ShapeletMatch {
        private final double distance;
        private final int position;
        private final Shapelet shapelet;
        
        public ShapeletMatch(double distance, int position, Shapelet shapelet) {
            this.distance = distance;
            this.position = position;
            this.shapelet = shapelet;
        }
        
        public double getDistance() { return distance; }
        public int getPosition() { return position; }
        public Shapelet getShapelet() { return shapelet; }
        
        @Override
        public String toString() {
            return String.format("Match{dist=%.4f, pos=%d}", distance, position);
        }
    }
}
