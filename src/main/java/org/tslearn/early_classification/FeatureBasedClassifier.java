package org.tslearn.early_classification;

import java.util.*;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Classificador baseado em features estatísticas para early classification.
 * 
 * Extrai features estatísticas de prefixos de séries temporais e usa
 * um classificador simples baseado em distância para classificação.
 */
public class FeatureBasedClassifier implements BaseClassifier {
    
    private static final Logger logger = LoggerFactory.getLogger(FeatureBasedClassifier.class);
    
    private String[] classes;
    private boolean fitted;
    private Map<String, double[]> classCentroids;
    private Map<String, double[]> classStds;
    private int numFeatures;
    
    /**
     * Construtor padrão.
     */
    public FeatureBasedClassifier() {
        this.fitted = false;
    }
    
    @Override
    public void fit(double[][][] X, String[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Dados não podem ser null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("Número de amostras deve ser igual ao número de labels");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("Dataset não pode estar vazio");
        }
        
        // Extrair classes únicas
        Set<String> uniqueClasses = new HashSet<>(Arrays.asList(y));
        this.classes = uniqueClasses.toArray(new String[0]);
        Arrays.sort(this.classes);
        
        // Extrair features de todas as séries
        List<double[]> allFeatures = new ArrayList<>();
        for (double[][] ts : X) {
            double[] features = extractFeatures(ts);
            allFeatures.add(features);
        }
        
        this.numFeatures = allFeatures.get(0).length;
        
        // Calcular centroides e desvios padrão por classe
        this.classCentroids = new HashMap<>();
        this.classStds = new HashMap<>();
        
        for (String cls : classes) {
            List<double[]> classFeatures = new ArrayList<>();
            
            // Coletar features desta classe
            for (int i = 0; i < y.length; i++) {
                if (y[i].equals(cls)) {
                    classFeatures.add(allFeatures.get(i));
                }
            }
            
            if (!classFeatures.isEmpty()) {
                // Calcular centroide e desvio padrão
                double[] centroid = calculateCentroid(classFeatures);
                double[] std = calculateStandardDeviation(classFeatures, centroid);
                
                classCentroids.put(cls, centroid);
                classStds.put(cls, std);
            }
        }
        
        this.fitted = true;
    }
    
    @Override
    public String predict(double[][] timeSeries) {
        Map<String, Double> probabilities = predictProba(timeSeries);
        return probabilities.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(classes[0]);
    }
    
    @Override
    public Map<String, Double> predictProba(double[][] timeSeries) {
        if (!fitted) {
            throw new RuntimeException("Classificador deve ser treinado antes da predição");
        }
        if (timeSeries == null || timeSeries.length == 0) {
            throw new IllegalArgumentException("Série temporal não pode estar vazia");
        }
        
        // Extrair features da série
        double[] features = extractFeatures(timeSeries);
        
        // Calcular distâncias para cada classe
        Map<String, Double> distances = new HashMap<>();
        
        for (String cls : classes) {
            double[] centroid = classCentroids.get(cls);
            double[] std = classStds.get(cls);
            
            if (centroid != null && std != null) {
                double distance = calculateNormalizedDistance(features, centroid, std);
                distances.put(cls, distance);
            }
        }
        
        // Converter distâncias em probabilidades
        return convertDistancesToProbabilities(distances);
    }
    
    /**
     * Extrai features estatísticas de uma série temporal.
     */
    private double[] extractFeatures(double[][] timeSeries) {
        List<Double> features = new ArrayList<>();
        
        // Features por dimensão
        for (int dim = 0; dim < timeSeries[0].length; dim++) {
            double[] series = extractDimension(timeSeries, dim);
            DescriptiveStatistics stats = new DescriptiveStatistics(series);
            
            // Features básicas
            features.add(stats.getMean());
            features.add(stats.getStandardDeviation());
            features.add(stats.getMin());
            features.add(stats.getMax());
            features.add(stats.getPercentile(50)); // Mediana
            
            // Features de forma
            features.add(calculateSkewness(series));
            features.add(calculateKurtosis(series));
            
            // Features de tendência
            features.add(calculateSlope(series));
            features.add(calculateVariation(series));
            
            // Features de autocorrelação
            features.add(calculateAutocorrelation(series, 1));
            if (series.length > 2) {
                features.add(calculateAutocorrelation(series, 2));
            } else {
                features.add(0.0);
            }
        }
        
        // Features globais
        features.add(calculateEnergy(timeSeries));
        features.add(calculateComplexity(timeSeries));
        
        return features.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * Extrai dimensão específica de série multivariada.
     */
    private double[] extractDimension(double[][] ts, int dimension) {
        double[] result = new double[ts.length];
        for (int i = 0; i < ts.length; i++) {
            result[i] = dimension < ts[i].length ? ts[i][dimension] : 0.0;
        }
        return result;
    }
    
    /**
     * Calcula skewness (assimetria).
     */
    private double calculateSkewness(double[] series) {
        DescriptiveStatistics stats = new DescriptiveStatistics(series);
        double mean = stats.getMean();
        double std = stats.getStandardDeviation();
        
        if (std == 0) return 0.0;
        
        double sum = 0.0;
        for (double value : series) {
            sum += Math.pow((value - mean) / std, 3);
        }
        
        return sum / series.length;
    }
    
    /**
     * Calcula kurtosis (curtose).
     */
    private double calculateKurtosis(double[] series) {
        DescriptiveStatistics stats = new DescriptiveStatistics(series);
        double mean = stats.getMean();
        double std = stats.getStandardDeviation();
        
        if (std == 0) return 0.0;
        
        double sum = 0.0;
        for (double value : series) {
            sum += Math.pow((value - mean) / std, 4);
        }
        
        return (sum / series.length) - 3.0; // Excess kurtosis
    }
    
    /**
     * Calcula inclinação (slope) usando regressão linear simples.
     */
    private double calculateSlope(double[] series) {
        if (series.length < 2) return 0.0;
        
        double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0;
        int n = series.length;
        
        for (int i = 0; i < n; i++) {
            double x = i;
            double y = series[i];
            
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
        }
        
        double denominator = n * sumX2 - sumX * sumX;
        if (Math.abs(denominator) < 1e-10) return 0.0;
        
        return (n * sumXY - sumX * sumY) / denominator;
    }
    
    /**
     * Calcula variação normalizada.
     */
    private double calculateVariation(double[] series) {
        DescriptiveStatistics stats = new DescriptiveStatistics(series);
        double mean = stats.getMean();
        double std = stats.getStandardDeviation();
        
        return mean != 0 ? std / Math.abs(mean) : 0.0;
    }
    
    /**
     * Calcula autocorrelação com lag específico.
     */
    private double calculateAutocorrelation(double[] series, int lag) {
        if (series.length <= lag) return 0.0;
        
        DescriptiveStatistics stats = new DescriptiveStatistics(series);
        double mean = stats.getMean();
        double variance = stats.getVariance();
        
        if (variance == 0) return 0.0;
        
        double sum = 0.0;
        int count = 0;
        
        for (int i = 0; i < series.length - lag; i++) {
            sum += (series[i] - mean) * (series[i + lag] - mean);
            count++;
        }
        
        return count > 0 ? (sum / count) / variance : 0.0;
    }
    
    /**
     * Calcula energia total da série.
     */
    private double calculateEnergy(double[][] timeSeries) {
        double energy = 0.0;
        
        for (double[] point : timeSeries) {
            for (double value : point) {
                energy += value * value;
            }
        }
        
        return energy;
    }
    
    /**
     * Calcula complexidade baseada em variações.
     */
    private double calculateComplexity(double[][] timeSeries) {
        if (timeSeries.length < 2) return 0.0;
        
        double complexity = 0.0;
        
        for (int dim = 0; dim < timeSeries[0].length; dim++) {
            double[] series = extractDimension(timeSeries, dim);
            
            double variations = 0.0;
            for (int i = 1; i < series.length; i++) {
                variations += Math.abs(series[i] - series[i-1]);
            }
            
            complexity += variations;
        }
        
        return complexity;
    }
    
    /**
     * Calcula centroide de um conjunto de features.
     */
    private double[] calculateCentroid(List<double[]> features) {
        if (features.isEmpty()) return new double[0];
        
        int numFeatures = features.get(0).length;
        double[] centroid = new double[numFeatures];
        
        for (double[] feature : features) {
            for (int i = 0; i < Math.min(numFeatures, feature.length); i++) {
                centroid[i] += feature[i];
            }
        }
        
        for (int i = 0; i < numFeatures; i++) {
            centroid[i] /= features.size();
        }
        
        return centroid;
    }
    
    /**
     * Calcula desvio padrão de um conjunto de features.
     */
    private double[] calculateStandardDeviation(List<double[]> features, double[] centroid) {
        if (features.isEmpty()) return new double[0];
        
        int numFeatures = centroid.length;
        double[] variance = new double[numFeatures];
        
        for (double[] feature : features) {
            for (int i = 0; i < Math.min(numFeatures, feature.length); i++) {
                double diff = feature[i] - centroid[i];
                variance[i] += diff * diff;
            }
        }
        
        double[] std = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            std[i] = Math.sqrt(variance[i] / features.size());
        }
        
        return std;
    }
    
    /**
     * Calcula distância normalizada entre features e centroide.
     */
    private double calculateNormalizedDistance(double[] features, double[] centroid, double[] std) {
        double distance = 0.0;
        
        for (int i = 0; i < Math.min(features.length, centroid.length); i++) {
            double diff = features[i] - centroid[i];
            double normalizedDiff = std[i] > 0 ? diff / std[i] : diff;
            distance += normalizedDiff * normalizedDiff;
        }
        
        return Math.sqrt(distance);
    }
    
    /**
     * Converte distâncias em probabilidades usando softmax.
     */
    private Map<String, Double> convertDistancesToProbabilities(Map<String, Double> distances) {
        Map<String, Double> probabilities = new HashMap<>();
        
        if (distances.isEmpty()) {
            double uniform = 1.0 / classes.length;
            for (String cls : classes) {
                probabilities.put(cls, uniform);
            }
            return probabilities;
        }
        
        // Usar softmax invertido (menor distância = maior probabilidade)
        double maxDistance = distances.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        double temperature = 1.0;
        
        double sum = 0.0;
        for (String cls : classes) {
            double distance = distances.getOrDefault(cls, maxDistance);
            double score = Math.exp(-(distance / maxDistance) / temperature);
            probabilities.put(cls, score);
            sum += score;
        }
        
        // Normalizar
        if (sum > 0) {
            for (String cls : classes) {
                probabilities.put(cls, probabilities.get(cls) / sum);
            }
        } else {
            double uniform = 1.0 / classes.length;
            for (String cls : classes) {
                probabilities.put(cls, uniform);
            }
        }
        
        return probabilities;
    }
    
    @Override
    public boolean isFitted() {
        return fitted;
    }
    
    @Override
    public String[] getClasses() {
        return classes != null ? classes.clone() : null;
    }
}
