package org.tslearn.early_classification;

import java.util.*;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.shapelets.ShapeletTransform;

/**
 * Classificador baseado em Shapelets para early classification.
 * 
 * Usa shapelets discriminativos extraídos durante o treinamento para
 * classificar prefixos de séries temporais.
 */
public class ShapeletClassifier implements BaseClassifier {
    
    private static final Logger logger = LoggerFactory.getLogger(ShapeletClassifier.class);
    
    private ShapeletTransform shapeletTransform;
    private String[] classes;
    private boolean fitted;
    private Map<String, List<double[]>> classFeatures;
    private Map<String, DescriptiveStatistics[]> classStats;
    
    /**
     * Construtor padrão.
     */
    public ShapeletClassifier() {
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
        
        try {
            // Extrair classes únicas
            Set<String> uniqueClasses = new HashSet<>(Arrays.asList(y));
            this.classes = uniqueClasses.toArray(new String[0]);
            Arrays.sort(this.classes);
            
            // Configurar ShapeletTransform para early classification
            this.shapeletTransform = new ShapeletTransform.Builder()
                .numShapelets(Math.min(50, X.length / 2))
                .minShapeletLength(3)
                .maxShapeletLength(Math.max(10, X[0].length / 4))
                .selectionMethod(ShapeletTransform.ShapeletSelectionMethod.INFORMATION_GAIN)
                .removeSimilar(true)
                .verbose(false)
                .randomSeed(42L)
                .build();
            
            // Treinar transform com dados completos
            double[][] features = shapeletTransform.fitTransform(X, y);
            
            // Organizar features por classe
            this.classFeatures = new HashMap<>();
            this.classStats = new HashMap<>();
            
            for (String cls : classes) {
                classFeatures.put(cls, new ArrayList<>());
                classStats.put(cls, new DescriptiveStatistics[features[0].length]);
                
                for (int i = 0; i < features[0].length; i++) {
                    classStats.get(cls)[i] = new DescriptiveStatistics();
                }
            }
            
            // Agrupar features por classe
            for (int i = 0; i < features.length; i++) {
                String cls = y[i];
                classFeatures.get(cls).add(features[i]);
                
                for (int j = 0; j < features[i].length; j++) {
                    classStats.get(cls)[j].addValue(features[i][j]);
                }
            }
            
            this.fitted = true;
            
        } catch (Exception e) {
            logger.error("Erro no treinamento do ShapeletClassifier: {}", e.getMessage());
            throw new RuntimeException("Falha no treinamento", e);
        }
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
        
        try {
            // Transformar série usando shapelets
            double[][][] singleSeries = {timeSeries};
            double[][] features = shapeletTransform.transform(singleSeries);
            double[] seriesFeatures = features[0];
            
            // Calcular probabilidades baseadas em distância para cada classe
            Map<String, Double> distances = new HashMap<>();
            
            for (String cls : classes) {
                double distance = calculateClassDistance(seriesFeatures, cls);
                distances.put(cls, distance);
            }
            
            // Converter distâncias em probabilidades (normalizar)
            return convertDistancesToProbabilities(distances);
            
        } catch (Exception e) {
            // Fallback: retornar probabilidades uniformes
            Map<String, Double> uniform = new HashMap<>();
            double prob = 1.0 / classes.length;
            for (String cls : classes) {
                uniform.put(cls, prob);
            }
            return uniform;
        }
    }
    
    /**
     * Calcula distância de features para uma classe específica.
     */
    private double calculateClassDistance(double[] features, String targetClass) {
        DescriptiveStatistics[] stats = classStats.get(targetClass);
        double distance = 0.0;
        
        for (int i = 0; i < Math.min(features.length, stats.length); i++) {
            double mean = stats[i].getMean();
            double std = stats[i].getStandardDeviation();
            
            if (std > 0) {
                // Distância normalizada
                distance += Math.pow((features[i] - mean) / std, 2);
            } else {
                // Se desvio padrão é 0, usar distância absoluta
                distance += Math.abs(features[i] - mean);
            }
        }
        
        return Math.sqrt(distance);
    }
    
    /**
     * Converte distâncias em probabilidades.
     */
    private Map<String, Double> convertDistancesToProbabilities(Map<String, Double> distances) {
        Map<String, Double> probabilities = new HashMap<>();
        
        // Usar softmax com temperatura para converter distâncias em probabilidades
        double temperature = 1.0;
        double maxDistance = distances.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        
        double sum = 0.0;
        for (String cls : classes) {
            double distance = distances.getOrDefault(cls, maxDistance);
            // Inverter distância (menor distância = maior probabilidade)
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
            // Fallback uniforme
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
