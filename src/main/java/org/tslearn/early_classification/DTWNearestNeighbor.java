package org.tslearn.early_classification;

import java.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.metrics.DTW;

/**
 * Classificador 1-NN baseado em DTW para early classification.
 * 
 * Usa Dynamic Time Warping para encontrar a série mais similar
 * no conjunto de treinamento e retorna sua classe.
 */
public class DTWNearestNeighbor implements BaseClassifier {
    
    private static final Logger logger = LoggerFactory.getLogger(DTWNearestNeighbor.class);
    
    private DTW dtw;
    private double[][][] trainingData;
    private String[] trainingLabels;
    private String[] classes;
    private boolean fitted;
    private int maxLength;
    
    /**
     * Construtor padrão.
     */
    public DTWNearestNeighbor() {
        // DTW com restrição moderada para performance
        this.dtw = new DTW.Builder()
            .sakoeChibaRadius(5)
            .build();
        this.fitted = false;
    }
    
    /**
     * Construtor com DTW customizada.
     */
    public DTWNearestNeighbor(DTW dtw) {
        this.dtw = dtw != null ? dtw : new DTW();
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
        
        // Armazenar dados de treinamento
        this.trainingData = new double[X.length][][];
        for (int i = 0; i < X.length; i++) {
            this.trainingData[i] = Arrays.copyOf(X[i], X[i].length);
        }
        this.trainingLabels = Arrays.copyOf(y, y.length);
        
        // Extrair classes únicas
        Set<String> uniqueClasses = new HashSet<>(Arrays.asList(y));
        this.classes = uniqueClasses.toArray(new String[0]);
        Arrays.sort(this.classes);
        
        // Determinar comprimento máximo
        this.maxLength = Arrays.stream(X).mapToInt(x -> x.length).max().orElse(100);
        
        this.fitted = true;
    }
    
    @Override
    public String predict(double[][] timeSeries) {
        if (!fitted) {
            throw new RuntimeException("Classificador deve ser treinado antes da predição");
        }
        if (timeSeries == null || timeSeries.length == 0) {
            throw new IllegalArgumentException("Série temporal não pode estar vazia");
        }
        
        double minDistance = Double.MAX_VALUE;
        String bestClass = classes[0];
        
        // Encontrar vizinho mais próximo
        for (int i = 0; i < trainingData.length; i++) {
            try {
                double[][] trainingTs = adaptSeriesLength(trainingData[i], timeSeries.length);
                double distance = calculateDistance(timeSeries, trainingTs);
                
                if (distance < minDistance) {
                    minDistance = distance;
                    bestClass = trainingLabels[i];
                }
            } catch (Exception e) {
                // Skip this training sample if error occurs
                continue;
            }
        }
        
        return bestClass;
    }
    
    @Override
    public Map<String, Double> predictProba(double[][] timeSeries) {
        if (!fitted) {
            throw new RuntimeException("Classificador deve ser treinado antes da predição");
        }
        if (timeSeries == null || timeSeries.length == 0) {
            throw new IllegalArgumentException("Série temporal não pode estar vazia");
        }
        
        // Calcular distâncias para todas as amostras de treinamento
        List<DistanceResult> distances = new ArrayList<>();
        
        for (int i = 0; i < trainingData.length; i++) {
            try {
                double[][] trainingTs = adaptSeriesLength(trainingData[i], timeSeries.length);
                double distance = calculateDistance(timeSeries, trainingTs);
                distances.add(new DistanceResult(distance, trainingLabels[i]));
            } catch (Exception e) {
                // Skip problematic samples
                continue;
            }
        }
        
        if (distances.isEmpty()) {
            // Fallback: probabilidades uniformes
            Map<String, Double> uniform = new HashMap<>();
            double prob = 1.0 / classes.length;
            for (String cls : classes) {
                uniform.put(cls, prob);
            }
            return uniform;
        }
        
        // Ordenar por distância
        distances.sort(Comparator.comparingDouble(d -> d.distance));
        
        // Usar k vizinhos mais próximos (k = min(5, número de amostras))
        int k = Math.min(5, distances.size());
        Map<String, Double> classVotes = new HashMap<>();
        
        for (String cls : classes) {
            classVotes.put(cls, 0.0);
        }
        
        // Pesos baseados em distância (inverso da distância)
        double totalWeight = 0.0;
        
        for (int i = 0; i < k; i++) {
            DistanceResult dr = distances.get(i);
            double weight = 1.0 / (1.0 + dr.distance); // Evitar divisão por zero
            
            classVotes.put(dr.label, classVotes.get(dr.label) + weight);
            totalWeight += weight;
        }
        
        // Normalizar
        Map<String, Double> probabilities = new HashMap<>();
        for (String cls : classes) {
            probabilities.put(cls, totalWeight > 0 ? classVotes.get(cls) / totalWeight : 0.0);
        }
        
        return probabilities;
    }
    
    /**
     * Adapta comprimento da série de treinamento ao prefixo atual.
     */
    private double[][] adaptSeriesLength(double[][] trainingSeries, int targetLength) {
        if (trainingSeries.length <= targetLength) {
            return trainingSeries;
        }
        
        // Truncar série de treinamento
        return Arrays.copyOf(trainingSeries, targetLength);
    }
    
    /**
     * Calcula distância entre duas séries temporais.
     */
    private double calculateDistance(double[][] ts1, double[][] ts2) {
        if (ts1[0].length == 1 && ts2[0].length == 1) {
            // Séries univariadas
            double[] s1 = extractUnivariate(ts1);
            double[] s2 = extractUnivariate(ts2);
            return dtw.distance(s1, s2);
        } else {
            // Séries multivariadas - usar média das distâncias por dimensão
            int numDimensions = Math.min(ts1[0].length, ts2[0].length);
            double totalDistance = 0.0;
            
            for (int dim = 0; dim < numDimensions; dim++) {
                double[] s1 = extractDimension(ts1, dim);
                double[] s2 = extractDimension(ts2, dim);
                totalDistance += dtw.distance(s1, s2);
            }
            
            return totalDistance / numDimensions;
        }
    }
    
    /**
     * Extrai série univariada (primeira dimensão).
     */
    private double[] extractUnivariate(double[][] ts) {
        double[] result = new double[ts.length];
        for (int i = 0; i < ts.length; i++) {
            result[i] = ts[i][0];
        }
        return result;
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
    
    @Override
    public boolean isFitted() {
        return fitted;
    }
    
    @Override
    public String[] getClasses() {
        return classes != null ? classes.clone() : null;
    }
    
    /**
     * Classe auxiliar para armazenar resultado de distância.
     */
    private static class DistanceResult {
        final double distance;
        final String label;
        
        DistanceResult(double distance, String label) {
            this.distance = distance;
            this.label = label;
        }
    }
}
