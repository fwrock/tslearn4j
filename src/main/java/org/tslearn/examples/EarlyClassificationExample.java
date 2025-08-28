package org.tslearn.examples;

import java.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.early_classification.*;

/**
 * Exemplo completo de Early Classification para Séries Temporais.
 * 
 * Demonstra como usar o EarlyClassifier para classificar séries temporais
 * antes de observar toda a sequência, analisando o trade-off entre
 * precisão e earliness.
 */
public class EarlyClassificationExample {
    
    private static final Logger logger = LoggerFactory.getLogger(EarlyClassificationExample.class);
    
    public static void main(String[] args) {
        System.out.println("=== Early Classification para Séries Temporais ===\n");
        
        try {
            // Gerar dataset sintético
            DatasetResult dataset = generateSyntheticDataset();
            
            // Demonstrar diferentes estratégias
            demonstrateStrategies(dataset);
            
            // Análise de trade-off
            analyzeTradeoffs(dataset);
            
            // Casos de uso específicos
            demonstrateUseCases(dataset);
            
        } catch (Exception e) {
            logger.error("Erro no exemplo: {}", e.getMessage(), e);
            e.printStackTrace();
        }
        
        System.out.println("\n=== Exemplo concluído ===");
    }
    
    /**
     * Gera dataset sintético com diferentes padrões.
     */
    private static DatasetResult generateSyntheticDataset() {
        System.out.println("1. Gerando Dataset Sintético");
        
        Random random = new Random(42);
        int numSamples = 200;
        int timeLength = 50;
        int numFeatures = 2;
        
        double[][][] X = new double[numSamples][timeLength][numFeatures];
        String[] y = new String[numSamples];
        
        String[] classes = {"Crescente", "Decrescente", "Senoidal", "Constante"};
        
        for (int i = 0; i < numSamples; i++) {
            String cls = classes[i % classes.length];
            y[i] = cls;
            
            double[][] timeSeries = generateTimeSeriesForClass(cls, timeLength, numFeatures, random);
            X[i] = timeSeries;
        }
        
        // Split train/test
        int trainSize = (int) (numSamples * 0.7);
        double[][][] X_train = Arrays.copyOf(X, trainSize);
        String[] y_train = Arrays.copyOf(y, trainSize);
        
        double[][][] X_test = Arrays.copyOfRange(X, trainSize, numSamples);
        String[] y_test = Arrays.copyOfRange(y, trainSize, numSamples);
        
        System.out.printf("Dataset criado: %d amostras de treino, %d de teste, %d classes\n", 
                         X_train.length, X_test.length, classes.length);
        System.out.printf("Dimensões: [%d][%d][%d]\n\n", timeLength, numFeatures, classes.length);
        
        return new DatasetResult(X_train, y_train, X_test, y_test, classes);
    }
    
    /**
     * Gera série temporal para uma classe específica.
     */
    private static double[][] generateTimeSeriesForClass(String cls, int length, int numFeatures, Random random) {
        double[][] series = new double[length][numFeatures];
        
        for (int dim = 0; dim < numFeatures; dim++) {
            for (int t = 0; t < length; t++) {
                double noise = random.nextGaussian() * 0.1;
                double base = 0.0;
                
                switch (cls) {
                    case "Crescente":
                        base = t * 0.1 + noise;
                        break;
                    case "Decrescente":
                        base = (length - t) * 0.1 + noise;
                        break;
                    case "Senoidal":
                        base = Math.sin(t * 0.3) + noise;
                        break;
                    case "Constante":
                        base = 1.0 + noise;
                        break;
                }
                
                series[t][dim] = base + dim * 0.5; // Offset por dimensão
            }
        }
        
        return series;
    }
    
    /**
     * Demonstra diferentes estratégias de early classification.
     */
    private static void demonstrateStrategies(DatasetResult dataset) {
        System.out.println("2. Comparando Estratégias de Early Classification");
        
        EarlyClassifier.StoppingStrategy[] strategies = {
            EarlyClassifier.StoppingStrategy.CONFIDENCE_THRESHOLD,
            EarlyClassifier.StoppingStrategy.MARGIN_BASED,
            EarlyClassifier.StoppingStrategy.PROBABILITY_STABILIZATION
        };
        
        for (EarlyClassifier.StoppingStrategy strategy : strategies) {
            System.out.printf("\n--- Estratégia: %s ---\n", strategy);
            
            EarlyClassifier classifier = new EarlyClassifier.Builder()
                .stoppingStrategy(strategy)
                .confidenceThreshold(0.8)
                .marginThreshold(0.3)
                .minLength(5)
                .stepSize(2)
                .verbose(false)
                .randomSeed(42)
                .build();
            
            // Treinar
            long startTime = System.currentTimeMillis();
            classifier.fit(dataset.X_train, dataset.y_train);
            long trainTime = System.currentTimeMillis() - startTime;
            
            // Avaliar
            startTime = System.currentTimeMillis();
            EarlyClassifier.EvaluationResult evaluation = classifier.evaluate(dataset.X_test, dataset.y_test);
            long testTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("Resultados: %s\n", evaluation);
            System.out.printf("Tempo de treino: %dms, Tempo de teste: %dms\n", trainTime, testTime);
            
            // Mostrar exemplo de classificação passo a passo
            if (dataset.X_test.length > 0) {
                EarlyClassifier.EarlyResult result = classifier.predictEarly(dataset.X_test[0]);
                System.out.printf("Exemplo de classificação: %s\n", result);
                System.out.printf("Passos da classificação: %d\n", result.getSteps().size());
            }
        }
        
        System.out.println();
    }
    
    /**
     * Analisa trade-offs entre precisão e earliness.
     */
    private static void analyzeTradeoffs(DatasetResult dataset) {
        System.out.println("3. Análise de Trade-offs (Precisão vs. Earliness)");
        
        double[] confidenceThresholds = {0.6, 0.7, 0.8, 0.9, 0.95};
        
        System.out.printf("%-15s %-10s %-12s %-12s %-12s\n", 
                         "Threshold", "Accuracy", "Earliness", "Confidence", "Harmonic");
        System.out.println("-".repeat(65));
        
        for (double threshold : confidenceThresholds) {
            EarlyClassifier classifier = new EarlyClassifier.Builder()
                .stoppingStrategy(EarlyClassifier.StoppingStrategy.CONFIDENCE_THRESHOLD)
                .confidenceThreshold(threshold)
                .minLength(3)
                .stepSize(1)
                .verbose(false)
                .build();
            
            classifier.fit(dataset.X_train, dataset.y_train);
            EarlyClassifier.EvaluationResult evaluation = classifier.evaluate(dataset.X_test, dataset.y_test);
            
            System.out.printf("%-15.2f %-10.4f %-12.4f %-12.4f %-12.4f\n",
                             threshold, 
                             evaluation.getAccuracy(),
                             evaluation.getAverageEarliness(),
                             evaluation.getAverageConfidence(),
                             evaluation.getHarmonicMean());
        }
        
        System.out.println();
    }
    
    /**
     * Demonstra casos de uso específicos.
     */
    private static void demonstrateUseCases(DatasetResult dataset) {
        System.out.println("4. Casos de Uso Específicos");
        
        // Caso 1: Classificação ultra-rápida (máximo 20% da série)
        System.out.println("\n--- Caso 1: Classificação Ultra-Rápida ---");
        EarlyClassifier fastClassifier = new EarlyClassifier.Builder()
            .stoppingStrategy(EarlyClassifier.StoppingStrategy.CONFIDENCE_THRESHOLD)
            .confidenceThreshold(0.7)
            .maxLength(10) // Máximo 20% da série (50 * 0.2)
            .minLength(3)
            .stepSize(1)
            .verbose(false)
            .build();
        
        fastClassifier.fit(dataset.X_train, dataset.y_train);
        EarlyClassifier.EvaluationResult fastResult = fastClassifier.evaluate(dataset.X_test, dataset.y_test);
        System.out.printf("Ultra-rápido: %s\n", fastResult);
        
        // Caso 2: Classificação conservadora (alta precisão)
        System.out.println("\n--- Caso 2: Classificação Conservadora ---");
        EarlyClassifier conservativeClassifier = new EarlyClassifier.Builder()
            .stoppingStrategy(EarlyClassifier.StoppingStrategy.MARGIN_BASED)
            .marginThreshold(0.4)
            .minLength(10)
            .stepSize(2)
            .verbose(false)
            .build();
        
        conservativeClassifier.fit(dataset.X_train, dataset.y_train);
        EarlyClassifier.EvaluationResult conservativeResult = conservativeClassifier.evaluate(dataset.X_test, dataset.y_test);
        System.out.printf("Conservador: %s\n", conservativeResult);
        
        // Caso 3: Ensemble com múltiplos classificadores
        System.out.println("\n--- Caso 3: Ensemble Personalizado ---");
        EarlyClassifier ensembleClassifier = new EarlyClassifier.Builder()
            .addClassifier(new DTWNearestNeighbor())
            .addClassifier(new FeatureBasedClassifier())
            .stoppingStrategy(EarlyClassifier.StoppingStrategy.ENSEMBLE_CONSENSUS)
            .aggregationMethod(EarlyClassifier.AggregationMethod.WEIGHTED_CONFIDENCE)
            .minLength(5)
            .stepSize(2)
            .verbose(false)
            .build();
        
        ensembleClassifier.fit(dataset.X_train, dataset.y_train);
        EarlyClassifier.EvaluationResult ensembleResult = ensembleClassifier.evaluate(dataset.X_test, dataset.y_test);
        System.out.printf("Ensemble: %s\n", ensembleResult);
        
        // Caso 4: Análise detalhada de uma predição
        System.out.println("\n--- Caso 4: Análise Detalhada ---");
        if (dataset.X_test.length > 0) {
            EarlyClassifier detailedClassifier = new EarlyClassifier.Builder()
                .stoppingStrategy(EarlyClassifier.StoppingStrategy.CONFIDENCE_THRESHOLD)
                .confidenceThreshold(0.8)
                .minLength(5)
                .stepSize(1)
                .verbose(true)
                .build();
            
            detailedClassifier.fit(dataset.X_train, dataset.y_train);
            EarlyClassifier.EarlyResult detailedResult = detailedClassifier.predictEarly(dataset.X_test[0]);
            
            System.out.printf("Série de teste: comprimento=%d, classe real=%s\n", 
                             dataset.X_test[0].length, dataset.y_test[0]);
            System.out.printf("Resultado: %s\n", detailedResult);
            
            System.out.println("Evolução da classificação:");
            for (EarlyClassifier.ClassificationStep step : detailedResult.getSteps()) {
                System.out.printf("  t=%d: confiança=%.3f, deve_parar=%s\n", 
                                 step.getTimePoint(), step.getConfidence(), step.shouldStop());
                
                // Mostrar probabilidades da classe predita
                String predicted = detailedResult.getPredictedClass();
                double prob = step.getProbabilities().getOrDefault(predicted, 0.0);
                System.out.printf("        prob(%s)=%.3f\n", predicted, prob);
            }
        }
        
        System.out.println();
    }
    
    /**
     * Classe auxiliar para armazenar dataset.
     */
    private static class DatasetResult {
        final double[][][] X_train;
        final String[] y_train;
        final double[][][] X_test;
        final String[] y_test;
        final String[] classes;
        
        DatasetResult(double[][][] X_train, String[] y_train, double[][][] X_test, 
                     String[] y_test, String[] classes) {
            this.X_train = X_train;
            this.y_train = y_train;
            this.X_test = X_test;
            this.y_test = y_test;
            this.classes = classes;
        }
    }
}
