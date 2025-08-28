package org.tslearn.examples;

import java.util.Arrays;
import java.util.Random;
import org.tslearn.shapelets.Shapelet;
import org.tslearn.shapelets.ShapeletTransform;

/**
 * Exemplo demonstrativo da transformação Shapelet.
 * 
 * Este exemplo mostra como usar shapelets para extrair features
 * discriminativas de séries temporais e utilizá-las para classificação.
 */
public class ShapeletTransformExample {
    
    public static void main(String[] args) {
        System.out.println("=== Exemplo de Transformação Shapelet ===\n");
        
        // Gerar dataset sintético
        SyntheticDataset dataset = generateSyntheticDataset();
        
        // Demonstrar Shapelet individual
        demonstrateIndividualShapelet(dataset);
        
        // Demonstrar ShapeletTransform
        demonstrateShapeletTransform(dataset);
        
        // Análise de sensibilidade
        performSensitivityAnalysis(dataset);
        
        System.out.println("\n=== Exemplo concluído ===");
    }
    
    /**
     * Gera um dataset sintético com padrões discriminativos.
     */
    static class SyntheticDataset {
        double[][][] X;
        String[] y;
        double[][][] X_test;
        String[] y_test;
        
        SyntheticDataset(double[][][] X, String[] y, double[][][] X_test, String[] y_test) {
            this.X = X;
            this.y = y;
            this.X_test = X_test;
            this.y_test = y_test;
        }
    }
    
    private static SyntheticDataset generateSyntheticDataset() {
        System.out.println("Gerando dataset sintético...");
        
        Random random = new Random(42);
        int trainSize = 200;
        int testSize = 100;
        int timeLength = 100;
        int nFeatures = 1;
        
        // Padrões discriminativos para cada classe
        double[][] patternA = {{1.0}, {2.0}, {3.0}, {2.0}, {1.0}}; // Pico ascendente-descendente
        double[][] patternB = {{-1.0}, {-2.0}, {-1.0}, {0.0}, {1.0}}; // Vale seguido de subida
        double[][] patternC = {{0.0}, {1.0}, {0.0}, {-1.0}, {0.0}}; // Oscilação
        
        // Gerar dados de treino
        double[][][] X_train = new double[trainSize][timeLength][nFeatures];
        String[] y_train = new String[trainSize];
        
        for (int i = 0; i < trainSize; i++) {
            String className = "Class" + (char)('A' + (i % 3));
            y_train[i] = className;
            
            // Preencher com ruído
            for (int t = 0; t < timeLength; t++) {
                X_train[i][t][0] = random.nextGaussian() * 0.5;
            }
            
            // Inserir padrão discriminativo
            double[][] pattern;
            switch (className) {
                case "ClassA": pattern = patternA; break;
                case "ClassB": pattern = patternB; break;
                default: pattern = patternC; break;
            }
            
            int insertPos = random.nextInt(timeLength - pattern.length + 1);
            for (int j = 0; j < pattern.length; j++) {
                X_train[i][insertPos + j][0] = pattern[j][0] + random.nextGaussian() * 0.1;
            }
        }
        
        // Gerar dados de teste
        double[][][] X_test = new double[testSize][timeLength][nFeatures];
        String[] y_test = new String[testSize];
        
        for (int i = 0; i < testSize; i++) {
            String className = "Class" + (char)('A' + (i % 3));
            y_test[i] = className;
            
            // Preencher com ruído
            for (int t = 0; t < timeLength; t++) {
                X_test[i][t][0] = random.nextGaussian() * 0.5;
            }
            
            // Inserir padrão discriminativo
            double[][] pattern;
            switch (className) {
                case "ClassA": pattern = patternA; break;
                case "ClassB": pattern = patternB; break;
                default: pattern = patternC; break;
            }
            
            int insertPos = random.nextInt(timeLength - pattern.length + 1);
            for (int j = 0; j < pattern.length; j++) {
                X_test[i][insertPos + j][0] = pattern[j][0] + random.nextGaussian() * 0.1;
            }
        }
        
        System.out.printf("Dataset gerado: %d treino, %d teste, comprimento %d\n\n", 
                         trainSize, testSize, timeLength);
        
        return new SyntheticDataset(X_train, y_train, X_test, y_test);
    }
    
    /**
     * Demonstra o uso de Shapelet individual.
     */
    private static void demonstrateIndividualShapelet(SyntheticDataset dataset) {
        System.out.println("=== Demonstração de Shapelet Individual ===");
        
        // Criar um shapelet do padrão conhecido (com formato correto)
        double[][] patternValues = {
            {1.0}, {2.0}, {3.0}, {2.0}, {1.0}
        };
        Shapelet shapelet = new Shapelet(patternValues, -1, -1, 0.0, "PatternA");
        
        System.out.println("Shapelet criado com comprimento: " + shapelet.getLength());
        System.out.println("Valores: " + Arrays.deepToString(shapelet.getValues()));
        
        // Testar matching em algumas séries
        System.out.println("\nTeste de matching:");
        for (int i = 0; i < 5; i++) {
            Shapelet.ShapeletMatch match = shapelet.findBestMatch(dataset.X[i]);
            System.out.printf("Série %d (classe %s): distância=%.4f, posição=%d\n", 
                             i, dataset.y[i], match.getDistance(), match.getPosition());
        }
        
        // Testar transformação
        System.out.println("\nTransformação de dataset pequeno:");
        double[][][] smallX = Arrays.copyOf(dataset.X, 10);
        double[] transformed = shapelet.transform(smallX);
        
        for (int i = 0; i < transformed.length; i++) {
            System.out.printf("Série %d: distância=%.4f\n", i, transformed[i]);
        }
        
        System.out.println();
    }
    
    /**
     * Demonstra o ShapeletTransform completo.
     */
    private static void demonstrateShapeletTransform(SyntheticDataset dataset) {
        System.out.println("=== Demonstração de ShapeletTransform ===");
        
        // Configurar transformador
        ShapeletTransform transform = new ShapeletTransform.Builder()
                .numShapelets(20)
                .minShapeletLength(3)
                .maxShapeletLength(10)
                .maxCandidates(1000)
                .selectionMethod(ShapeletTransform.ShapeletSelectionMethod.INFORMATION_GAIN)
                .initializationMethod(ShapeletTransform.InitializationMethod.RANDOM)
                .removeSimilar(true)
                .similarityThreshold(0.1)
                .verbose(true)
                .randomSeed(42L)
                .build();
        
        System.out.println("Transformador configurado");
        
        // Treinar transformador
        System.out.println("\nTreinando transformador...");
        long startTime = System.currentTimeMillis();
        transform.fit(dataset.X, dataset.y);
        long trainTime = System.currentTimeMillis() - startTime;
        
        System.out.printf("Treinamento concluído em %d ms\n", trainTime);
        System.out.printf("Shapelets descobertos: %d\n", transform.getNumShapelets());
        
        // Analisar shapelets descobertos
        analyzeDiscoveredShapelets(transform);
        
        // Transformar dados de treino
        System.out.println("\nTransformando dados de treino...");
        startTime = System.currentTimeMillis();
        double[][] X_train_transformed = transform.transform(dataset.X);
        long transformTime = System.currentTimeMillis() - startTime;
        
        System.out.printf("Transformação concluída em %d ms\n", transformTime);
        System.out.printf("Dimensões transformadas: [%d, %d]\n", 
                         X_train_transformed.length, X_train_transformed[0].length);
        
        // Transformar dados de teste
        System.out.println("\nTransformando dados de teste...");
        double[][] X_test_transformed = transform.transform(dataset.X_test);
        
        // Analisar separabilidade das classes
        analyzeClassSeparability(X_train_transformed, dataset.y, transform);
        
        System.out.println();
    }
    
    /**
     * Analisa os shapelets descobertos.
     */
    private static void analyzeDiscoveredShapelets(ShapeletTransform transform) {
        System.out.println("\n--- Análise de Shapelets Descobertos ---");
        
        var shapelets = transform.getShapelets();
        if (shapelets == null || shapelets.isEmpty()) {
            System.out.println("Nenhum shapelet disponível para análise");
            return;
        }
        
        // Estatísticas gerais
        double[] qualities = shapelets.stream()
                .mapToDouble(Shapelet::getQualityScore)
                .toArray();
        
        int[] lengths = shapelets.stream()
                .mapToInt(Shapelet::getLength)
                .toArray();
        
        System.out.printf("Qualidade média: %.4f\n", Arrays.stream(qualities).average().orElse(0.0));
        System.out.printf("Qualidade máxima: %.4f\n", Arrays.stream(qualities).max().orElse(0.0));
        System.out.printf("Comprimento médio: %.1f\n", Arrays.stream(lengths).average().orElse(0.0));
        
        // Top 5 shapelets
        System.out.println("\nTop 5 shapelets:");
        for (int i = 0; i < Math.min(5, shapelets.size()); i++) {
            Shapelet s = shapelets.get(i);
            System.out.printf("  %d. Qualidade=%.4f, Comprimento=%d, Origem=%s\n", 
                             i+1, s.getQualityScore(), s.getLength(), s.getLabel());
        }
    }
    
    /**
     * Analisa a separabilidade das classes no espaço transformado.
     */
    private static void analyzeClassSeparability(double[][] X_transformed, String[] y, 
                                                ShapeletTransform transform) {
        System.out.println("\n--- Análise de Separabilidade ---");
        
        String[] classes = transform.getClasses();
        
        // Calcular centroides por classe
        for (String className : classes) {
            double[] centroid = new double[X_transformed[0].length];
            int count = 0;
            
            for (int i = 0; i < y.length; i++) {
                if (y[i].equals(className)) {
                    for (int j = 0; j < centroid.length; j++) {
                        centroid[j] += X_transformed[i][j];
                    }
                    count++;
                }
            }
            
            if (count > 0) {
                for (int j = 0; j < centroid.length; j++) {
                    centroid[j] /= count;
                }
                
                // Calcular distância média ao centroide
                double avgDistance = 0.0;
                int samples = 0;
                
                for (int i = 0; i < y.length; i++) {
                    if (y[i].equals(className)) {
                        double distance = 0.0;
                        for (int j = 0; j < centroid.length; j++) {
                            distance += Math.pow(X_transformed[i][j] - centroid[j], 2);
                        }
                        avgDistance += Math.sqrt(distance);
                        samples++;
                    }
                }
                
                avgDistance /= samples;
                
                System.out.printf("Classe %s: %d amostras, distância média ao centroide=%.4f\n", 
                                 className, count, avgDistance);
            }
        }
        
        // Análise dimensional
        System.out.println("\nEstatísticas por dimensão (primeiras 5):");
        for (int j = 0; j < Math.min(5, X_transformed[0].length); j++) {
            double[] feature = new double[X_transformed.length];
            for (int i = 0; i < X_transformed.length; i++) {
                feature[i] = X_transformed[i][j];
            }
            
            double mean = Arrays.stream(feature).average().orElse(0.0);
            double std = Math.sqrt(Arrays.stream(feature)
                    .map(x -> Math.pow(x - mean, 2))
                    .average().orElse(0.0));
            
            System.out.printf("  Feature %d: média=%.4f, std=%.4f\n", j, mean, std);
        }
    }
    
    /**
     * Realiza análise de sensibilidade dos parâmetros.
     */
    private static void performSensitivityAnalysis(SyntheticDataset dataset) {
        System.out.println("\n=== Análise de Sensibilidade ===");
        
        // Usar subset menor para análise rápida
        int subsetSize = 50;
        double[][][] X_subset = Arrays.copyOf(dataset.X, subsetSize);
        String[] y_subset = Arrays.copyOf(dataset.y, subsetSize);
        
        // Testar diferentes números de shapelets
        System.out.println("Testando diferentes números de shapelets:");
        int[] numShapeletsOptions = {5, 10, 20, 50};
        
        for (int numShapelets : numShapeletsOptions) {
            long startTime = System.currentTimeMillis();
            
            ShapeletTransform transform = new ShapeletTransform.Builder()
                    .numShapelets(numShapelets)
                    .minShapeletLength(3)
                    .maxShapeletLength(8)
                    .maxCandidates(500)
                    .verbose(false)
                    .randomSeed(42L)
                    .build();
            
            transform.fit(X_subset, y_subset);
            double[][] transformed = transform.transform(X_subset);
            
            long duration = System.currentTimeMillis() - startTime;
            
            System.out.printf("  %d shapelets: %d ms, dimensões [%d, %d]\n", 
                             numShapelets, duration, transformed.length, transformed[0].length);
        }
        
        // Testar diferentes métodos de seleção
        System.out.println("\nTestando métodos de seleção:");
        ShapeletTransform.ShapeletSelectionMethod[] methods = {
            ShapeletTransform.ShapeletSelectionMethod.INFORMATION_GAIN,
            ShapeletTransform.ShapeletSelectionMethod.F_STATISTIC,
            ShapeletTransform.ShapeletSelectionMethod.MOODS_MEDIAN
        };
        
        for (ShapeletTransform.ShapeletSelectionMethod method : methods) {
            long startTime = System.currentTimeMillis();
            
            ShapeletTransform transform = new ShapeletTransform.Builder()
                    .numShapelets(10)
                    .selectionMethod(method)
                    .verbose(false)
                    .randomSeed(42L)
                    .build();
            
            transform.fit(X_subset, y_subset);
            
            long duration = System.currentTimeMillis() - startTime;
            double avgQuality = transform.getShapelets().stream()
                    .mapToDouble(Shapelet::getQualityScore)
                    .average().orElse(0.0);
            
            System.out.printf("  %s: %d ms, qualidade média=%.4f\n", 
                             method, duration, avgQuality);
        }
    }
}
