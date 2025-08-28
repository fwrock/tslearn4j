package org.tslearn.examples;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.metrics.DTW;
import org.tslearn.metrics.DTWLowerBound;
import org.tslearn.metrics.DTWNeighbors;

/**
 * Demonstração da implementação DTW otimizada com diferentes estratégias de aceleração.
 * 
 * Este exemplo mostra:
 * - DTW básica vs DTW com restrições
 * - Lower bounds para aceleração
 * - Busca de vizinhos mais próximos
 * - Comparação de performance
 */
public class OptimizedDTWExample {
    
    private static final Logger logger = LoggerFactory.getLogger(OptimizedDTWExample.class);
    
    public static void main(String[] args) {
        logger.info("=".repeat(60));
        logger.info("DTW Otimizada - Demonstração de Performance");
        logger.info("=".repeat(60));
        
        // 1. Demonstração básica de DTW
        demonstrateBasicDTW();
        
        // 2. Comparação de restrições globais
        demonstrateGlobalConstraints();
        
        // 3. Lower bounds para aceleração
        demonstrateLowerBounds();
        
        // 4. Busca de vizinhos mais próximos
        demonstrateNearestNeighbors();
        
        // 5. Teste de performance em larga escala
        demonstrateScalabilityTest();
        
        logger.info("\n" + "=".repeat(60));
        logger.info("DTW Otimizada implementada com sucesso! 🚀");
        logger.info("=".repeat(60));
    }
    
    private static void demonstrateBasicDTW() {
        logger.info("\n1. Demonstração Básica de DTW");
        logger.info("-".repeat(40));
        
        // Séries temporais de exemplo
        double[] ts1 = {1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0};
        double[] ts2 = {0.5, 1.5, 2.5, 3.5, 2.5, 1.5, 0.5}; // Versão escalada
        double[] ts3 = {2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0}; // Versão deslocada
        
        DTW standardDTW = new DTW();
        
        logger.info("Séries temporais:");
        logger.info("TS1: {}", Arrays.toString(ts1));
        logger.info("TS2: {}", Arrays.toString(ts2));
        logger.info("TS3: {}", Arrays.toString(ts3));
        
        double dist12 = standardDTW.distance(ts1, ts2);
        double dist13 = standardDTW.distance(ts1, ts3);
        double dist23 = standardDTW.distance(ts2, ts3);
        
        logger.info("\nDistâncias DTW:");
        logger.info("DTW(TS1, TS2): {}", String.format("%.6f", dist12));
        logger.info("DTW(TS1, TS3): {}", String.format("%.6f", dist13));
        logger.info("DTW(TS2, TS3): {}", String.format("%.6f", dist23));
        
        // Demonstrar path alignment
        DTW.DTWResult result = standardDTW.distanceWithPath(ts1, ts2);
        logger.info("\nAlinhamento ótimo TS1-TS2:");
        logger.info("Distância: {}", String.format("%.6f", result.getDistance()));
        logger.info("Comprimento do caminho: {}", result.getPathLength());
        
        int[][] path = result.getPath();
        StringBuilder pathStr = new StringBuilder("Caminho: ");
        for (int i = 0; i < Math.min(5, path.length); i++) {
            pathStr.append(String.format("(%d,%d) ", path[i][0], path[i][1]));
        }
        if (path.length > 5) pathStr.append("...");
        logger.info(pathStr.toString());
    }
    
    private static void demonstrateGlobalConstraints() {
        logger.info("\n2. Comparação de Restrições Globais");
        logger.info("-".repeat(40));
        
        // Criar séries mais longas para melhor demonstração
        double[] ts1 = generateSineWave(50, 1.0, 0.0);
        double[] ts2 = generateSineWave(50, 1.0, Math.PI / 4); // Com shift de fase
        
        DTW standardDTW = new DTW();
        DTW sakoeChibaDTW = new DTW(5); // Band width = 5
        DTW itakuraDTW = new DTW(DTW.GlobalConstraint.ITAKURA, 0.0, false, Double.POSITIVE_INFINITY);
        
        long startTime, endTime;
        
        // DTW sem restrições
        startTime = System.nanoTime();
        double standardDist = standardDTW.distance(ts1, ts2);
        endTime = System.nanoTime();
        long standardTime = endTime - startTime;
        
        // DTW com Sakoe-Chiba
        startTime = System.nanoTime();
        double sakoeChibaDist = sakoeChibaDTW.distance(ts1, ts2);
        endTime = System.nanoTime();
        long sakoeChibaTime = endTime - startTime;
        
        // DTW com Itakura
        startTime = System.nanoTime();
        double itakuraDist = itakuraDTW.distance(ts1, ts2);
        endTime = System.nanoTime();
        long itakuraTime = endTime - startTime;
        
        logger.info("Resultados para séries de {} pontos:", ts1.length);
        logger.info("DTW Padrão:     distância={}, tempo={}ms", 
                   String.format("%.6f", standardDist), String.format("%.3f", standardTime / 1_000_000.0));
        logger.info("Sakoe-Chiba:   distância={}, tempo={}ms, speedup={}x", 
                   String.format("%.6f", sakoeChibaDist), String.format("%.3f", sakoeChibaTime / 1_000_000.0), 
                   String.format("%.2f", (double)standardTime / sakoeChibaTime));
        logger.info("Itakura:       distância={}, tempo={}ms, speedup={}x", 
                   String.format("%.6f", itakuraDist), String.format("%.3f", itakuraTime / 1_000_000.0), 
                   String.format("%.2f", (double)standardTime / itakuraTime));
    }
    
    private static void demonstrateLowerBounds() {
        logger.info("\n3. Lower Bounds para Aceleração");
        logger.info("-".repeat(40));
        
        double[] query = generateSineWave(30, 1.0, 0.0);
        double[] candidate = generateSineWave(30, 1.2, Math.PI / 6);
        
        int bandWidth = 3;
        DTW constrainedDTW = new DTW(bandWidth);
        
        long startTime, endTime;
        
        // Lower bounds
        startTime = System.nanoTime();
        double lbYi = DTWLowerBound.lbYi(query, candidate);
        endTime = System.nanoTime();
        long lbYiTime = endTime - startTime;
        
        startTime = System.nanoTime();
        double lbKeogh = DTWLowerBound.lbKeogh(query, candidate, bandWidth);
        endTime = System.nanoTime();
        long lbKeoghTime = endTime - startTime;
        
        startTime = System.nanoTime();
        double lbImproved = DTWLowerBound.lbImproved(query, candidate, bandWidth);
        endTime = System.nanoTime();
        long lbImprovedTime = endTime - startTime;
        
        startTime = System.nanoTime();
        double lbPAA = DTWLowerBound.lbPAA(query, candidate, 5);
        endTime = System.nanoTime();
        long lbPAATime = endTime - startTime;
        
        // DTW completa
        startTime = System.nanoTime();
        double dtwActual = constrainedDTW.distance(query, candidate);
        endTime = System.nanoTime();
        long dtwTime = endTime - startTime;
        
        logger.info("Lower Bounds vs DTW Real:");
        logger.info("LB_Yi:       {} (tempo: {}μs, speedup: {}x)", 
                   String.format("%.6f", lbYi), String.format("%.3f", lbYiTime / 1000.0), 
                   String.format("%.1f", (double)dtwTime / lbYiTime));
        logger.info("LB_Keogh:    {} (tempo: {}μs, speedup: {}x)", 
                   String.format("%.6f", lbKeogh), String.format("%.3f", lbKeoghTime / 1000.0), 
                   String.format("%.1f", (double)dtwTime / lbKeoghTime));
        logger.info("LB_Improved: {} (tempo: {}μs, speedup: {}x)", 
                   String.format("%.6f", lbImproved), String.format("%.3f", lbImprovedTime / 1000.0), 
                   String.format("%.1f", (double)dtwTime / lbImprovedTime));
        logger.info("LB_PAA:      {} (tempo: {}μs, speedup: {}x)", 
                   String.format("%.6f", lbPAA), String.format("%.3f", lbPAATime / 1000.0), 
                   String.format("%.1f", (double)dtwTime / lbPAATime));
        logger.info("DTW Real:    {} (tempo: {}μs)", 
                   String.format("%.6f", dtwActual), String.format("%.3f", dtwTime / 1000.0));
        
        // Verificar propriedade de lower bound
        logger.info("\nVerificação de Lower Bounds:");
        logger.info("LB_Yi ≤ DTW:       {} ({} ≤ {})", 
                   lbYi <= dtwActual + 1e-10, String.format("%.6f", lbYi), String.format("%.6f", dtwActual));
        logger.info("LB_Keogh ≤ DTW:    {} ({} ≤ {})", 
                   lbKeogh <= dtwActual + 1e-10, String.format("%.6f", lbKeogh), String.format("%.6f", dtwActual));
        logger.info("LB_Improved ≤ DTW: {} ({} ≤ {})", 
                   lbImproved <= dtwActual + 1e-10, String.format("%.6f", lbImproved), String.format("%.6f", dtwActual));
    }
    
    private static void demonstrateNearestNeighbors() {
        logger.info("\n4. Busca de Vizinhos Mais Próximos");
        logger.info("-".repeat(40));
        
        // Criar dataset de séries temporais
        Random random = new Random(42);
        int datasetSize = 50;
        int seriesLength = 20;
        
        double[][] dataset = new double[datasetSize][];
        for (int i = 0; i < datasetSize; i++) {
            double frequency = 0.5 + random.nextDouble() * 2.0; // Frequência entre 0.5 e 2.5
            double phase = random.nextDouble() * 2 * Math.PI;   // Fase aleatória
            double amplitude = 0.5 + random.nextDouble() * 1.5; // Amplitude entre 0.5 e 2.0
            dataset[i] = generateSineWave(seriesLength, amplitude, phase, frequency);
        }
        
        // Query série
        double[] query = generateSineWave(seriesLength, 1.0, 0.0, 1.0);
        
        DTW dtw = new DTW(3); // Sakoe-Chiba band width = 3
        DTWNeighbors neighbors = new DTWNeighbors(dtw, true, 4, true);
        
        int k = 5;
        
        long startTime = System.nanoTime();
        List<DTWNeighbors.NeighborResult> results = neighbors.kNearest(query, dataset, k);
        long endTime = System.nanoTime();
        
        logger.info("Busca de {} vizinhos mais próximos em dataset de {} séries:", k, datasetSize);
        logger.info("Tempo total: {}ms", String.format("%.3f", (endTime - startTime) / 1_000_000.0));
        
        logger.info("\nTop {} vizinhos mais próximos:", k);
        for (int i = 0; i < results.size(); i++) {
            DTWNeighbors.NeighborResult result = results.get(i);
            logger.info("{}. Índice: {}, Distância: {}", 
                       i + 1, result.getIndex(), String.format("%.6f", result.getDistance()));
        }
        
        // Estatísticas de lower bounds
        DTWLowerBound.LBStats stats = neighbors.getStats();
        logger.info("\nEstatísticas de otimização:");
        logger.info(stats.toString());
    }
    
    private static void demonstrateScalabilityTest() {
        logger.info("\n5. Teste de Escalabilidade");
        logger.info("-".repeat(40));
        
        int[] datasetSizes = {10, 50, 100, 200};
        int seriesLength = 30;
        int k = 3;
        
        Random random = new Random(123);
        
        logger.info("Teste de escalabilidade (k={}, comprimento={}):", k, seriesLength);
        logger.info("Dataset Size    Time (ms)       Time/Query (ms) Speedup");
        
        for (int size : datasetSizes) {
            // Gerar dataset
            double[][] dataset = new double[size][];
            for (int i = 0; i < size; i++) {
                dataset[i] = generateRandomWalk(seriesLength, random);
            }
            
            double[] query = generateRandomWalk(seriesLength, random);
            
            // Teste sem otimizações
            DTW basicDTW = new DTW();
            DTWNeighbors basicNeighbors = new DTWNeighbors(basicDTW, false, 1, false);
            
            long startTime = System.nanoTime();
            basicNeighbors.kNearest(query, dataset, k);
            long basicTime = System.nanoTime() - startTime;
            
            // Teste com otimizações
            DTW optimizedDTW = new DTW(5); // Sakoe-Chiba
            DTWNeighbors optimizedNeighbors = new DTWNeighbors(optimizedDTW, true, 4, true);
            
            startTime = System.nanoTime();
            optimizedNeighbors.kNearest(query, dataset, k);
            long optimizedTime = System.nanoTime() - startTime;
            
            double speedup = (double) basicTime / optimizedTime;
            
            logger.info("{}              {}           {}              {}x", 
                       String.format("%-15d", size),
                       String.format("%.2f", optimizedTime / 1_000_000.0),
                       String.format("%.3f", optimizedTime / 1_000_000.0 / size),
                       String.format("%.2f", speedup));
        }
    }
    
    // Métodos auxiliares para gerar séries temporais
    
    private static double[] generateSineWave(int length, double amplitude, double phase) {
        return generateSineWave(length, amplitude, phase, 1.0);
    }
    
    private static double[] generateSineWave(int length, double amplitude, double phase, double frequency) {
        double[] series = new double[length];
        for (int i = 0; i < length; i++) {
            series[i] = amplitude * Math.sin(frequency * 2 * Math.PI * i / length + phase);
        }
        return series;
    }
    
    private static double[] generateRandomWalk(int length, Random random) {
        double[] series = new double[length];
        series[0] = random.nextGaussian();
        for (int i = 1; i < length; i++) {
            series[i] = series[i-1] + random.nextGaussian() * 0.1;
        }
        return series;
    }
}
