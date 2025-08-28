package org.tslearn.examples;

import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.metrics.DTW;
import org.tslearn.metrics.DTWLowerBound;
import org.tslearn.metrics.DTWNeighbors;

/**
 * Exemplo de comparação de performance entre implementações DTW básica e otimizada.
 * 
 * Demonstra os ganhos de performance obtidos com:
 * - Restrições globais (Sakoe-Chiba, Itakura)
 * - Lower bounds para pruning
 * - Processamento paralelo
 * - Otimizações de memória
 */
public class DTWPerformanceComparison {
    
    private static final Logger logger = LoggerFactory.getLogger(DTWPerformanceComparison.class);
    
    public static void main(String[] args) {
        logger.info("=" .repeat(70));
        logger.info("DTW Performance Comparison - Básica vs Otimizada");
        logger.info("=" .repeat(70));
        
        // Teste 1: Comparação de distâncias DTW
        compareDTWDistances();
        
        // Teste 2: Performance com séries de diferentes tamanhos
        compareSeriesLengthPerformance();
        
        // Teste 3: Busca k-NN em datasets de diferentes tamanhos
        compareKNNPerformance();
        
        // Teste 4: Eficácia dos lower bounds
        demonstrateLowerBoundEffectiveness();
        
        logger.info("\n" + "=" .repeat(70));
        logger.info("Conclusão: DTW Otimizada oferece speedups significativos! 🚀");
        logger.info("=" .repeat(70));
    }
    
    private static void compareDTWDistances() {
        logger.info("\n1. Comparação de Distâncias DTW");
        logger.info("-" .repeat(50));
        
        double[] ts1 = generateSineWave(30, 1.0, 0.0, 1.0);
        double[] ts2 = generateSineWave(30, 1.2, Math.PI/4, 1.2);
        
        DTW standardDTW = new DTW();
        DTW sakoeChibaDTW = new DTW(5);
        DTW itakuraDTW = new DTW(DTW.GlobalConstraint.ITAKURA, 0.0, false, Double.POSITIVE_INFINITY);
        
        long startTime = System.nanoTime();
        double standardDist = standardDTW.distance(ts1, ts2);
        long standardTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        double sakoeChibaDist = sakoeChibaDTW.distance(ts1, ts2);
        long sakoeChibaTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        double itakuraDist = itakuraDTW.distance(ts1, ts2);
        long itakuraTime = System.nanoTime() - startTime;
        
        logger.info("Séries de {} pontos:", ts1.length);
        logger.info("DTW Padrão:     {:.6f} ({}μs)", standardDist, standardTime / 1000);
        logger.info("Sakoe-Chiba:   {:.6f} ({}μs) - Speedup: {:.2f}x", 
                   sakoeChibaDist, sakoeChibaTime / 1000, (double)standardTime / sakoeChibaTime);
        logger.info("Itakura:       {:.6f} ({}μs) - Speedup: {:.2f}x", 
                   itakuraDist, itakuraTime / 1000, (double)standardTime / itakuraTime);
    }
    
    private static void compareSeriesLengthPerformance() {
        logger.info("\n2. Performance vs Tamanho das Séries");
        logger.info("-" .repeat(50));
        
        int[] lengths = {20, 50, 100, 200, 500};
        
        logger.info("Tamanho     DTW Padrão    Sakoe-Chiba   Speedup");
        logger.info("-" .repeat(50));
        
        for (int length : lengths) {
            double[] ts1 = generateSineWave(length, 1.0, 0.0, 1.0);
            double[] ts2 = generateSineWave(length, 1.1, Math.PI/6, 1.1);
            
            DTW standardDTW = new DTW();
            DTW constrainedDTW = new DTW(Math.max(5, length / 10)); // Band width adaptativo
            
            // Warm-up
            standardDTW.distance(ts1, ts2);
            constrainedDTW.distance(ts1, ts2);
            
            // Benchmark DTW padrão
            long startTime = System.nanoTime();
            for (int i = 0; i < 10; i++) {
                standardDTW.distance(ts1, ts2);
            }
            long standardTime = (System.nanoTime() - startTime) / 10;
            
            // Benchmark DTW com restrição
            startTime = System.nanoTime();
            for (int i = 0; i < 10; i++) {
                constrainedDTW.distance(ts1, ts2);
            }
            long constrainedTime = (System.nanoTime() - startTime) / 10;
            
            double speedup = (double) standardTime / constrainedTime;
            
            logger.info("{:<10} {:<12}μs {:<12}μs {:.2f}x", 
                       length, standardTime / 1000, constrainedTime / 1000, speedup);
        }
    }
    
    private static void compareKNNPerformance() {
        logger.info("\n3. Performance de Busca k-NN");
        logger.info("-" .repeat(50));
        
        int[] datasetSizes = {50, 100, 200, 500};
        int seriesLength = 50;
        int k = 5;
        
        logger.info("Dataset     Básico       Otimizado    Speedup    Pruning%%");
        logger.info("-" .repeat(55));
        
        Random random = new Random(123);
        
        for (int size : datasetSizes) {
            // Gerar dataset
            double[][] dataset = new double[size][];
            for (int i = 0; i < size; i++) {
                dataset[i] = generateRandomWalk(seriesLength, random);
            }
            double[] query = generateRandomWalk(seriesLength, random);
            
            // Configuração básica (sem otimizações)
            DTW basicDTW = new DTW();
            DTWNeighbors basicNeighbors = new DTWNeighbors(basicDTW, false, 1, false);
            
            // Configuração otimizada
            DTW optimizedDTW = new DTW(10); // Sakoe-Chiba band
            DTWNeighbors optimizedNeighbors = new DTWNeighbors(optimizedDTW, true, 4, true);
            
            // Warm-up
            basicNeighbors.kNearest(query, dataset, k);
            optimizedNeighbors.kNearest(query, dataset, k);
            
            // Benchmark básico
            long startTime = System.nanoTime();
            basicNeighbors.kNearest(query, dataset, k);
            long basicTime = System.nanoTime() - startTime;
            
            // Benchmark otimizado
            startTime = System.nanoTime();
            optimizedNeighbors.kNearest(query, dataset, k);
            long optimizedTime = System.nanoTime() - startTime;
            
            double speedup = (double) basicTime / optimizedTime;
            DTWLowerBound.LBStats stats = optimizedNeighbors.getStats();
            
            logger.info("{:<10} {:<12}ms {:<11}ms {:.2f}x      {:.1f}%%", 
                       size, 
                       basicTime / 1_000_000.0, 
                       optimizedTime / 1_000_000.0, 
                       speedup,
                       stats.getPruningRate() * 100);
        }
    }
    
    private static void demonstrateLowerBoundEffectiveness() {
        logger.info("\n4. Eficácia dos Lower Bounds");
        logger.info("-" .repeat(50));
        
        Random random = new Random(456);
        int seriesLength = 40;
        int numTests = 1000;
        
        double[] query = generateRandomWalk(seriesLength, random);
        
        int lbYiPrunes = 0;
        int lbKeoghPrunes = 0;
        int lbPAAPrunes = 0;
        int lbImprovedPrunes = 0;
        
        double threshold = 5.0; // Threshold arbitrário para pruning
        int bandWidth = 8;
        
        long totalLBTime = 0;
        long totalDTWTime = 0;
        
        for (int i = 0; i < numTests; i++) {
            double[] candidate = generateRandomWalk(seriesLength, random);
            
            // Teste lower bounds
            long startTime = System.nanoTime();
            
            double lbYi = DTWLowerBound.lbYi(query, candidate);
            if (lbYi >= threshold) {
                lbYiPrunes++;
                continue;
            }
            
            double lbKeogh = DTWLowerBound.lbKeogh(query, candidate, bandWidth);
            if (lbKeogh >= threshold) {
                lbKeoghPrunes++;
                continue;
            }
            
            double lbPAA = DTWLowerBound.lbPAA(query, candidate, 8);
            if (lbPAA >= threshold) {
                lbPAAPrunes++;
                continue;
            }
            
            double lbImproved = DTWLowerBound.lbImproved(query, candidate, bandWidth);
            if (lbImproved >= threshold) {
                lbImprovedPrunes++;
                continue;
            }
            
            long lbTime = System.nanoTime() - startTime;
            totalLBTime += lbTime;
            
            // DTW completa
            startTime = System.nanoTime();
            DTW dtw = new DTW(bandWidth);
            dtw.distance(query, candidate);
            totalDTWTime += System.nanoTime() - startTime;
        }
        
        int totalPrunes = lbYiPrunes + lbKeoghPrunes + lbPAAPrunes + lbImprovedPrunes;
        int dtwCalculations = numTests - totalPrunes;
        
        logger.info("Resultados para {} testes:", numTests);
        logger.info("LB_Yi prunes:       {} ({:.1f}%%)", lbYiPrunes, (double)lbYiPrunes / numTests * 100);
        logger.info("LB_Keogh prunes:    {} ({:.1f}%%)", lbKeoghPrunes, (double)lbKeoghPrunes / numTests * 100);
        logger.info("LB_PAA prunes:      {} ({:.1f}%%)", lbPAAPrunes, (double)lbPAAPrunes / numTests * 100);
        logger.info("LB_Improved prunes: {} ({:.1f}%%)", lbImprovedPrunes, (double)lbImprovedPrunes / numTests * 100);
        logger.info("DTW calculations:   {} ({:.1f}%%)", dtwCalculations, (double)dtwCalculations / numTests * 100);
        logger.info("");
        logger.info("Tempo médio LB:     {:.2f}μs", (double)totalLBTime / numTests / 1000);
        logger.info("Tempo médio DTW:    {:.2f}μs", (double)totalDTWTime / dtwCalculations / 1000);
        logger.info("Speedup total:      {:.2f}x", 
                   (double)(totalLBTime + totalDTWTime) / numTests / 
                   ((double)totalDTWTime / dtwCalculations));
    }
    
    // Métodos auxiliares
    
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
