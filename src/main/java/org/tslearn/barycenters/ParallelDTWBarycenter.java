package org.tslearn.barycenters;

import org.apache.commons.math3.util.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.metrics.DTW;
import org.tslearn.utils.ArrayUtils;
import org.tslearn.utils.ParallelUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Implementação paralelizada do DBA (DTW Barycenter Averaging) para calcular
 * baricentros de séries temporais usando Dynamic Time Warping.
 * 
 * Esta versão otimizada oferece:
 * - Paralelização da computação de alinhamentos DTW
 * - Processamento em lote para grandes conjuntos de dados
 * - Balanceamento adaptivo de carga
 * - Otimizações específicas para diferentes tamanhos de dados
 * 
 * Baseado no algoritmo de Petitjean et al. (2011) com otimizações paralelas.
 * 
 * @author TSLearn4J
 */
public class ParallelDTWBarycenter {
    
    private static final Logger logger = LoggerFactory.getLogger(ParallelDTWBarycenter.class);
    
    // Limite para decidir quando usar paralelização
    private static final int PARALLEL_THRESHOLD = 50;
    private static final int BATCH_SIZE = 10;
    
    /**
     * Calcula o baricentro DTW de um conjunto de séries temporais usando paralelização.
     * 
     * @param timeSeries Array de séries temporais [n_samples][time_length][n_features]
     * @param initialization Baricentro inicial
     * @param maxIter Número máximo de iterações
     * @return Baricentro DTW otimizado
     */
    public static double[][] compute(double[][][] timeSeries, double[][] initialization, int maxIter) {
        if (timeSeries == null || timeSeries.length == 0) {
            throw new IllegalArgumentException("Conjunto de séries temporais não pode ser vazio");
        }
        
        if (initialization == null) {
            // Usar média euclidiana como inicialização padrão
            initialization = EuclideanBarycenter.compute(timeSeries);
        }
        
        // Decidir estratégia baseada no tamanho dos dados
        if (timeSeries.length >= PARALLEL_THRESHOLD) {
            return computeDBAParallel(timeSeries, initialization, maxIter, 1e-6);
        } else {
            return DTWBarycenter.compute(timeSeries, initialization, maxIter);
        }
    }
    
    /**
     * Calcula o baricentro DTW com inicialização automática.
     */
    public static double[][] compute(double[][][] timeSeries, int maxIter) {
        double[][] initialization = EuclideanBarycenter.compute(timeSeries);
        return compute(timeSeries, initialization, maxIter);
    }
    
    /**
     * Implementação principal do algoritmo DBA paralelo.
     */
    private static double[][] computeDBAParallel(double[][][] timeSeries, double[][] initialization, 
                                                int maxIter, double tolerance) {
        
        int nSamples = timeSeries.length;
        int barycenterLength = initialization.length;
        int nFeatures = initialization[0].length;
        
        // Copiar inicialização
        double[][] barycenter = ArrayUtils.deepCopy2D(initialization);
        double[][] newBarycenter = new double[barycenterLength][nFeatures];
        
        // Configurar paralelização adaptiva
        ParallelUtils.ParallelConfig config = ParallelUtils.ParallelConfig.forDataSize(nSamples);
        
        double previousCost = Double.POSITIVE_INFINITY;
        
        for (int iter = 0; iter < maxIter; iter++) {
            // Resetar acumuladores
            double[][][] accumulator = new double[barycenterLength][nFeatures][1];
            int[][] counts = new int[barycenterLength][nFeatures];
            
            // Processar séries temporais em paralelo
            double totalCost = computeAlignmentsParallel(timeSeries, barycenter, accumulator, counts, config);
            
            // Calcular novo baricentro
            computeNewBarycenter(accumulator, counts, newBarycenter, barycenter);
            
            // Verificar convergência
            double avgCost = totalCost / nSamples;
            if (FastMath.abs(previousCost - avgCost) < tolerance) {
                logger.debug("DBA paralelo convergiu na iteração {} com custo {:.6f}", iter + 1, avgCost);
                break;
            }
            
            // Atualizar baricentro
            barycenter = ArrayUtils.deepCopy2D(newBarycenter);
            previousCost = avgCost;
        }
        
        return barycenter;
    }
    
    /**
     * Computa alinhamentos DTW em paralelo usando diferentes estratégias.
     */
    private static double computeAlignmentsParallel(double[][][] timeSeries, double[][] barycenter,
                                                   double[][][] accumulator, int[][] counts,
                                                   ParallelUtils.ParallelConfig config) {
        
        int nSamples = timeSeries.length;
        
        if (nSamples <= BATCH_SIZE) {
            // Para conjuntos pequenos, usar paralelização simples
            return computeAlignmentsSimpleParallel(timeSeries, barycenter, accumulator, counts);
        } else {
            // Para conjuntos grandes, usar processamento em lote
            return computeAlignmentsBatched(timeSeries, barycenter, accumulator, counts, config);
        }
    }
    
    /**
     * Paralelização simples por série temporal.
     */
    private static double computeAlignmentsSimpleParallel(double[][][] timeSeries, double[][] barycenter,
                                                         double[][][] accumulator, int[][] counts) {
        
        DTW dtw = new DTW();
        Object lock = new Object();
        
        return IntStream.range(0, timeSeries.length)
                .parallel()
                .mapToDouble(s -> {
                    double[][] ts = timeSeries[s];
                    
                    // Calcular caminho DTW ótimo
                    DTW.DTWPathResult pathResult = dtw.distanceWithPath(barycenter, ts);
                    List<int[]> path = pathResult.getPath();
                    double distance = pathResult.getDistance();
                    
                    // Acumular valores baseado no alinhamento (thread-safe)
                    synchronized (lock) {
                        for (int[] step : path) {
                            int barycenterIdx = step[0];
                            int timeSeriesIdx = step[1];
                            
                            for (int d = 0; d < barycenter[0].length; d++) {
                                accumulator[barycenterIdx][d][0] += ts[timeSeriesIdx][d];
                                counts[barycenterIdx][d]++;
                            }
                        }
                    }
                    
                    return distance;
                })
                .sum();
    }
    
    /**
     * Processamento em lote para otimizar a paralelização.
     */
    private static double computeAlignmentsBatched(double[][][] timeSeries, double[][] barycenter,
                                                  double[][][] accumulator, int[][] counts,
                                                  ParallelUtils.ParallelConfig config) {
        
        int nSamples = timeSeries.length;
        int batchSize = Math.max(BATCH_SIZE, config.getChunkSize());
        int numBatches = (nSamples + batchSize - 1) / batchSize;
        
        // Criar tasks para cada lote
        List<CompletableFuture<BatchResult>> futures = new ArrayList<>();
        
        for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
            final int startIdx = batchIndex * batchSize;
            final int endIdx = Math.min(startIdx + batchSize, nSamples);
            
            CompletableFuture<BatchResult> future = CompletableFuture.supplyAsync(() -> {
                return processBatch(timeSeries, barycenter, startIdx, endIdx);
            }, ParallelUtils.getGlobalPool());
            
            futures.add(future);
        }
        
        // Coletar resultados e acumular
        double totalCost = 0.0;
        Object lock = new Object();
        
        for (CompletableFuture<BatchResult> future : futures) {
            try {
                BatchResult result = future.get();
                totalCost += result.cost;
                
                // Combinar acumuladores (thread-safe)
                synchronized (lock) {
                    combineAccumulators(accumulator, counts, result.accumulator, result.counts);
                }
                
            } catch (Exception e) {
                logger.warn("Erro ao processar lote: {}", e.getMessage());
            }
        }
        
        return totalCost;
    }
    
    /**
     * Processa um lote de séries temporais.
     */
    private static BatchResult processBatch(double[][][] timeSeries, double[][] barycenter, 
                                          int startIdx, int endIdx) {
        
        int barycenterLength = barycenter.length;
        int nFeatures = barycenter[0].length;
        
        // Acumuladores locais para este lote
        double[][][] localAccumulator = new double[barycenterLength][nFeatures][1];
        int[][] localCounts = new int[barycenterLength][nFeatures];
        
        DTW dtw = new DTW();
        double localCost = 0.0;
        
        for (int s = startIdx; s < endIdx; s++) {
            double[][] ts = timeSeries[s];
            
            // Calcular caminho DTW ótimo
            DTW.DTWPathResult pathResult = dtw.distanceWithPath(barycenter, ts);
            List<int[]> path = pathResult.getPath();
            localCost += pathResult.getDistance();
            
            // Acumular valores baseado no alinhamento
            for (int[] step : path) {
                int barycenterIdx = step[0];
                int timeSeriesIdx = step[1];
                
                for (int d = 0; d < nFeatures; d++) {
                    localAccumulator[barycenterIdx][d][0] += ts[timeSeriesIdx][d];
                    localCounts[barycenterIdx][d]++;
                }
            }
        }
        
        return new BatchResult(localAccumulator, localCounts, localCost);
    }
    
    /**
     * Resultado do processamento de um lote.
     */
    private static class BatchResult {
        final double[][][] accumulator;
        final int[][] counts;
        final double cost;
        
        BatchResult(double[][][] accumulator, int[][] counts, double cost) {
            this.accumulator = accumulator;
            this.counts = counts;
            this.cost = cost;
        }
    }
    
    /**
     * Combina acumuladores de diferentes lotes.
     */
    private static void combineAccumulators(double[][][] globalAccumulator, int[][] globalCounts,
                                          double[][][] localAccumulator, int[][] localCounts) {
        
        int barycenterLength = globalAccumulator.length;
        int nFeatures = globalAccumulator[0].length;
        
        for (int t = 0; t < barycenterLength; t++) {
            for (int d = 0; d < nFeatures; d++) {
                globalAccumulator[t][d][0] += localAccumulator[t][d][0];
                globalCounts[t][d] += localCounts[t][d];
            }
        }
    }
    
    /**
     * Computa o novo baricentro baseado nos acumuladores.
     */
    private static void computeNewBarycenter(double[][][] accumulator, int[][] counts,
                                           double[][] newBarycenter, double[][] oldBarycenter) {
        
        int barycenterLength = accumulator.length;
        int nFeatures = accumulator[0].length;
        
        for (int t = 0; t < barycenterLength; t++) {
            for (int d = 0; d < nFeatures; d++) {
                if (counts[t][d] > 0) {
                    newBarycenter[t][d] = accumulator[t][d][0] / counts[t][d];
                } else {
                    // Manter valor anterior se não há alinhamento
                    newBarycenter[t][d] = oldBarycenter[t][d];
                }
            }
        }
    }
    
    /**
     * Calcula múltiplos baricentros DTW em paralelo com otimização de recursos.
     */
    public static double[][][] computeMultiple(double[][][][] timeSeriesGroups, int maxIter) {
        int nGroups = timeSeriesGroups.length;
        
        if (nGroups <= 1) {
            // Um único grupo - usar método padrão
            double[][][] result = new double[nGroups][][];
            if (nGroups == 1) {
                result[0] = compute(timeSeriesGroups[0], maxIter);
            }
            return result;
        }
        
        // Múltiplos grupos - paralelizar por grupo
        return IntStream.range(0, nGroups)
                .parallel()
                .mapToObj(g -> compute(timeSeriesGroups[g], maxIter))
                .toArray(double[][][]::new);
    }
    
    /**
     * Versão otimizada para datasets muito grandes usando chunking adaptivo.
     */
    public static double[][] computeLargeDataset(double[][][] timeSeries, double[][] initialization, 
                                                int maxIter, int chunkSize) {
        
        if (timeSeries.length <= chunkSize) {
            return compute(timeSeries, initialization, maxIter);
        }
        
        // Processar em chunks para gerenciar memória
        int nChunks = (timeSeries.length + chunkSize - 1) / chunkSize;
        logger.info("Processando dataset grande em {} chunks de tamanho {}", nChunks, chunkSize);
        
        double[][] barycenter = ArrayUtils.deepCopy2D(initialization);
        
        for (int iter = 0; iter < maxIter; iter++) {
            double[][] newBarycenter = processChunksForIteration(timeSeries, barycenter, chunkSize);
            
            // Verificar convergência (simplificado para datasets grandes)
            double diff = computeDifference(barycenter, newBarycenter);
            if (diff < 1e-4) { // Tolerância relaxada para datasets grandes
                logger.debug("Convergência alcançada na iteração {} com diferença {:.6f}", iter + 1, diff);
                break;
            }
            
            barycenter = newBarycenter;
        }
        
        return barycenter;
    }
    
    /**
     * Processa chunks para uma iteração do algoritmo.
     */
    private static double[][] processChunksForIteration(double[][][] timeSeries, double[][] barycenter, 
                                                       int chunkSize) {
        
        int nSamples = timeSeries.length;
        int nChunks = (nSamples + chunkSize - 1) / chunkSize;
        int barycenterLength = barycenter.length;
        int nFeatures = barycenter[0].length;
        
        // Acumuladores globais
        double[][][] globalAccumulator = new double[barycenterLength][nFeatures][1];
        int[][] globalCounts = new int[barycenterLength][nFeatures];
        
        // Processar chunks em paralelo
        IntStream.range(0, nChunks)
                .parallel()
                .forEach(chunkIndex -> {
                    int startIdx = chunkIndex * chunkSize;
                    int endIdx = Math.min(startIdx + chunkSize, nSamples);
                    
                    // Criar chunk
                    double[][][] chunk = new double[endIdx - startIdx][][];
                    System.arraycopy(timeSeries, startIdx, chunk, 0, endIdx - startIdx);
                    
                    // Processar chunk e combinar resultados
                    BatchResult result = processBatch(timeSeries, barycenter, startIdx, endIdx);
                    
                    synchronized (globalAccumulator) {
                        combineAccumulators(globalAccumulator, globalCounts, 
                                          result.accumulator, result.counts);
                    }
                });
        
        // Computar novo baricentro
        double[][] newBarycenter = new double[barycenterLength][nFeatures];
        computeNewBarycenter(globalAccumulator, globalCounts, newBarycenter, barycenter);
        
        return newBarycenter;
    }
    
    /**
     * Calcula a diferença entre dois baricentros.
     */
    private static double computeDifference(double[][] barycenter1, double[][] barycenter2) {
        double sum = 0.0;
        int count = 0;
        
        for (int t = 0; t < barycenter1.length; t++) {
            for (int d = 0; d < barycenter1[t].length; d++) {
                double diff = barycenter1[t][d] - barycenter2[t][d];
                sum += diff * diff;
                count++;
            }
        }
        
        return Math.sqrt(sum / count);
    }
    
    /**
     * Calcula o custo total de um baricentro em relação a um conjunto de séries.
     * Versão paralelizada para melhor performance.
     */
    public static double computeCost(double[][] barycenter, double[][][] timeSeries) {
        DTW dtw = new DTW();
        
        return IntStream.range(0, timeSeries.length)
                .parallel()
                .mapToDouble(i -> dtw.distance(barycenter, timeSeries[i]))
                .sum();
    }
    
    /**
     * Configurações para controle fino da paralelização.
     */
    public static class ParallelConfig {
        private final int parallelThreshold;
        private final int batchSize;
        private final boolean useAdaptiveChunking;
        
        public ParallelConfig(int parallelThreshold, int batchSize, boolean useAdaptiveChunking) {
            this.parallelThreshold = parallelThreshold;
            this.batchSize = batchSize;
            this.useAdaptiveChunking = useAdaptiveChunking;
        }
        
        public static ParallelConfig defaultConfig() {
            return new ParallelConfig(PARALLEL_THRESHOLD, BATCH_SIZE, true);
        }
        
        public static ParallelConfig forDataSize(int dataSize) {
            int threshold = Math.min(PARALLEL_THRESHOLD, dataSize / 4);
            int batch = Math.max(BATCH_SIZE, dataSize / 20);
            return new ParallelConfig(threshold, batch, dataSize > 1000);
        }
        
        public int getParallelThreshold() { return parallelThreshold; }
        public int getBatchSize() { return batchSize; }
        public boolean useAdaptiveChunking() { return useAdaptiveChunking; }
    }
}
