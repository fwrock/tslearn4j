package org.tslearn.utils;

import java.util.concurrent.*;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

/**
 * Utilitários para computação paralela otimizada para algoritmos de machine learning
 * em séries temporais.
 * 
 * Fornece thread pools configuráveis, particionamento de trabalho otimizado,
 * e padrões comuns de paralelização para operações computacionalmente intensivas.
 * 
 * @author TSLearn4J
 */
public class ParallelUtils {
    
    // Thread pool global para operações pesadas
    private static final int DEFAULT_PARALLELISM = Runtime.getRuntime().availableProcessors();
    private static volatile ForkJoinPool globalPool;
    private static volatile ExecutorService globalExecutor;
    
    // Configurações de paralelização
    private static final int MIN_PARALLEL_THRESHOLD = 100;
    private static final int CHUNK_SIZE_MULTIPLIER = 4;
    
    static {
        initializeGlobalPools();
    }
    
    /**
     * Inicializa os pools de threads globais.
     */
    private static void initializeGlobalPools() {
        globalPool = new ForkJoinPool(DEFAULT_PARALLELISM);
        globalExecutor = Executors.newWorkStealingPool(DEFAULT_PARALLELISM);
        
        // Shutdown hook para limpeza
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (globalPool != null && !globalPool.isShutdown()) {
                globalPool.shutdown();
            }
            if (globalExecutor != null && !globalExecutor.isShutdown()) {
                globalExecutor.shutdown();
            }
        }));
    }
    
    /**
     * Obtém o pool ForkJoin global.
     */
    public static ForkJoinPool getGlobalPool() {
        return globalPool;
    }
    
    /**
     * Obtém o executor global.
     */
    public static ExecutorService getGlobalExecutor() {
        return globalExecutor;
    }
    
    /**
     * Verifica se uma operação deve ser paralelizada baseado no tamanho dos dados.
     */
    public static boolean shouldParallelize(int dataSize) {
        return dataSize >= MIN_PARALLEL_THRESHOLD;
    }
    
    /**
     * Calcula o tamanho ótimo de chunk para paralelização.
     */
    public static int getOptimalChunkSize(int totalSize, int parallelism) {
        if (totalSize < MIN_PARALLEL_THRESHOLD) {
            return totalSize;
        }
        
        int baseChunkSize = Math.max(1, totalSize / (parallelism * CHUNK_SIZE_MULTIPLIER));
        return Math.min(baseChunkSize, totalSize / 2);
    }
    
    /**
     * Executa uma operação em paralelo sobre um array de índices.
     * 
     * @param size Tamanho do array de índices
     * @param operation Função que processa cada índice
     * @param <T> Tipo do resultado
     * @return Array com os resultados
     */
    public static <T> T[] parallelMap(int size, IntFunction<T> operation, IntFunction<T[]> arrayConstructor) {
        if (!shouldParallelize(size)) {
            return IntStream.range(0, size)
                    .mapToObj(operation)
                    .toArray(arrayConstructor);
        }
        
        return IntStream.range(0, size)
                .parallel()
                .mapToObj(operation)
                .toArray(arrayConstructor);
    }
    
    /**
     * Executa uma operação em paralelo sobre um array de doubles.
     */
    public static double[] parallelMapToDouble(int size, IntFunction<Double> operation) {
        if (!shouldParallelize(size)) {
            return IntStream.range(0, size)
                    .mapToDouble(i -> operation.apply(i))
                    .toArray();
        }
        
        return IntStream.range(0, size)
                .parallel()
                .mapToDouble(i -> operation.apply(i))
                .toArray();
    }
    
    /**
     * Executa uma operação em paralelo sobre um array de ints.
     */
    public static int[] parallelMapToInt(int size, IntFunction<Integer> operation) {
        if (!shouldParallelize(size)) {
            return IntStream.range(0, size)
                    .map(i -> operation.apply(i))
                    .toArray();
        }
        
        return IntStream.range(0, size)
                .parallel()
                .map(i -> operation.apply(i))
                .toArray();
    }
    
    /**
     * Executa uma operação paralela usando ForkJoinPool customizado.
     */
    public static <T> T executeInCustomPool(int parallelism, Callable<T> task) throws Exception {
        if (parallelism <= 1) {
            return task.call();
        }
        
        try (ForkJoinPool customPool = new ForkJoinPool(parallelism)) {
            return customPool.submit(task).get();
        }
    }
    
    /**
     * Divide um trabalho em chunks e executa em paralelo.
     */
    public static <T> T[] parallelChunkProcess(int totalSize, int chunkSize, 
                                             ChunkProcessor<T> processor,
                                             IntFunction<T[]> arrayConstructor) {
        
        if (!shouldParallelize(totalSize)) {
            T result = processor.processChunk(0, totalSize);
            T[] results = arrayConstructor.apply(1);
            results[0] = result;
            return results;
        }
        
        int numChunks = (totalSize + chunkSize - 1) / chunkSize;
        
        return IntStream.range(0, numChunks)
                .parallel()
                .mapToObj(chunkIndex -> {
                    int startIdx = chunkIndex * chunkSize;
                    int endIdx = Math.min(startIdx + chunkSize, totalSize);
                    return processor.processChunk(startIdx, endIdx);
                })
                .toArray(arrayConstructor);
    }
    
    /**
     * Redução paralela com função de combinação.
     */
    public static <T> T parallelReduce(int size, IntFunction<T> mapper, 
                                     java.util.function.BinaryOperator<T> combiner, T identity) {
        if (!shouldParallelize(size)) {
            T result = identity;
            for (int i = 0; i < size; i++) {
                result = combiner.apply(result, mapper.apply(i));
            }
            return result;
        }
        
        return IntStream.range(0, size)
                .parallel()
                .mapToObj(mapper)
                .reduce(identity, combiner);
    }
    
    /**
     * Computação paralela de matriz 2D com trabalho dividido por linhas.
     */
    public static double[][] parallelMatrix2D(int rows, int cols, MatrixElementFunction function) {
        double[][] result = new double[rows][cols];
        
        if (!shouldParallelize(rows * cols)) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i][j] = function.compute(i, j);
                }
            }
            return result;
        }
        
        IntStream.range(0, rows)
                .parallel()
                .forEach(i -> {
                    for (int j = 0; j < cols; j++) {
                        result[i][j] = function.compute(i, j);
                    }
                });
        
        return result;
    }
    
    /**
     * Suma paralela de array.
     */
    public static double parallelSum(double[] array) {
        if (!shouldParallelize(array.length)) {
            double sum = 0.0;
            for (double value : array) {
                sum += value;
            }
            return sum;
        }
        
        return IntStream.range(0, array.length)
                .parallel()
                .mapToDouble(i -> array[i])
                .sum();
    }
    
    /**
     * Encontra o índice do valor mínimo em paralelo.
     */
    public static int parallelArgMin(double[] array) {
        if (!shouldParallelize(array.length)) {
            int minIndex = 0;
            double minValue = array[0];
            for (int i = 1; i < array.length; i++) {
                if (array[i] < minValue) {
                    minValue = array[i];
                    minIndex = i;
                }
            }
            return minIndex;
        }
        
        return IntStream.range(0, array.length)
                .parallel()
                .reduce((i, j) -> array[i] <= array[j] ? i : j)
                .orElse(0);
    }
    
    /**
     * Batch processing paralelo para grandes datasets.
     */
    public static <T, R> R[] parallelBatchProcess(T[] data, int batchSize,
                                                Function<T[], R> batchProcessor,
                                                IntFunction<R[]> arrayConstructor) {
        
        int numBatches = (data.length + batchSize - 1) / batchSize;
        
        return IntStream.range(0, numBatches)
                .parallel()
                .mapToObj(batchIndex -> {
                    int startIdx = batchIndex * batchSize;
                    int endIdx = Math.min(startIdx + batchSize, data.length);
                    
                    T[] batch = (T[]) new Object[endIdx - startIdx];
                    System.arraycopy(data, startIdx, batch, 0, endIdx - startIdx);
                    
                    return batchProcessor.apply(batch);
                })
                .toArray(arrayConstructor);
    }
    
    /**
     * Interface funcional para processamento de chunks.
     */
    @FunctionalInterface
    public interface ChunkProcessor<T> {
        T processChunk(int startIndex, int endIndex);
    }
    
    /**
     * Interface funcional para computação de elementos de matriz.
     */
    @FunctionalInterface
    public interface MatrixElementFunction {
        double compute(int row, int col);
    }
    
    /**
     * Configuração personalizada de paralelização.
     */
    public static class ParallelConfig {
        private final int parallelism;
        private final int minThreshold;
        private final int chunkSize;
        
        public ParallelConfig(int parallelism, int minThreshold, int chunkSize) {
            this.parallelism = Math.max(1, parallelism);
            this.minThreshold = Math.max(1, minThreshold);
            this.chunkSize = Math.max(1, chunkSize);
        }
        
        public static ParallelConfig defaultConfig() {
            return new ParallelConfig(DEFAULT_PARALLELISM, MIN_PARALLEL_THRESHOLD, -1);
        }
        
        public static ParallelConfig forDataSize(int dataSize) {
            int parallelism = Math.min(DEFAULT_PARALLELISM, Math.max(1, dataSize / MIN_PARALLEL_THRESHOLD));
            int chunkSize = getOptimalChunkSize(dataSize, parallelism);
            return new ParallelConfig(parallelism, MIN_PARALLEL_THRESHOLD, chunkSize);
        }
        
        public int getParallelism() { return parallelism; }
        public int getMinThreshold() { return minThreshold; }
        public int getChunkSize() { return chunkSize; }
        
        public boolean shouldParallelize(int dataSize) {
            return dataSize >= minThreshold && parallelism > 1;
        }
    }
    
    /**
     * Worker especializado para operações de séries temporais.
     */
    public static class TimeSeriesWorker {
        private final ParallelConfig config;
        
        public TimeSeriesWorker(ParallelConfig config) {
            this.config = config;
        }
        
        public TimeSeriesWorker() {
            this(ParallelConfig.defaultConfig());
        }
        
        /**
         * Computa distâncias entre uma série temporal e um conjunto de centróides.
         */
        public double[] computeDistancesToCentroids(double[][] timeSeries, 
                                                   double[][][] centroids,
                                                   DistanceFunction distanceFunction) {
            
            int nCentroids = centroids.length;
            
            if (!config.shouldParallelize(nCentroids)) {
                double[] distances = new double[nCentroids];
                for (int i = 0; i < nCentroids; i++) {
                    distances[i] = distanceFunction.compute(timeSeries, centroids[i]);
                }
                return distances;
            }
            
            return IntStream.range(0, nCentroids)
                    .parallel()
                    .mapToDouble(i -> distanceFunction.compute(timeSeries, centroids[i]))
                    .toArray();
        }
        
        /**
         * Atribui amostras a clusters em paralelo.
         */
        public int[] assignSamplesToClusters(double[][][] samples, 
                                           double[][][] centroids,
                                           DistanceFunction distanceFunction) {
            
            int nSamples = samples.length;
            
            if (!config.shouldParallelize(nSamples)) {
                int[] assignments = new int[nSamples];
                for (int i = 0; i < nSamples; i++) {
                    double[] distances = computeDistancesToCentroids(samples[i], centroids, distanceFunction);
                    assignments[i] = parallelArgMin(distances);
                }
                return assignments;
            }
            
            return IntStream.range(0, nSamples)
                    .parallel()
                    .map(i -> {
                        double[] distances = computeDistancesToCentroids(samples[i], centroids, distanceFunction);
                        return parallelArgMin(distances);
                    })
                    .toArray();
        }
    }
    
    /**
     * Interface funcional para funções de distância.
     */
    @FunctionalInterface
    public interface DistanceFunction {
        double compute(double[][] ts1, double[][] ts2);
    }
}
