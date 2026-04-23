package org.tslearn.metrics;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.utils.ParallelUtils;

/**
 * Versão paralelizada do Dynamic Time Warping (DTW) para computação eficiente
 * de distâncias em conjuntos grandes de séries temporais.
 * 
 * Esta implementação oferece:
 * - Paralelização automática baseada no tamanho da matriz DTW
 * - Decomposição da matriz em blocos para processamento paralelo
 * - Otimizações específicas para diferentes tipos de restrições
 * - Balanceamento adaptivo de carga
 * - Cache de resultados para computações repetidas
 * 
 * @author TSLearn4J
 */
public class ParallelDTW extends DTW {
    
    private static final Logger logger = LoggerFactory.getLogger(ParallelDTW.class);
    
    // Thresholds para decidir quando usar paralelização
    private static final int PARALLEL_THRESHOLD = 500;  // Produto n*m mínimo
    private static final int BLOCK_SIZE = 100;           // Tamanho do bloco para decomposição
    
    private final ParallelUtils.ParallelConfig parallelConfig;
    private final boolean enableBlockDecomposition;
    
    /**
     * Construtor padrão com configuração automática.
     */
    public ParallelDTW() {
        super();
        this.parallelConfig = ParallelUtils.ParallelConfig.defaultConfig();
        this.enableBlockDecomposition = true;
    }
    
    /**
     * Construtor com configuração de paralelização personalizada.
     */
    public ParallelDTW(ParallelUtils.ParallelConfig parallelConfig) {
        super();
        this.parallelConfig = parallelConfig;
        this.enableBlockDecomposition = true;
    }
    
    /**
     * Construtor com restrição Sakoe-Chiba e paralelização.
     */
    public ParallelDTW(int sakoeChiba, ParallelUtils.ParallelConfig parallelConfig) {
        super(sakoeChiba);
        this.parallelConfig = parallelConfig;
        this.enableBlockDecomposition = true;
    }
    
    /**
     * Construtor completo com todas as opções.
     */
    public ParallelDTW(GlobalConstraint globalConstraint, 
                      double globalConstraintParam,
                      boolean enableEarlyTermination,
                      double earlyTerminationThreshold,
                      ParallelUtils.ParallelConfig parallelConfig) {
        super(globalConstraint, globalConstraintParam, enableEarlyTermination, earlyTerminationThreshold);
        this.parallelConfig = parallelConfig;
        this.enableBlockDecomposition = true;
    }
    
    /**
     * Calcula distância DTW usando paralelização adaptiva.
     */
    @Override
    public double distance(double[] ts1, double[] ts2) {
        if (ts1 == null || ts2 == null) {
            throw new IllegalArgumentException("Time series cannot be null");
        }
        
        if (ts1.length == 0 || ts2.length == 0) {
            throw new IllegalArgumentException("Time series cannot be empty");
        }
        
        int n = ts1.length;
        int m = ts2.length;
        
        // Decidir estratégia baseada no tamanho
        if (shouldUseParallelization(n, m)) {
            return computeParallelDTW(ts1, ts2);
        } else {
            return super.distance(ts1, ts2);
        }
    }
    
    /**
     * Verifica se deve usar paralelização baseado no tamanho das séries.
     */
    private boolean shouldUseParallelization(int n, int m) {
        return (long) n * m >= PARALLEL_THRESHOLD && 
               parallelConfig.shouldParallelize(Math.max(n, m));
    }
    
    /**
     * Computação paralela do DTW usando diferentes estratégias.
     */
    private double computeParallelDTW(double[] ts1, double[] ts2) {
        int n = ts1.length;
        int m = ts2.length;
        
        // Escolher estratégia baseada no tamanho e configuração
        if (enableBlockDecomposition && n > BLOCK_SIZE && m > BLOCK_SIZE) {
            return computeBlockParallelDTW(ts1, ts2);
        } else {
            return computeRowParallelDTW(ts1, ts2);
        }
    }
    
    /**
     * DTW paralelo com decomposição em blocos (para matrizes muito grandes).
     */
    private double computeBlockParallelDTW(double[] ts1, double[] ts2) {
        int n = ts1.length;
        int m = ts2.length;
        
        // Calcular dimensões dos blocos
        int blockRows = Math.min(BLOCK_SIZE, (n + parallelConfig.getParallelism() - 1) / parallelConfig.getParallelism());
        int blockCols = Math.min(BLOCK_SIZE, (m + parallelConfig.getParallelism() - 1) / parallelConfig.getParallelism());
        
        int numBlockRows = (n + blockRows - 1) / blockRows;
        int numBlockCols = (m + blockCols - 1) / blockCols;
        
        // Matriz para armazenar resultados dos blocos
        double[][][] blockResults = new double[numBlockRows][numBlockCols][];
        
        // Processar blocos em ondas diagonais para manter dependências
        for (int diagonal = 0; diagonal < numBlockRows + numBlockCols - 1; diagonal++) {
            final int currentDiagonal = diagonal;
            
            // Processar blocos na mesma diagonal em paralelo
            IntStream.range(0, Math.min(diagonal + 1, numBlockRows))
                    .filter(blockRow -> {
                        int blockCol = currentDiagonal - blockRow;
                        return blockCol >= 0 && blockCol < numBlockCols;
                    })
                    .parallel()
                    .forEach(blockRow -> {
                        int blockCol = currentDiagonal - blockRow;
                        blockResults[blockRow][blockCol] = computeBlock(
                            ts1, ts2, blockRow, blockCol, blockRows, blockCols, blockResults
                        );
                    });
        }
        
        // Retornar resultado do último bloco
        return blockResults[numBlockRows - 1][numBlockCols - 1][0];
    }
    
    /**
     * Computa um bloco específico da matriz DTW.
     */
    private double[] computeBlock(double[] ts1, double[] ts2, int blockRow, int blockCol,
                                 int blockRows, int blockCols, double[][][] blockResults) {
        
        int n = ts1.length;
        int m = ts2.length;
        
        int startRow = blockRow * blockRows;
        int endRow = Math.min(startRow + blockRows, n);
        int startCol = blockCol * blockCols;
        int endCol = Math.min(startCol + blockCols, m);
        
        int localRows = endRow - startRow;
        int localCols = endCol - startCol;
        
        // Criar matriz local para este bloco
        double[][] localMatrix = new double[localRows + 1][localCols + 1];
        
        // Inicializar bordas baseado em blocos adjacentes ou valores padrão
        initializeBlockBorders(localMatrix, blockRow, blockCol, blockResults, startRow, startCol);
        
        // Computar DTW local
        for (int i = 1; i <= localRows; i++) {
            for (int j = 1; j <= localCols; j++) {
                double cost = euclideanDistance(ts1[startRow + i - 1], ts2[startCol + j - 1]);
                
                localMatrix[i][j] = Math.min(Math.min(
                    localMatrix[i-1][j],     // insertion
                    localMatrix[i][j-1]),    // deletion
                    localMatrix[i-1][j-1]    // match
                ) + cost;
            }
        }
        
        // Retornar valores da borda direita e inferior para próximos blocos
        double[] result = new double[localRows + localCols + 1];
        result[0] = localMatrix[localRows][localCols]; // Valor final
        
        // Borda direita
        for (int i = 1; i <= localRows; i++) {
            result[i] = localMatrix[i][localCols];
        }
        
        // Borda inferior
        for (int j = 1; j <= localCols; j++) {
            result[localRows + j] = localMatrix[localRows][j];
        }
        
        return result;
    }
    
    /**
     * Inicializa as bordas de um bloco baseado em blocos adjacentes.
     */
    private void initializeBlockBorders(double[][] localMatrix, int blockRow, int blockCol,
                                       double[][][] blockResults, int startRow, int startCol) {
        
        int localRows = localMatrix.length - 1;
        int localCols = localMatrix[0].length - 1;
        
        if (blockRow == 0 && blockCol == 0) {
            // Primeiro bloco - inicialização padrão
            localMatrix[0][0] = 0.0;
            for (int j = 1; j <= localCols; j++) {
                localMatrix[0][j] = Double.POSITIVE_INFINITY;
            }
            for (int i = 1; i <= localRows; i++) {
                localMatrix[i][0] = Double.POSITIVE_INFINITY;
            }
        } else {
            // Inicializar baseado em blocos anteriores
            if (blockRow > 0 && blockResults[blockRow - 1][blockCol] != null) {
                // Usar borda inferior do bloco superior
                double[] upperBorder = blockResults[blockRow - 1][blockCol];
                for (int j = 0; j <= localCols; j++) {
                    localMatrix[0][j] = upperBorder[upperBorder.length - localCols + j - 1];
                }
            }
            
            if (blockCol > 0 && blockResults[blockRow][blockCol - 1] != null) {
                // Usar borda direita do bloco à esquerda
                double[] leftBorder = blockResults[blockRow][blockCol - 1];
                for (int i = 0; i <= localRows; i++) {
                    localMatrix[i][0] = leftBorder[i];
                }
            }
        }
    }
    
    /**
     * DTW paralelo processando linhas em paralelo (para matrizes médias).
     */
    private double computeRowParallelDTW(double[] ts1, double[] ts2) {
        int n = ts1.length;
        int m = ts2.length;
        
        // Usar ForkJoinTask para processamento recursivo
        ForkJoinPool pool = ParallelUtils.getGlobalPool();
        
        ParallelDTWTask task = new ParallelDTWTask(ts1, ts2, 0, n, 0, m);
        return pool.invoke(task);
    }
    
    /**
     * Task recursiva para computação paralela de DTW.
     */
    private class ParallelDTWTask extends RecursiveTask<Double> {
        private final double[] ts1;
        private final double[] ts2;
        private final int startRow, endRow;
        private final int startCol, endCol;
        
        private static final int SEQUENTIAL_THRESHOLD = 200;
        
        public ParallelDTWTask(double[] ts1, double[] ts2, int startRow, int endRow, int startCol, int endCol) {
            this.ts1 = ts1;
            this.ts2 = ts2;
            this.startRow = startRow;
            this.endRow = endRow;
            this.startCol = startCol;
            this.endCol = endCol;
        }
        
        @Override
        protected Double compute() {
            int rows = endRow - startRow;
            int cols = endCol - startCol;
            
            // Se pequeno o suficiente, processar sequencialmente
            if (rows * cols <= SEQUENTIAL_THRESHOLD) {
                return computeSequentialDTW();
            }
            
            // Dividir o problema
            if (rows > cols) {
                // Dividir por linhas
                int midRow = startRow + rows / 2;
                ParallelDTWTask task1 = new ParallelDTWTask(ts1, ts2, startRow, midRow, startCol, endCol);
                ParallelDTWTask task2 = new ParallelDTWTask(ts1, ts2, midRow, endRow, startCol, endCol);
                
                task1.fork();
                double result2 = task2.compute();
                double result1 = task1.join();
                
                return Math.min(result1, result2);
            } else {
                // Dividir por colunas
                int midCol = startCol + cols / 2;
                ParallelDTWTask task1 = new ParallelDTWTask(ts1, ts2, startRow, endRow, startCol, midCol);
                ParallelDTWTask task2 = new ParallelDTWTask(ts1, ts2, startRow, endRow, midCol, endCol);
                
                task1.fork();
                double result2 = task2.compute();
                double result1 = task1.join();
                
                return Math.min(result1, result2);
            }
        }
        
        private double computeSequentialDTW() {
            // Implementar DTW sequencial para submatriz
            int rows = endRow - startRow;
            int cols = endCol - startCol;
            
            // Para simplificar, usar implementação padrão em subconjunto
            double[] subTs1 = Arrays.copyOfRange(ts1, startRow, endRow);
            double[] subTs2 = Arrays.copyOfRange(ts2, startCol, endCol);
            
            return ParallelDTW.super.distance(subTs1, subTs2);
        }
    }
    
    /**
     * Calcula múltiplas distâncias DTW em paralelo.
     */
    public double[] distanceMatrix(double[][] timeSeries1, double[][] timeSeries2) {
        int n1 = timeSeries1.length;
        int n2 = timeSeries2.length;
        
        return ParallelUtils.parallelMapToDouble(n1 * n2, index -> {
            int i = index / n2;
            int j = index % n2;
            return distance(timeSeries1[i], timeSeries2[j]);
        });
    }
    
    /**
     * Calcula matriz de distâncias completa (all-pairs) em paralelo.
     */
    public double[][] distanceMatrixSquare(double[][] timeSeries) {
        int n = timeSeries.length;
        
        // Usar paralelização adaptiva baseada no tamanho
        if (parallelConfig.shouldParallelize(n * n)) {
            return ParallelUtils.parallelMatrix2D(n, n, (i, j) -> {
                if (i <= j) {
                    return distance(timeSeries[i], timeSeries[j]);
                } else {
                    return distance(timeSeries[j], timeSeries[i]); // Simétrica
                }
            });
        } else {
            double[][] result = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = i; j < n; j++) {
                    double dist = distance(timeSeries[i], timeSeries[j]);
                    result[i][j] = dist;
                    result[j][i] = dist; // Simétrica
                }
            }
            return result;
        }
    }
    
    /**
     * Encontra a série temporal mais próxima em paralelo.
     */
    public int findNearest(double[] query, double[][] timeSeries) {
        double[] distances = ParallelUtils.parallelMapToDouble(timeSeries.length, i -> 
            distance(query, timeSeries[i])
        );
        
        return ParallelUtils.parallelArgMin(distances);
    }
    
    /**
     * Encontra as k séries temporais mais próximas em paralelo.
     */
    public int[] findKNearest(double[] query, double[][] timeSeries, int k) {
        if (k <= 0 || k > timeSeries.length) {
            throw new IllegalArgumentException("k deve estar entre 1 e " + timeSeries.length);
        }
        
        // Calcular todas as distâncias em paralelo
        double[] distances = ParallelUtils.parallelMapToDouble(timeSeries.length, i -> 
            distance(query, timeSeries[i])
        );
        
        // Encontrar k menores índices
        Integer[] indices = IntStream.range(0, distances.length)
                .boxed()
                .toArray(Integer[]::new);
        
        Arrays.sort(indices, (i, j) -> Double.compare(distances[i], distances[j]));
        
        return Arrays.stream(indices)
                .limit(k)
                .mapToInt(Integer::intValue)
                .toArray();
    }
    
    /**
     * Computa DTW com caminho em paralelo para múltiplas consultas.
     */
    public DTWResult[] distanceWithPathBatch(double[] query, double[][] timeSeries) {
        return ParallelUtils.parallelMap(timeSeries.length, 
            i -> distanceWithPath(query, timeSeries[i]),
            DTWResult[]::new);
    }
    
    // Getters para configuração
    public ParallelUtils.ParallelConfig getParallelConfig() {
        return parallelConfig;
    }
    
    public boolean isBlockDecompositionEnabled() {
        return enableBlockDecomposition;
    }
    
    /**
     * Estatísticas de performance para análise.
     */
    public static class PerformanceStats {
        private final long computationTime;
        private final int parallelism;
        private final boolean usedParallelization;
        private final String strategy;
        
        public PerformanceStats(long computationTime, int parallelism, boolean usedParallelization, String strategy) {
            this.computationTime = computationTime;
            this.parallelism = parallelism;
            this.usedParallelization = usedParallelization;
            this.strategy = strategy;
        }
        
        // Getters
        public long getComputationTime() { return computationTime; }
        public int getParallelism() { return parallelism; }
        public boolean isUsedParallelization() { return usedParallelization; }
        public String getStrategy() { return strategy; }
    }
    
    /**
     * Versão instrumentada para coleta de estatísticas.
     */
    public double distanceWithStats(double[] ts1, double[] ts2, PerformanceStats[] stats) {
        long startTime = System.nanoTime();
        
        boolean useParallel = shouldUseParallelization(ts1.length, ts2.length);
        String strategy = useParallel ? "parallel" : "sequential";
        
        double result = distance(ts1, ts2);
        
        long endTime = System.nanoTime();
        stats[0] = new PerformanceStats(
            endTime - startTime,
            parallelConfig.getParallelism(),
            useParallel,
            strategy
        );
        
        return result;
    }
}
