package org.tslearn.barycenters;

import org.apache.commons.math3.util.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.metrics.DTW;
import org.tslearn.utils.ArrayUtils;

/**
 * Implementação do DBA (DTW Barycenter Averaging) para calcular
 * baricentros de séries temporais usando Dynamic Time Warping.
 * 
 * Baseado no algoritmo de Petitjean et al. (2011).
 */
public class DTWBarycenter {
    
    private static final Logger logger = LoggerFactory.getLogger(DTWBarycenter.class);
    
    /**
     * Calcula o baricentro DTW de um conjunto de séries temporais.
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
        
        return computeDBA(timeSeries, initialization, maxIter, 1e-6);
    }
    
    /**
     * Calcula o baricentro DTW com inicialização automática.
     */
    public static double[][] compute(double[][][] timeSeries, int maxIter) {
        double[][] initialization = EuclideanBarycenter.compute(timeSeries);
        return compute(timeSeries, initialization, maxIter);
    }
    
    /**
     * Implementação principal do algoritmo DBA.
     */
    private static double[][] computeDBA(double[][][] timeSeries, double[][] initialization, 
                                        int maxIter, double tolerance) {
        
        int nSamples = timeSeries.length;
        int barycenterLength = initialization.length;
        int nFeatures = initialization[0].length;
        
        // Copiar inicialização
        double[][] barycenter = ArrayUtils.deepCopy2D(initialization);
        double[][] newBarycenter = new double[barycenterLength][nFeatures];
        
        DTW dtw = new DTW();
        double previousCost = Double.POSITIVE_INFINITY;
        
        for (int iter = 0; iter < maxIter; iter++) {
            // Resetar acumuladores
            double[][][] accumulator = new double[barycenterLength][nFeatures][1];
            int[][] counts = new int[barycenterLength][nFeatures];
            
            double totalCost = 0.0;
            
            // Para cada série temporal
            for (int s = 0; s < nSamples; s++) {
                double[][] ts = timeSeries[s];
                
                // Calcular caminho DTW ótimo
                DTW.DTWPathResult pathResult = dtw.distanceWithPath(barycenter, ts);
                java.util.List<int[]> path = pathResult.getPath();
                totalCost += pathResult.getDistance();
                
                // Acumular valores baseados no alinhamento
                for (int[] step : path) {
                    int barycenterIdx = step[0];
                    int timeSeriesIdx = step[1];
                    
                    for (int d = 0; d < nFeatures; d++) {
                        accumulator[barycenterIdx][d][0] += ts[timeSeriesIdx][d];
                        counts[barycenterIdx][d]++;
                    }
                }
            }
            
            // Calcular novo baricentro
            for (int t = 0; t < barycenterLength; t++) {
                for (int d = 0; d < nFeatures; d++) {
                    if (counts[t][d] > 0) {
                        newBarycenter[t][d] = accumulator[t][d][0] / counts[t][d];
                    } else {
                        // Manter valor anterior se não há alinhamento
                        newBarycenter[t][d] = barycenter[t][d];
                    }
                }
            }
            
            // Verificar convergência
            double avgCost = totalCost / nSamples;
            if (FastMath.abs(previousCost - avgCost) < tolerance) {
                logger.debug("DBA convergiu na iteração {} com custo {:.6f}", iter + 1, avgCost);
                break;
            }
            
            // Atualizar baricentro
            barycenter = ArrayUtils.deepCopy2D(newBarycenter);
            previousCost = avgCost;
        }
        
        return barycenter;
    }
    
    /**
     * Versão simplificada que usa apenas média para compatibilidade.
     */
    public static double[][] computeSimple(double[][][] timeSeries) {
        return EuclideanBarycenter.compute(timeSeries);
    }
    
    /**
     * Calcula múltiplos baricentros DTW em paralelo.
     */
    public static double[][][] computeMultiple(double[][][][] timeSeriesGroups, int maxIter) {
        int nGroups = timeSeriesGroups.length;
        double[][][] barycenters = new double[nGroups][][];
        
        for (int g = 0; g < nGroups; g++) {
            barycenters[g] = compute(timeSeriesGroups[g], maxIter);
        }
        
        return barycenters;
    }
    
    /**
     * Calcula o custo total de um baricentro em relação a um conjunto de séries.
     */
    public static double computeCost(double[][] barycenter, double[][][] timeSeries) {
        DTW dtw = new DTW();
        double totalCost = 0.0;
        
        for (double[][] ts : timeSeries) {
            totalCost += dtw.distance(barycenter, ts);
        }
        
        return totalCost / timeSeries.length;
    }
}
