package org.tslearn.barycenters;

import org.tslearn.utils.ArrayUtils;

/**
 * Implementação de baricentro euclidiano para séries temporais.
 * 
 * O baricentro euclidiano é simplesmente a média aritmética
 * ponto a ponto das séries temporais.
 */
public class EuclideanBarycenter {
    
    /**
     * Calcula o baricentro euclidiano de um conjunto de séries temporais.
     * 
     * @param timeSeries Array de séries temporais de formato [n_samples][time_length][n_features]
     * @return Baricentro de formato [time_length][n_features]
     */
    public static double[][] compute(double[][][] timeSeries) {
        if (timeSeries == null || timeSeries.length == 0) {
            throw new IllegalArgumentException("Conjunto de séries temporais não pode ser vazio");
        }
        
        return ArrayUtils.mean(timeSeries);
    }
    
    /**
     * Calcula o baricentro euclidiano com inicialização específica.
     * 
     * @param timeSeries Array de séries temporais
     * @param initialization Inicialização (ignorada no caso euclidiano)
     * @return Baricentro euclidiano
     */
    public static double[][] compute(double[][][] timeSeries, double[][] initialization) {
        // Para baricentro euclidiano, a inicialização é ignorada
        return compute(timeSeries);
    }
    
    /**
     * Calcula o baricentro euclidiano iterativamente.
     * 
     * @param timeSeries Array de séries temporais
     * @param initialization Inicialização (ignorada)
     * @param maxIter Número máximo de iterações (ignorado)
     * @return Baricentro euclidiano
     */
    public static double[][] compute(double[][][] timeSeries, double[][] initialization, int maxIter) {
        // Para baricentro euclidiano, não há iteração necessária
        return compute(timeSeries);
    }
}
