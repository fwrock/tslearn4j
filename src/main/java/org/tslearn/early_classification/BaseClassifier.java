package org.tslearn.early_classification;

import java.util.Map;

/**
 * Interface base para classificadores usados em early classification.
 * 
 * Todos os classificadores devem implementar esta interface para serem
 * utilizados pelo EarlyClassifier.
 */
public interface BaseClassifier {
    
    /**
     * Treina o classificador com dados rotulados.
     * 
     * @param X Dados de treinamento [n_samples][time_length][n_features]
     * @param y Labels correspondentes [n_samples]
     */
    void fit(double[][][] X, String[] y);
    
    /**
     * Prediz a classe mais provável para uma série temporal.
     * 
     * @param timeSeries Série temporal [time_length][n_features]
     * @return Classe predita
     */
    String predict(double[][] timeSeries);
    
    /**
     * Retorna probabilidades para todas as classes.
     * 
     * @param timeSeries Série temporal [time_length][n_features]
     * @return Mapa classe -> probabilidade
     */
    Map<String, Double> predictProba(double[][] timeSeries);
    
    /**
     * Verifica se o classificador foi treinado.
     * 
     * @return true se treinado, false caso contrário
     */
    boolean isFitted();
    
    /**
     * Retorna as classes conhecidas pelo classificador.
     * 
     * @return Array de classes
     */
    String[] getClasses();
}
