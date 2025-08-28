package org.tslearn.examples;

import java.util.Arrays;
import java.util.Random;
import org.tslearn.metrics.advanced.LCSS;
import org.tslearn.metrics.advanced.MSM;
import org.tslearn.metrics.advanced.TWE;

/**
 * Exemplo demonstrativo das métricas avançadas: LCSS, MSM e TWE.
 * 
 * Este exemplo mostra o uso prático dessas métricas para diferentes
 * tipos de séries temporais e compara seus comportamentos.
 */
public class AdvancedMetricsExample {
    
    public static void main(String[] args) {
        System.out.println("=== Métricas Avançadas para Séries Temporais ===\n");
        
        // Demonstrar métricas individualmente
        demonstrateLCSS();
        demonstrateMSM();
        demonstrateTWE();
        
        // Comparação entre métricas
        compareMetrics();
        
        // Casos de uso específicos
        specificUseCases();
        
        System.out.println("\n=== Exemplo concluído ===");
    }
    
    /**
     * Demonstra o uso da métrica LCSS.
     */
    private static void demonstrateLCSS() {
        System.out.println("=== LCSS (Longest Common Subsequence) ===");
        
        // Séries temporais com padrões similares mas deslocados
        double[] ts1 = {1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0};
        double[] ts2 = {0.5, 1.5, 2.5, 3.5, 4.5, 3.5, 2.5, 1.5};
        double[] ts3 = {1.1, 2.1, 2.9, 4.1, 2.9, 2.1, 1.1}; // Similar com ruído
        
        System.out.println("Série 1: " + Arrays.toString(ts1));
        System.out.println("Série 2: " + Arrays.toString(ts2));
        System.out.println("Série 3: " + Arrays.toString(ts3));
        
        // LCSS com parâmetros padrão
        LCSS lcss = new LCSS();
        double dist12 = lcss.distance(ts1, ts2);
        double dist13 = lcss.distance(ts1, ts3);
        
        System.out.printf("LCSS(ts1, ts2) = %.4f\n", dist12);
        System.out.printf("LCSS(ts1, ts3) = %.4f\n", dist13);
        
        // LCSS com configuração automática
        LCSS autoLCSS = new LCSS.Builder()
            .autoEpsilon(ts1, ts2)
            .autoDelta(ts1, ts2)
            .verbose(true)
            .build();
        
        System.out.printf("Epsilon automático: %.4f\n", autoLCSS.getEpsilon());
        System.out.printf("Delta automático: %d\n", autoLCSS.getDelta());
        
        LCSS.LCSSResult result = autoLCSS.distanceWithDetails(ts1, ts2);
        System.out.println("Resultado detalhado: " + result);
        
        // Teste com séries multivariadas
        double[][] multiTs1 = {{1.0, 0.5}, {2.0, 1.0}, {3.0, 1.5}};
        double[][] multiTs2 = {{1.1, 0.6}, {2.1, 1.1}, {2.9, 1.4}};
        
        double multiDist = lcss.distance(multiTs1, multiTs2);
        System.out.printf("LCSS multivariada: %.4f\n", multiDist);
        
        System.out.println();
    }
    
    /**
     * Demonstra o uso da métrica MSM.
     */
    private static void demonstrateMSM() {
        System.out.println("=== MSM (Move-Split-Merge) ===");
        
        // Séries com diferentes resoluções
        double[] ts1 = {1.0, 3.0, 2.0, 4.0, 1.0};
        double[] ts2 = {1.0, 2.0, 3.0, 3.5, 2.0, 4.0, 1.0}; // Mais pontos
        double[] ts3 = {1.5, 3.5, 2.5}; // Menos pontos
        
        System.out.println("Série 1: " + Arrays.toString(ts1));
        System.out.println("Série 2: " + Arrays.toString(ts2));
        System.out.println("Série 3: " + Arrays.toString(ts3));
        
        // MSM com parâmetros padrão
        MSM msm = new MSM();
        double dist12 = msm.distance(ts1, ts2);
        double dist13 = msm.distance(ts1, ts3);
        
        System.out.printf("MSM(ts1, ts2) = %.4f\n", dist12);
        System.out.printf("MSM(ts1, ts3) = %.4f\n", dist13);
        
        // MSM com configuração personalizada
        MSM customMSM = new MSM.Builder()
            .moveCost(0.5)
            .splitMergeCost(1.0)
            .verbose(true)
            .build();
        
        MSM.MSMResult result = customMSM.distanceWithDetails(ts1, ts2);
        System.out.println("Resultado detalhado: " + result);
        
        // MSM com configuração automática
        MSM autoMSM = MSM.createAutoConfigured(ts1, ts2);
        System.out.printf("MSM auto-configurada: move_cost=%.4f, split_merge_cost=%.4f\n", 
                         autoMSM.getMoveCost(), autoMSM.getSplitMergeCost());
        
        double autoDist = autoMSM.distance(ts1, ts2);
        System.out.printf("Distância auto-configurada: %.4f\n", autoDist);
        
        System.out.println();
    }
    
    /**
     * Demonstra o uso da métrica TWE.
     */
    private static void demonstrateTWE() {
        System.out.println("=== TWE (Time Warp Edit) ===");
        
        // Séries com deslocamentos temporais e estruturais
        double[] ts1 = {0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0};
        double[] ts2 = {0.0, 0.5, 1.0, 2.0, 3.0, 2.5, 2.0, 1.0, 0.0}; // Mais longa
        double[] ts3 = {0.1, 1.1, 2.1, 3.1, 2.1}; // Mais curta com ruído
        
        System.out.println("Série 1: " + Arrays.toString(ts1));
        System.out.println("Série 2: " + Arrays.toString(ts2));
        System.out.println("Série 3: " + Arrays.toString(ts3));
        
        // TWE com parâmetros padrão
        TWE twe = new TWE();
        double dist12 = twe.distance(ts1, ts2);
        double dist13 = twe.distance(ts1, ts3);
        
        System.out.printf("TWE(ts1, ts2) = %.6f\n", dist12);
        System.out.printf("TWE(ts1, ts3) = %.6f\n", dist13);
        
        // TWE com diferentes configurações de stiffness
        TWE flexibleTWE = new TWE.Builder()
            .nu(0.001)
            .lambda(0.1)  // Menos rígido
            .verbose(true)
            .build();
        
        TWE rigidTWE = new TWE.Builder()
            .nu(0.001)
            .lambda(2.0)  // Mais rígido
            .build();
        
        double flexDist = flexibleTWE.distance(ts1, ts2);
        double rigidDist = rigidTWE.distance(ts1, ts2);
        
        System.out.printf("TWE flexível (λ=0.1): %.6f\n", flexDist);
        System.out.printf("TWE rígida (λ=2.0): %.6f\n", rigidDist);
        
        // Resultado detalhado
        TWE.TWEResult result = twe.distanceWithDetails(ts1, ts2);
        System.out.println("Resultado detalhado: " + result);
        
        // TWE auto-configurada
        TWE autoTWE = TWE.createAutoConfigured(ts1, ts2);
        System.out.printf("TWE auto-configurada: nu=%.6f, lambda=%.4f\n", 
                         autoTWE.getNu(), autoTWE.getLambda());
        
        System.out.println();
    }
    
    /**
     * Compara as três métricas em diferentes cenários.
     */
    private static void compareMetrics() {
        System.out.println("=== Comparação de Métricas ===");
        
        // Cenário 1: Séries similares
        double[] similar1 = {1.0, 2.0, 3.0, 2.0, 1.0};
        double[] similar2 = {1.1, 2.1, 3.1, 2.1, 1.1};
        
        // Cenário 2: Séries com outliers
        double[] clean = {1.0, 2.0, 3.0, 2.0, 1.0};
        double[] withOutliers = {1.0, 2.0, 10.0, 2.0, 1.0}; // Outlier no meio
        
        // Cenário 3: Séries com diferentes comprimentos
        double[] shortSeries = {1.0, 3.0, 1.0};
        double[] longSeries = {1.0, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1.0};
        
        LCSS lcss = new LCSS(0.2, 1, false);
        MSM msm = new MSM();
        TWE twe = new TWE();
        
        System.out.println("Cenário 1 - Séries similares:");
        System.out.printf("  LCSS: %.4f\n", lcss.distance(similar1, similar2));
        System.out.printf("  MSM:  %.4f\n", msm.distance(similar1, similar2));
        System.out.printf("  TWE:  %.6f\n", twe.distance(similar1, similar2));
        
        System.out.println("Cenário 2 - Série com outlier:");
        System.out.printf("  LCSS: %.4f\n", lcss.distance(clean, withOutliers));
        System.out.printf("  MSM:  %.4f\n", msm.distance(clean, withOutliers));
        System.out.printf("  TWE:  %.6f\n", twe.distance(clean, withOutliers));
        
        System.out.println("Cenário 3 - Comprimentos diferentes:");
        System.out.printf("  LCSS: %.4f\n", lcss.distance(shortSeries, longSeries));
        System.out.printf("  MSM:  %.4f\n", msm.distance(shortSeries, longSeries));
        System.out.printf("  TWE:  %.6f\n", twe.distance(shortSeries, longSeries));
        
        System.out.println();
    }
    
    /**
     * Demonstra casos de uso específicos para cada métrica.
     */
    private static void specificUseCases() {
        System.out.println("=== Casos de Uso Específicos ===");
        
        Random random = new Random(42);
        
        // Caso 1: LCSS para séries com ruído
        System.out.println("1. LCSS para robustez a ruído:");
        double[] signal = generateSineWave(50, 1.0, 0.0);
        double[] noisySignal = addNoise(signal, 0.2, random);
        
        LCSS robustLCSS = new LCSS.Builder()
            .autoEpsilon(signal, noisySignal, 0.3)
            .autoDelta(signal, noisySignal, 0.1)
            .build();
        
        double noisyDist = robustLCSS.distance(signal, noisySignal);
        System.out.printf("   Distância com ruído: %.4f\n", noisyDist);
        
        // Caso 2: MSM para séries com diferentes resoluções
        System.out.println("2. MSM para diferentes resoluções:");
        double[] highRes = generateSineWave(100, 1.0, 0.0);
        double[] lowRes = downsample(highRes, 3);
        
        MSM resolutionMSM = new MSM.Builder()
            .autoConfigured(highRes, lowRes)
            .build();
        
        double resDist = resolutionMSM.distance(highRes, lowRes);
        System.out.printf("   Distância multi-resolução: %.4f\n", resDist);
        
        // Caso 3: TWE para séries com deslocamentos complexos
        System.out.println("3. TWE para alinhamento complexo:");
        double[] pattern = {0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0};
        double[] shifted = shiftAndScale(pattern, 2, 1.1, random);
        
        TWE alignmentTWE = new TWE.Builder()
            .nu(0.01)
            .lambda(0.5)
            .build();
        
        double alignDist = alignmentTWE.distance(pattern, shifted);
        System.out.printf("   Distância com deslocamento: %.6f\n", alignDist);
        
        // Análise de performance
        performanceAnalysis();
        
        System.out.println();
    }
    
    /**
     * Análise de performance das métricas.
     */
    private static void performanceAnalysis() {
        System.out.println("4. Análise de Performance:");
        
        Random random = new Random(42);
        int[] lengths = {50, 100, 200};
        
        LCSS lcss = new LCSS();
        MSM msm = new MSM();
        TWE twe = new TWE();
        
        for (int length : lengths) {
            double[] ts1 = generateRandomSeries(length, random);
            double[] ts2 = generateRandomSeries(length, random);
            
            // LCSS
            long startTime = System.nanoTime();
            lcss.distance(ts1, ts2);
            long lcssTime = System.nanoTime() - startTime;
            
            // MSM
            startTime = System.nanoTime();
            msm.distance(ts1, ts2);
            long msmTime = System.nanoTime() - startTime;
            
            // TWE
            startTime = System.nanoTime();
            twe.distance(ts1, ts2);
            long tweTime = System.nanoTime() - startTime;
            
            System.out.printf("   Comprimento %d: LCSS=%.2fµs, MSM=%.2fµs, TWE=%.2fµs\n", 
                             length, lcssTime/1000.0, msmTime/1000.0, tweTime/1000.0);
        }
    }
    
    // Métodos utilitários
    
    private static double[] generateSineWave(int length, double amplitude, double phase) {
        double[] wave = new double[length];
        for (int i = 0; i < length; i++) {
            wave[i] = amplitude * Math.sin(2 * Math.PI * i / length + phase);
        }
        return wave;
    }
    
    private static double[] addNoise(double[] signal, double noiseLevel, Random random) {
        double[] noisy = new double[signal.length];
        for (int i = 0; i < signal.length; i++) {
            noisy[i] = signal[i] + random.nextGaussian() * noiseLevel;
        }
        return noisy;
    }
    
    private static double[] downsample(double[] signal, int factor) {
        int newLength = signal.length / factor;
        double[] downsampled = new double[newLength];
        for (int i = 0; i < newLength; i++) {
            downsampled[i] = signal[i * factor];
        }
        return downsampled;
    }
    
    private static double[] shiftAndScale(double[] signal, int shift, double scale, Random random) {
        double[] shifted = new double[signal.length + shift];
        
        // Adicionar valores aleatórios no início
        for (int i = 0; i < shift; i++) {
            shifted[i] = random.nextGaussian() * 0.1;
        }
        
        // Copiar e escalar sinal original
        for (int i = 0; i < signal.length; i++) {
            shifted[i + shift] = signal[i] * scale;
        }
        
        return shifted;
    }
    
    private static double[] generateRandomSeries(int length, Random random) {
        double[] series = new double[length];
        for (int i = 0; i < length; i++) {
            series[i] = random.nextGaussian();
        }
        return series;
    }
}
