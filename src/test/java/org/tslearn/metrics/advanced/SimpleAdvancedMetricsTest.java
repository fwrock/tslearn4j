package org.tslearn.metrics.advanced;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Testes unitários para as métricas avançadas LCSS, MSM e TWE.
 */
public class SimpleAdvancedMetricsTest {
    
    private double[] ts1;
    private double[] ts2;
    private double[] ts3;
    private double[][] multiTs1;
    private double[][] multiTs2;
    
    @BeforeEach
    void setUp() {
        // Séries temporais univariadas para teste
        ts1 = new double[]{1.0, 2.0, 3.0, 2.0, 1.0};
        ts2 = new double[]{1.1, 2.1, 3.1, 2.1, 1.1}; // Similar com ruído
        ts3 = new double[]{1.0, 3.0, 2.0}; // Diferente
        
        // Séries temporais multivariadas
        multiTs1 = new double[][]{{1.0, 0.5}, {2.0, 1.0}, {3.0, 1.5}};
        multiTs2 = new double[][]{{1.1, 0.6}, {2.1, 1.1}, {2.9, 1.4}};
    }
    
    // Testes LCSS
    
    @Test
    void testLCSSBasicFunctionality() {
        LCSS lcss = new LCSS();
        
        // Séries idênticas
        assertEquals(0.0, lcss.distance(ts1, ts1), 1e-6);
        
        // Séries diferentes
        double dist12 = lcss.distance(ts1, ts2);
        assertTrue(dist12 >= 0.0 && dist12 <= 1.0);
        
        double dist13 = lcss.distance(ts1, ts3);
        assertTrue(dist13 >= 0.0 && dist13 <= 1.0);
    }
    
    @Test
    void testLCSSConstructors() {
        LCSS lcss1 = new LCSS();
        assertEquals(0.1, lcss1.getEpsilon(), 1e-6);
        assertEquals(1, lcss1.getDelta());
        
        LCSS lcss2 = new LCSS(0.2);
        assertEquals(0.2, lcss2.getEpsilon(), 1e-6);
        
        LCSS lcss3 = new LCSS(0.3, 2, true);
        assertEquals(0.3, lcss3.getEpsilon(), 1e-6);
        assertEquals(2, lcss3.getDelta());
        assertTrue(lcss3.isVerbose());
    }
    
    @Test
    void testLCSSBuilder() {
        LCSS lcss = new LCSS.Builder()
            .epsilon(0.3)
            .delta(2)
            .verbose(true)
            .build();
        
        assertEquals(0.3, lcss.getEpsilon(), 1e-6);
        assertEquals(2, lcss.getDelta());
        assertTrue(lcss.isVerbose());
    }
    
    @Test
    void testLCSSMultivariate() {
        LCSS lcss = new LCSS(0.2, 1, false);
        
        double distance = lcss.distance(multiTs1, multiTs2);
        assertTrue(distance >= 0.0 && distance <= 1.0);
        
        // Séries idênticas
        assertEquals(0.0, lcss.distance(multiTs1, multiTs1), 1e-6);
    }
    
    // Testes MSM
    
    @Test
    void testMSMBasicFunctionality() {
        MSM msm = new MSM();
        
        // Séries idênticas
        assertEquals(0.0, msm.distance(ts1, ts1), 1e-6);
        
        // Séries diferentes
        double dist12 = msm.distance(ts1, ts2);
        assertTrue(dist12 >= 0.0);
        
        double dist13 = msm.distance(ts1, ts3);
        assertTrue(dist13 >= 0.0);
    }
    
    @Test
    void testMSMConstructors() {
        MSM msm1 = new MSM();
        assertEquals(1.0, msm1.getMoveCost(), 1e-6);
        assertEquals(1.0, msm1.getSplitMergeCost(), 1e-6);
        
        MSM msm2 = new MSM(0.5);
        assertEquals(0.5, msm2.getMoveCost(), 1e-6);
        
        MSM msm3 = new MSM(0.5, 1.5, true);
        assertEquals(0.5, msm3.getMoveCost(), 1e-6);
        assertEquals(1.5, msm3.getSplitMergeCost(), 1e-6);
        assertTrue(msm3.isVerbose());
    }
    
    @Test
    void testMSMBuilder() {
        MSM msm = new MSM.Builder()
            .moveCost(0.8)
            .splitMergeCost(1.2)
            .verbose(true)
            .build();
        
        assertEquals(0.8, msm.getMoveCost(), 1e-6);
        assertEquals(1.2, msm.getSplitMergeCost(), 1e-6);
        assertTrue(msm.isVerbose());
    }
    
    @Test
    void testMSMMultivariate() {
        MSM msm = new MSM();
        
        double distance = msm.distance(multiTs1, multiTs2);
        assertTrue(distance >= 0.0);
        
        // Séries idênticas
        assertEquals(0.0, msm.distance(multiTs1, multiTs1), 1e-6);
    }
    
    // Testes TWE
    
    @Test
    void testTWEBasicFunctionality() {
        TWE twe = new TWE();
        
        // Séries idênticas
        assertEquals(0.0, twe.distance(ts1, ts1), 1e-6);
        
        // Séries diferentes
        double dist12 = twe.distance(ts1, ts2);
        assertTrue(dist12 >= 0.0);
        
        double dist13 = twe.distance(ts1, ts3);
        assertTrue(dist13 >= 0.0);
    }
    
    @Test
    void testTWEConstructors() {
        TWE twe1 = new TWE();
        assertEquals(0.001, twe1.getNu(), 1e-6);
        assertEquals(1.0, twe1.getLambda(), 1e-6);
        
        TWE twe2 = new TWE(0.01);
        assertEquals(0.01, twe2.getNu(), 1e-6);
        
        TWE twe3 = new TWE(0.01, 2.0, true);
        assertEquals(0.01, twe3.getNu(), 1e-6);
        assertEquals(2.0, twe3.getLambda(), 1e-6);
        assertTrue(twe3.isVerbose());
    }
    
    @Test
    void testTWEBuilder() {
        TWE twe = new TWE.Builder()
            .nu(0.005)
            .lambda(1.5)
            .verbose(true)
            .build();
        
        assertEquals(0.005, twe.getNu(), 1e-6);
        assertEquals(1.5, twe.getLambda(), 1e-6);
        assertTrue(twe.isVerbose());
    }
    
    @Test
    void testTWEMultivariate() {
        TWE twe = new TWE();
        
        double distance = twe.distance(multiTs1, multiTs2);
        assertTrue(distance >= 0.0);
        
        // Séries idênticas
        assertEquals(0.0, twe.distance(multiTs1, multiTs1), 1e-6);
    }
    
    // Testes de propriedades das métricas
    
    @Test
    void testMetricProperties() {
        LCSS lcss = new LCSS();
        MSM msm = new MSM();
        TWE twe = new TWE();
        
        // Simetria: d(x,y) = d(y,x)
        assertEquals(lcss.distance(ts1, ts2), lcss.distance(ts2, ts1), 1e-6);
        assertEquals(msm.distance(ts1, ts2), msm.distance(ts2, ts1), 1e-6);
        assertEquals(twe.distance(ts1, ts2), twe.distance(ts2, ts1), 1e-6);
        
        // Identidade: d(x,x) = 0
        assertEquals(0.0, lcss.distance(ts1, ts1), 1e-6);
        assertEquals(0.0, msm.distance(ts1, ts1), 1e-6);
        assertEquals(0.0, twe.distance(ts1, ts1), 1e-6);
        
        // Não-negatividade: d(x,y) >= 0
        assertTrue(lcss.distance(ts1, ts2) >= 0);
        assertTrue(msm.distance(ts1, ts2) >= 0);
        assertTrue(twe.distance(ts1, ts2) >= 0);
    }
    
    @Test
    void testAutoConfiguration() {
        // Teste auto-configuração LCSS
        double autoEpsilon = LCSS.calculateAutoEpsilon(ts1, ts2);
        assertTrue(autoEpsilon > 0);
        
        int autoDelta = LCSS.calculateAutoDelta(ts1, ts2);
        assertTrue(autoDelta >= 1);
        
        // Teste auto-configuração MSM
        MSM autoMSM = MSM.createAutoConfigured(ts1, ts2);
        assertTrue(autoMSM.getMoveCost() > 0);
        assertTrue(autoMSM.getSplitMergeCost() > 0);
        
        // Teste auto-configuração TWE
        TWE autoTWE = TWE.createAutoConfigured(ts1, ts2);
        assertTrue(autoTWE.getNu() > 0);
        assertTrue(autoTWE.getLambda() > 0);
    }
    
    @Test
    void testDetailedResults() {
        // LCSS com detalhes
        LCSS lcss = new LCSS(0.2, 1, false);
        LCSS.LCSSResult lcssResult = lcss.distanceWithDetails(ts1, ts2);
        assertNotNull(lcssResult);
        assertTrue(lcssResult.getDistance() >= 0.0);
        assertTrue(lcssResult.getLcsLength() >= 0);
        
        // MSM com detalhes
        MSM msm = new MSM();
        MSM.MSMResult msmResult = msm.distanceWithDetails(ts1, ts2);
        assertNotNull(msmResult);
        assertTrue(msmResult.getDistance() >= 0.0);
        assertTrue(msmResult.getTotalOperations() >= 0);
        
        // TWE com detalhes
        TWE twe = new TWE();
        TWE.TWEResult tweResult = twe.distanceWithDetails(ts1, ts2);
        assertNotNull(tweResult);
        assertTrue(tweResult.getDistance() >= 0.0);
        assertTrue(tweResult.getEditComponent() >= 0.0);
        assertTrue(tweResult.getWarpComponent() >= 0.0);
    }
    
    @Test
    void testPerformance() {
        // Teste de performance com séries maiores
        double[] largeSeries1 = new double[50];
        double[] largeSeries2 = new double[50];
        
        for (int i = 0; i < 50; i++) {
            largeSeries1[i] = Math.sin(i * 0.1);
            largeSeries2[i] = Math.sin(i * 0.1 + 0.1);
        }
        
        LCSS lcss = new LCSS();
        MSM msm = new MSM();
        TWE twe = new TWE();
        
        // Todas as métricas devem completar sem erro
        assertDoesNotThrow(() -> lcss.distance(largeSeries1, largeSeries2));
        assertDoesNotThrow(() -> msm.distance(largeSeries1, largeSeries2));
        assertDoesNotThrow(() -> twe.distance(largeSeries1, largeSeries2));
    }
}
