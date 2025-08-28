package org.tslearn.metrics.advanced;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Testes unitários para as métricas avançadas LCSS, MSM e TWE.
 */
public class AdvancedMetricsTest {
    
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
    void testLCSSConstructors() {
        LCSS lcss1 = new LCSS();
        assertEquals(0.1, lcss1.getEpsilon(), 1e-6);
        assertEquals(1, lcss1.getDelta());
        assertFalse(lcss1.isVerbose());
        
        LCSS lcss2 = new LCSS(0.2);
        assertEquals(0.2, lcss2.getEpsilon(), 1e-6);
        assertEquals(1, lcss2.getDelta());
        
        LCSS lcss3 = new LCSS(0.3, 2, true);
        assertEquals(0.3, lcss3.getEpsilon(), 1e-6);
        assertEquals(2, lcss3.getDelta());
        assertTrue(lcss3.isVerbose());
    }
    
    @Test
    void testLCSSDistance() {
        LCSS lcss = new LCSS(0.2, 1, false);
        
        // Séries idênticas
        assertEquals(0.0, lcss.distance(ts1, ts1), 1e-6);
        
        // Séries similares
        double dist12 = lcss.distance(ts1, ts2);
        assertTrue(dist12 >= 0.0 && dist12 <= 1.0);
        assertTrue(dist12 < 0.5); // Deve ser relativamente pequena
        
        // Séries diferentes
        double dist13 = lcss.distance(ts1, ts3);
        assertTrue(dist13 > dist12); // Deve ser maior que a distância similar
        
        // Séries vazias
        assertEquals(1.0, lcss.distance(new double[0], ts1), 1e-6);
        assertEquals(0.0, lcss.distance(new double[0], new double[0]), 1e-6);
    }
    
    @Test
    void testLCSSMultivariate() {
        LCSS lcss = new LCSS(0.2, 1, false);
        
        double distance = lcss.distance(multiTs1, multiTs2);
        assertTrue(distance >= 0.0 && distance <= 1.0);
        
        // Séries idênticas
        assertEquals(0.0, lcss.distance(multiTs1, multiTs1), 1e-6);
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
    void testLCSSAutoConfiguration() {
        double autoEpsilon = LCSS.calculateAutoEpsilon(ts1, ts2);
        assertTrue(autoEpsilon > 0);
        
        int autoDelta = LCSS.calculateAutoDelta(ts1, ts2);
        assertTrue(autoDelta >= 1);
        
        LCSS autoLCSS = new LCSS.Builder()
            .autoEpsilon(ts1, ts2)
            .autoDelta(ts1, ts2)
            .build();
        
        assertTrue(autoLCSS.getEpsilon() > 0);
        assertTrue(autoLCSS.getDelta() >= 1);
    }
    
    @Test
    void testLCSSWithDetails() {
        LCSS lcss = new LCSS(0.2, 1, false);
        LCSS.LCSSResult result = lcss.distanceWithDetails(ts1, ts2);
        
        assertNotNull(result);
        assertTrue(result.getDistance() >= 0.0);
        assertTrue(result.getLcsLength() >= 0);
        assertTrue(result.getMinLength() > 0);
        assertTrue(result.getSimilarity() >= 0.0 && result.getSimilarity() <= 1.0);
        
        // Verificar consistência
        assertEquals(1.0 - result.getSimilarity(), result.getDistance(), 1e-6);
    }
    
    // Testes MSM
    
    @Test
    void testMSMConstructors() {
        MSM msm1 = new MSM();
        assertEquals(1.0, msm1.getMoveCost(), 1e-6);
        assertEquals(1.0, msm1.getSplitMergeCost(), 1e-6);
        assertFalse(msm1.isVerbose());
        
        MSM msm2 = new MSM(0.5);
        assertEquals(0.5, msm2.getMoveCost(), 1e-6);
        assertEquals(1.0, msm2.getSplitMergeCost(), 1e-6);
        
        MSM msm3 = new MSM(0.5, 1.5, true);
        assertEquals(0.5, msm3.getMoveCost(), 1e-6);
        assertEquals(1.5, msm3.getSplitMergeCost(), 1e-6);
        assertTrue(msm3.isVerbose());
    }
    
    @Test
    void testMSMDistance() {
        MSM msm = new MSM();
        
        // Séries idênticas
        assertEquals(0.0, msm.distance(ts1, ts1), 1e-6);
        
        // Séries diferentes
        double dist12 = msm.distance(ts1, ts2);
        assertTrue(dist12 >= 0.0);
        
        double dist13 = msm.distance(ts1, ts3);
        assertTrue(dist13 >= 0.0);
        
        // Séries vazias
        assertEquals(0.0, msm.distance(new double[0], new double[0]), 1e-6);
        assertTrue(msm.distance(new double[0], ts1) > 0);
        assertTrue(msm.distance(ts1, new double[0]) > 0);
    }
    
    @Test
    void testMSMMultivariate() {
        MSM msm = new MSM();
        
        double distance = msm.distance(multiTs1, multiTs2);
        assertTrue(distance >= 0.0);
        
        // Séries idênticas
        assertEquals(0.0, msm.distance(multiTs1, multiTs1), 1e-6);
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
    void testMSMAutoConfiguration() {
        MSM autoMSM = MSM.createAutoConfigured(ts1, ts2);
        assertTrue(autoMSM.getMoveCost() > 0);
        assertTrue(autoMSM.getSplitMergeCost() > 0);
    }
    
    @Test
    void testMSMWithDetails() {
        MSM msm = new MSM();
        MSM.MSMResult result = msm.distanceWithDetails(ts1, ts2);
        
        assertNotNull(result);
        assertTrue(result.getDistance() >= 0.0);
        assertTrue(result.getEstimatedMoves() >= 0);
        assertTrue(result.getEstimatedSplitsMerges() >= 0);
        assertTrue(result.getTotalOperations() > 0);
        assertTrue(result.getMoveRatio() >= 0.0 && result.getMoveRatio() <= 1.0);
        assertTrue(result.getSplitMergeRatio() >= 0.0 && result.getSplitMergeRatio() <= 1.0);
    }
    
    // Testes TWE
    
    @Test
    void testTWEConstructors() {
        TWE twe1 = new TWE();
        assertEquals(0.001, twe1.getNu(), 1e-6);
        assertEquals(1.0, twe1.getLambda(), 1e-6);
        assertFalse(twe1.isVerbose());
        
        TWE twe2 = new TWE(0.01);
        assertEquals(0.01, twe2.getNu(), 1e-6);
        assertEquals(1.0, twe2.getLambda(), 1e-6);
        
        TWE twe3 = new TWE(0.01, 2.0, true);
        assertEquals(0.01, twe3.getNu(), 1e-6);
        assertEquals(2.0, twe3.getLambda(), 1e-6);
        assertTrue(twe3.isVerbose());
    }
    
    @Test
    void testTWEDistance() {
        TWE twe = new TWE();
        
        // Séries idênticas
        assertEquals(0.0, twe.distance(ts1, ts1), 1e-6);
        
        // Séries diferentes
        double dist12 = twe.distance(ts1, ts2);
        assertTrue(dist12 >= 0.0);
        
        double dist13 = twe.distance(ts1, ts3);
        assertTrue(dist13 >= 0.0);
        
        // Séries vazias
        assertEquals(0.0, twe.distance(new double[0], new double[0]), 1e-6);
        assertTrue(twe.distance(new double[0], ts1) > 0);
        assertTrue(twe.distance(ts1, new double[0]) > 0);
    }
    
    @Test
    void testTWEMultivariate() {
        TWE twe = new TWE();
        
        double distance = twe.distance(multiTs1, multiTs2);
        assertTrue(distance >= 0.0);
        
        // Séries idênticas
        assertEquals(0.0, twe.distance(multiTs1, multiTs1), 1e-6);
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
    void testTWEAutoConfiguration() {
        TWE autoTWE = TWE.createAutoConfigured(ts1, ts2);
        assertTrue(autoTWE.getNu() > 0);
        assertTrue(autoTWE.getLambda() > 0);
    }
    
    @Test
    void testTWEWithDetails() {
        TWE twe = new TWE();
        TWE.TWEResult result = twe.distanceWithDetails(ts1, ts2);
        
        assertNotNull(result);
        assertTrue(result.getDistance() >= 0.0);
        assertTrue(result.getEditComponent() >= 0.0);
        assertTrue(result.getWarpComponent() >= 0.0);
        assertTrue(result.getLengthDifference() >= 0.0);
        assertTrue(result.getEditRatio() >= 0.0 && result.getEditRatio() <= 1.0);
        assertTrue(result.getWarpRatio() >= 0.0 && result.getWarpRatio() <= 1.0);
    }
    
    @Test
    void testTWEStiffness() {
        // Testar diferentes níveis de rigidez
        TWE flexibleTWE = new TWE(0.001, 0.1, false); // Pouco rígido
        TWE rigidTWE = new TWE(0.001, 2.0, false);     // Muito rígido
        
        double[] shifted1 = {1.0, 2.0, 3.0, 2.0, 1.0};
        double[] shifted2 = {0.0, 1.0, 2.0, 3.0, 2.0, 1.0}; // Deslocado
        
        double flexDist = flexibleTWE.distance(shifted1, shifted2);
        double rigidDist = rigidTWE.distance(shifted1, shifted2);
        
        // TWE rígida deve penalizar mais o warping
        assertTrue(rigidDist >= flexDist);
    }
    
    // Testes de erro
    
    @Test
    void testErrorHandling() {
        LCSS lcss = new LCSS();
        MSM msm = new MSM();
        TWE twe = new TWE();
        
        // Argumentos null
        Exception e1 = assertThrows(IllegalArgumentException.class, () -> lcss.distance(null, ts1));
        assertNotNull(e1);
        
        Exception e2 = assertThrows(IllegalArgumentException.class, () -> msm.distance(ts1, null));
        assertNotNull(e2);
        
        Exception e3 = assertThrows(IllegalArgumentException.class, () -> twe.distance((double[])null, (double[])null));
        assertNotNull(e3);
        
        // Parâmetros inválidos
        Exception e4 = assertThrows(IllegalArgumentException.class, () -> new LCSS(-0.1, 1, false));
        assertNotNull(e4);
        
        Exception e5 = assertThrows(IllegalArgumentException.class, () -> new MSM(-1.0, 1.0, false));
        assertNotNull(e5);
        
        Exception e6 = assertThrows(IllegalArgumentException.class, () -> new TWE(-0.001, 1.0, false));
        assertNotNull(e6);
        
        // Dimensões incompatíveis para multivariadas
        double[][] wrongDim = {{1.0}, {2.0}};
        Exception e7 = assertThrows(IllegalArgumentException.class, () -> lcss.distance(multiTs1, wrongDim));
        assertNotNull(e7);
        
        Exception e8 = assertThrows(IllegalArgumentException.class, () -> msm.distance(multiTs1, wrongDim));
        assertNotNull(e8);
        
        Exception e9 = assertThrows(IllegalArgumentException.class, () -> twe.distance(multiTs1, wrongDim));
        assertNotNull(e9);
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
    void testComparativePerformance() {
        // Criar séries maiores para teste de performance
        double[] largeSeries1 = new double[100];
        double[] largeSeries2 = new double[100];
        
        for (int i = 0; i < 100; i++) {
            largeSeries1[i] = Math.sin(i * 0.1);
            largeSeries2[i] = Math.sin(i * 0.1 + 0.1);
        }
        
        LCSS lcss = new LCSS();
        MSM msm = new MSM();
        TWE twe = new TWE();
        
        // Todas as métricas devem completar em tempo razoável
        long start = System.currentTimeMillis();
        lcss.distance(largeSeries1, largeSeries2);
        long lcssTime = System.currentTimeMillis() - start;
        
        start = System.currentTimeMillis();
        msm.distance(largeSeries1, largeSeries2);
        long msmTime = System.currentTimeMillis() - start;
        
        start = System.currentTimeMillis();
        twe.distance(largeSeries1, largeSeries2);
        long tweTime = System.currentTimeMillis() - start;
        
        // Verificar que todas executaram em tempo razoável (< 1 segundo)
        assertTrue(lcssTime < 1000);
        assertTrue(msmTime < 1000);
        assertTrue(tweTime < 1000);
    }
}
