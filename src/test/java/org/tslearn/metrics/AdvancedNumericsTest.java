package org.tslearn.metrics;
import org.junit.jupiter.api.Test;
import org.tslearn.utils.NativeLinearAlgebra;
import org.tslearn.preprocessing.TimeSeriesResampler;
import org.tslearn.decomposition.PCAModel;
import static org.junit.jupiter.api.Assertions.*;

class AdvancedNumericsTest {
    @Test void leastSquaresReturnsMinimumNormForUnderdeterminedAndRankDeficientInputs() {
        assertArrayEquals(new double[]{1,1},NativeLinearAlgebra.leastSquares(new double[][]{{1,1}},new double[]{2}),1e-12);
        assertArrayEquals(new double[]{1,1},NativeLinearAlgebra.leastSquares(new double[][]{{1,1},{1,1}},new double[]{2,2}),1e-12);
    }
    @Test void interpolationPreservesCubicPolynomialIncludingNotAKnotSpline() {
        double[] input={0,1,8,27};
        for(String method:new String[]{"poly","lagrange","newton","vandermonde","spline"}) {
            double[] actual=TimeSeriesResampler.resample(input,19,method);
            for(int i=0;i<actual.length;i++)assertEquals(27*Math.pow((double)i/18,3),actual[i],1e-8,method);
        }
    }
    @Test void pcaFitsRawCountsAndReconstructsFromPartialObservations() {
        var model=PCAModel.fit(new double[][]{{1,2,3,4},{2,3,4,5},{3,4,5,6}},1);
        assertArrayEquals(new double[]{1,2,3,4},model.reconstruct(model.weights(new double[]{1,2},false,"linear"),4),1e-10);
        assertEquals(4,model.eigenvalues()[0],1e-10);
        assertEquals(1,model.varianceRatio()[0],1e-10);
    }
    @Test void numpyRandomStatePermutationMatchesReferenceAndAdvancesState() {
        var random=new org.tslearn.utils.NumpyRandomState(10);
        assertArrayEquals(new int[]{2,5,0,3,4,1},random.permutation(6));
        assertArrayEquals(new int[]{2,0,5,1,4,3},random.permutation(6));
    }
    @Test void pythonRandomSamplingAndShuffleMatchCPython() {
        var random=new org.tslearn.utils.PythonRandom(42);
        assertEquals(java.util.List.of("f","a","e"),random.sample(java.util.List.of("a","b","c","d","e","f"),3));
        var values=new java.util.ArrayList<>(java.util.List.of("a","b","c","d","e","f"));
        random.shuffle(values);
        assertEquals(java.util.List.of("d","e","a","b","c","f"),values);
        assertEquals(java.util.List.of(654,114,25,759,281,250,228,142),new org.tslearn.utils.PythonRandom(42).sample(java.util.stream.IntStream.range(0,1000).boxed().toList(),8));
    }
    @Test void resamplingDistinguishesTslearnAndScipySinglePointConventions() {
        assertArrayEquals(new double[]{3},TimeSeriesResampler.resample(new double[]{1,3,5},1,"tslearn"),0);
        assertArrayEquals(new double[]{1},TimeSeriesResampler.resample(new double[]{1,3,5},1,"linear"),0);
        assertArrayEquals(new double[]{4,4,4},TimeSeriesResampler.resample(new double[]{4},3,"tslearn"),0);
        assertTrue(Double.isNaN(TimeSeriesResampler.resample(new double[]{4},3,"linear")[0]));
        assertThrows(IllegalArgumentException.class,()->TimeSeriesResampler.resample(new double[]{4},3,"spline"));
    }
    @Test void pcg64GeneratorMatchesNumpyDefaultRng() {
        var random=new org.tslearn.utils.NumpyGenerator(42);
        for(double expected:new double[]{.7739560485559633,.4388784397520523,.8585979199113825,.6973680290593639,.09417734788764953})assertEquals(expected,random.random(),0);
        random=new org.tslearn.utils.NumpyGenerator(42);
        for(int expected:new int[]{0,6,5,3,3,7,0,6,1,0})assertEquals(expected,random.integer(9));
        assertArrayEquals(new int[]{5,6,0,7,3,2,4,9,1,8},new org.tslearn.utils.NumpyGenerator(42).permutation(10));
    }
}
