package org.tslearn.metrics;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ConstrainedDTWTest {
    @Test void preservesPythonCBackendNarrowWindowAndRelaxationConventions() {
        double[] a={1,3,1,1,1,2,3,3,3,1},b={3,1,3,0,1,0,1,1,0,3};
        var path=ConstrainedDTW.compute(a,b,.1);
        assertEquals(5,path.distance(),1e-12);
        assertEquals(10,path.i().length);
        assertEquals(0,path.i()[0]);assertEquals(0,path.j()[0]);
        assertEquals(9,path.i()[9]);assertEquals(9,path.j()[9]);
        assertEquals(Math.sqrt(38),path.cost()[9],1e-12);
    }
    @Test void usesFullWindowWhenTheRequestedIntegerWindowIsZero() {
        var path=ConstrainedDTW.compute(new double[]{0,1,2,3},new double[]{0,1,2,3},.1);
        assertEquals(0,path.distance(),0);assertEquals(4,path.i().length);
        assertArrayEquals(new int[]{0,1,2,3},path.i());assertArrayEquals(path.i(),path.j());
    }
    @Test void rejectsInvalidInputsBeforeAllocatingTheMatrix() {
        assertThrows(IllegalArgumentException.class,()->ConstrainedDTW.compute(new double[0],new double[]{1},.1));
        assertThrows(IllegalArgumentException.class,()->ConstrainedDTW.compute(new double[]{1},new double[]{1},Double.NaN));
    }
}
