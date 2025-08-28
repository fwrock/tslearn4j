package org.tslearn.matrix_profile;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;

/**
 * Basic test for MatrixProfile.
 */
public class BasicMatrixProfileTest {
    
    @Test
    void testBasicMatrixProfile() {
        // Create simple test series
        double[] series = {1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4};
        
        MatrixProfile mp = new MatrixProfile.Builder(series, 5)
                .useFFT(true)
                .normalize(true)
                .build();
        
        MatrixProfile.MatrixProfileResult result = mp.stamp();
        
        assertNotNull(result);
        assertNotNull(result.matrixProfile);
        assertTrue(result.matrixProfile.length > 0);
        
        // Test motif discovery
        List<MatrixProfile.MotifResult> motifs = mp.findMotifs(1, result);
        assertNotNull(motifs);
        
        // Test discord discovery
        List<MatrixProfile.DiscordResult> discords = mp.findDiscords(1, result);
        assertNotNull(discords);
        
        System.out.println("Basic Matrix Profile test passed!");
        System.out.println("Matrix Profile length: " + result.matrixProfile.length);
        System.out.println("Found " + motifs.size() + " motifs");
        System.out.println("Found " + discords.size() + " discords");
    }
}
