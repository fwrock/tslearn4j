package org.tslearn.matrix_profile;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

/**
 * Basic test for MatrixProfile.
 */
public class BasicMatrixProfileTest {
    
    @Test
    void testBasicMatrixProfile() {
        // Create simple test series
        double[] series = {1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4};
        
        MatrixProfile mp = new MatrixProfile.Builder()
                .subsequenceLength(5)
                .normalize(true)
                .build();
        
        MatrixProfile.MatrixProfileResult result = mp.stamp(series);
        
        assertNotNull(result);
        assertNotNull(result.getMatrixProfile());
        assertTrue(result.getMatrixProfile().length > 0);
        
        // Test motif discovery
        MatrixProfile.MotifResult motifResult = mp.findMotifs(result, 1);
        assertNotNull(motifResult);
        List<MatrixProfile.MotifResult.MotifPair> motifs = motifResult.getMotifs();
        assertNotNull(motifs);
        
        // Test discord discovery
        MatrixProfile.DiscordResult discordResult = mp.findDiscords(result, 1);
        assertNotNull(discordResult);
        List<MatrixProfile.DiscordResult.Discord> discords = discordResult.getDiscords();
        assertNotNull(discords);
        
        System.out.println("Basic Matrix Profile test passed!");
        System.out.println("Matrix Profile length: " + result.getMatrixProfile().length);
        System.out.println("Found " + motifs.size() + " motifs");
        System.out.println("Found " + discords.size() + " discords");
    }
}
