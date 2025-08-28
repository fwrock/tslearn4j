package org.tslearn.utils;

/**
 * Exception thrown when a cluster becomes empty during clustering
 */
public class EmptyClusterException extends RuntimeException {
    
    public EmptyClusterException(String message) {
        super(message);
    }
    
    public EmptyClusterException(String message, Throwable cause) {
        super(message, cause);
    }
}
