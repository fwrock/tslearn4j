package org.tslearn.utils;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.exception.OutOfRangeException;

/** Commons Math matrix API backed by one primitive buffer instead of one array per time tick. */
public final class TimeSeriesMatrix extends AbstractRealMatrix {
    private final int rows, columns;
    private final double[] values;
    public TimeSeriesMatrix(int rows, int columns) {
        super(rows, columns);
        this.rows=rows; this.columns=columns;
        this.values=new double[Math.multiplyExact(rows,columns)];
    }
    public TimeSeriesMatrix(double[] column) {
        this(column.length,1);
        System.arraycopy(column,0,values,0,column.length);
    }
    private TimeSeriesMatrix(double[] column, boolean copy) {
        super(column.length,1);this.rows=column.length;this.columns=1;
        this.values=copy?column.clone():column;
    }
    /** Caller owns a stable buffer; the matrix must not be mutated during fitting. */
    public static TimeSeriesMatrix wrapColumn(double[] column) { return new TimeSeriesMatrix(column,false); }
    @Override public int getRowDimension() { return rows; }
    @Override public int getColumnDimension() { return columns; }
    private int index(int row,int column) {
        if(row<0 || row>=rows) throw new OutOfRangeException(row,0,rows-1);
        if(column<0 || column>=columns) throw new OutOfRangeException(column,0,columns-1);
        return row*columns+column;
    }
    @Override public double getEntry(int row,int column) { return values[index(row,column)]; }
    @Override public void setEntry(int row,int column,double value) { values[index(row,column)]=value; }
    @Override public RealMatrix createMatrix(int rows,int columns) { return new TimeSeriesMatrix(rows,columns); }
    @Override public RealMatrix copy() {
        TimeSeriesMatrix result=new TimeSeriesMatrix(rows,columns);
        System.arraycopy(values,0,result.values,0,values.length);
        return result;
    }
}
