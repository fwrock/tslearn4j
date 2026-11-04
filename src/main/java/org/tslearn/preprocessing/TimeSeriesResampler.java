package org.tslearn.preprocessing;

import org.tslearn.utils.NativeLinearAlgebra;

/** Uniform-grid interpolation methods used by SimEDaPE estimation variants. */
public final class TimeSeriesResampler {
    private TimeSeriesResampler() {}
    public static double[] resample(double[] input,int length,String method) {
        if(input.length==0 || length<1) throw new IllegalArgumentException("Nonempty series and positive length required");
        if(input.length==length) return input.clone();
        double[] out=new double[length];
        boolean tslearn=method.equals("tslearn");
        if(tslearn && length==1) {out[0]=java.util.Arrays.stream(input).average().orElseThrow();return out;}
        if(input.length==1) {
            if(!tslearn && method.equals("spline"))throw new IllegalArgumentException("Spline requires at least two points");
            java.util.Arrays.fill(out,!tslearn && method.equals("linear")?Double.NaN:input[0]);return out;
        }
        if(tslearn)method="linear";
        switch(method) {
            case "linear" -> {
                for(int j=0;j<length;j++) {
                    double pos=(double)j*(input.length-1)/Math.max(1,length-1);int lo=(int)pos,hi=Math.min(lo+1,input.length-1);
                    out[j]=input[lo]+(input[hi]-input[lo])*(pos-lo);
                }
            }
            case "poly" -> polynomial(input,length,Math.min(3,input.length-1)+1,false,out);
            case "vandermonde" -> polynomial(input,length,Math.min(12,input.length),true,out);
            case "newton", "lagrange" -> {
                int n=Math.min(method.equals("newton")?10:15,input.length);
                double[] x=new double[n],y=new double[n];
                for(int i=0;i<n;i++) {int ix=(int)Math.rint((double)i*(input.length-1)/(n-1));x[i]=(double)ix/(input.length-1);y[i]=input[ix];}
                if(method.equals("newton")) {
                    for(int k=1;k<n;k++) for(int i=n-1;i>=k;i--) y[i]=(y[i]-y[i-1])/(x[i]-x[i-k]);
                    for(int i=0;i<length;i++) {double t=(double)i/Math.max(1,length-1),v=y[n-1];for(int j=n-2;j>=0;j--)v=y[j]+(t-x[j])*v;out[i]=v;}
                } else {
                    double[] coefficients=new double[n];
                    for(int i=0;i<n;i++) {
                        double[] term={y[i]};
                        for(int j=0;j<n;j++) if(i!=j) {
                            double[] next=new double[term.length+1];
                            for(int k=0;k<term.length;k++) {next[k]-=x[j]*term[k]/(x[i]-x[j]);next[k+1]+=term[k]/(x[i]-x[j]);}
                            term=next;
                        }
                        for(int k=0;k<n;k++)coefficients[k]+=term[k];
                    }
                    evaluate(coefficients,out);
                }
            }
            case "spline" -> spline(input,out);
            default -> throw new IllegalArgumentException("Unknown interpolation: "+method);
        }
        return out;
    }
    private static void polynomial(double[] input,int length,int count,boolean subset,double[] out) {
        int rows=subset?count:input.length;
        double[][] design=new double[rows][count];double[] y=new double[rows];
        for(int i=0;i<rows;i++) {
            int ix=subset?(int)Math.rint((double)i*(input.length-1)/(rows-1)):i;
            double x=(double)ix/(input.length-1);y[i]=input[ix];design[i][0]=1;
            for(int j=1;j<count;j++)design[i][j]=design[i][j-1]*x;
        }
        evaluate(NativeLinearAlgebra.leastSquares(design,y),out);
    }
    private static void evaluate(double[] coefficients,double[] out) {
        for(int i=0;i<out.length;i++) {
            double x=(double)i/Math.max(1,out.length-1),v=0;
            for(int j=coefficients.length-1;j>=0;j--)v=v*x+coefficients[j];out[i]=v;
        }
    }
    /** SciPy CubicSpline default: not-a-knot, on the uniform input grid. */
    private static void spline(double[] y,double[] out) {
        int n=y.length;
        if(n<4) {
            if(n==2) {for(int i=0;i<out.length;i++)out[i]=y[0]+(y[1]-y[0])*i/Math.max(1,out.length-1);}
            else polynomial(y,out.length,3,false,out);
            return;
        }
        double h=1.0/(n-1);double[] m=new double[n];
        m[1]=(y[2]-2*y[1]+y[0])/(h*h);
        m[n-2]=(y[n-1]-2*y[n-2]+y[n-3])/(h*h);
        int size=n-4;
        if(size>0) {
            double[] upper=new double[size],rhs=new double[size];
            for(int k=0;k<size;k++) {
                int i=k+2;double value=6*(y[i+1]-2*y[i]+y[i-1])/(h*h);
                if(k==0)value-=m[1];if(k==size-1)value-=m[n-2];
                double diagonal=4-(k>0?upper[k-1]:0);
                upper[k]=1/diagonal;rhs[k]=(value-(k>0?rhs[k-1]:0))/diagonal;
            }
            for(int k=size-1;k>=0;k--)m[k+2]=rhs[k]-(k<size-1?upper[k]*m[k+3]:0);
        }
        m[0]=2*m[1]-m[2];m[n-1]=2*m[n-2]-m[n-3];
        for(int j=0;j<out.length;j++) {
            double pos=(double)j*(n-1)/Math.max(1,out.length-1);int i=Math.min((int)pos,n-2);
            double b=pos-i,a=1-b;
            out[j]=a*y[i]+b*y[i+1]+((a*a*a-a)*m[i]+(b*b*b-b)*m[i+1])*h*h/6;
        }
    }
}
