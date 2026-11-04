package org.tslearn.decomposition;

import org.tslearn.utils.NativeLinearAlgebra;
import org.tslearn.preprocessing.TimeSeriesResampler;

/** Portable double-precision PCA; no framework objects or Python pickle required. */
public record PCAModel(double[] mean,double[][] components,double[] eigenvalues,double[] varianceRatio,int size) {
    public static PCAModel fit(double[][] data,int count) {
        if(data.length==0 || data[0].length==0 || count<1)throw new IllegalArgumentException("Invalid PCA dataset");
        int n=data.length,t=data[0].length,k=Math.min(count,Math.min(n,t));double[] mean=new double[t];
        for(double[] row:data) {if(row.length!=t)throw new IllegalArgumentException("Ragged PCA dataset");for(int j=0;j<t;j++)mean[j]+=row[j]/n;}
        double[] centred=new double[Math.multiplyExact(n,t)];
        for(int i=0;i<n;i++)for(int j=0;j<t;j++)centred[i*t+j]=data[i][j]-mean[j];
        double total=0;for(double value:centred)total+=value*value;
        var eig=NativeLinearAlgebra.rightSingularSystem(centred,n,t,k);double[] ratio=new double[k],values=new double[k];
        for(int j=0;j<k;j++) {values[j]=Math.max(0,eig.values()[j])/Math.max(1,n-1);ratio[j]=total>0?Math.max(0,eig.values()[j])/total:Double.NaN;}
        return new PCAModel(mean,eig.vectors(),values,ratio,n);
    }
    public double[] weights(double[] observed,boolean stretch,String interpolation) {
        double[] x=stretch?TimeSeriesResampler.resample(observed,mean.length,interpolation):observed;
        if(x.length>mean.length)throw new IllegalArgumentException("Observation exceeds trained horizon");
        double[] centred=new double[x.length];for(int i=0;i<x.length;i++)centred[i]=x[i]-mean[i];
        if(stretch) {
            double[] result=new double[components.length];
            for(int k=0;k<result.length;k++)for(int j=0;j<x.length;j++)result[k]+=components[k][j]*centred[j];
            return result;
        }
        double[][] design=new double[x.length][components.length];
        for(int j=0;j<x.length;j++)for(int k=0;k<components.length;k++)design[j][k]=components[k][j];
        return NativeLinearAlgebra.leastSquares(design,centred);
    }
    public double[] project(double[] x) {
        if(x.length!=mean.length)throw new IllegalArgumentException("Projection length mismatch");
        double[] out=new double[components.length];
        for(int k=0;k<out.length;k++)for(int j=0;j<x.length;j++)out[k]+=components[k][j]*(x[j]-mean[j]);
        return out;
    }
    public double[] filter(double[] weights) {
        double[] out=weights.clone();double noise=eigenvalues[eigenvalues.length-1];
        for(int i=0;i<out.length;i++)out[i]*=eigenvalues[i]/(eigenvalues[i]+noise);
        return out;
    }
    public double[] reconstruct(double[] weights,int length) {
        double[] out=new double[length];
        for(int j=0;j<length;j++) {
            out[j]=mean[Math.min(j,mean.length-1)];
            if(j<mean.length)for(int k=0;k<weights.length;k++)out[j]+=components[k][j]*weights[k];
        }
        return out;
    }
}
