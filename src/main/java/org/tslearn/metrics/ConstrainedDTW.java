package org.tslearn.metrics;

import java.util.Arrays;
import java.util.concurrent.CancellationException;

/** Exact dtaidistance-compatible squared-Euclidean DTW, psi=2, with band-only storage.
 * Retains cumulative path costs used by SimEDaPE's translation; this is not FastDTW. */
public final class ConstrainedDTW {
    private ConstrainedDTW() {}
    public record Path(int[] i,int[] j,double[] cost,boolean[] descending,double distance) {}
    private static final int PSI=2;
    private static final double INF=Double.POSITIVE_INFINITY;
    private static double get(double[][] rows,int[] first,int i,int j) {
        int col=j-first[i];return col<0 || col>=rows[i].length?INF:rows[i][col];
    }
    private static void set(double[][] rows,int[] first,int i,int j,double value) {
        int col=j-first[i];if(col>=0 && col<rows[i].length)rows[i][col]=value;
    }
    public static Path compute(double[] a,double[] b,double windowFraction) {
        if(a.length==0 || b.length==0 || !Double.isFinite(windowFraction) || windowFraction<0)
            throw new IllegalArgumentException("Nonempty DTW inputs and finite nonnegative window are required");
        int r=a.length,c=b.length;
        int requested=(int)Math.floor(c*windowFraction),window=requested==0?Math.max(r,c):Math.min(requested,Math.max(r,c));
        int effectiveWindow=Math.min(window,Math.max(r,c));
        int compactWidth=Math.min(c+1,Math.abs(r-c)+2*effectiveWindow+1);
        int overlapLeft=Math.min(effectiveWindow+Math.max(0,r-c),r+1);
        int right=effectiveWindow+Math.max(0,r-c)<=r?r+1-effectiveWindow-Math.max(0,r-c):0;
        int region2=Math.min(r,overlapLeft),region3=Math.min(r,Math.max(overlapLeft,right));
        double[][] rows=new double[r+1][];int[] first=new int[r+1];
        rows[0]=new double[Math.min(PSI,c)+1];
        for(int i=1;i<=r;i++) {
            if(Thread.currentThread().isInterrupted())throw new CancellationException("DTW interrupted");
            int min=Math.max(0,i-1-Math.max(0,r-c)-window+1);
            int end=Math.min(c,i-1+Math.max(0,c-r)+window);
            int rowIndex=i-1;
            first[i]=rowIndex<region2?0:rowIndex<region3?1+rowIndex-region2:region2==region3?0:region3-region2;
            rows[i]=new double[compactWidth];Arrays.fill(rows[i],INF);
            if(i<=PSI && i-1<region2)set(rows,first,i,0,0);
            for(int j=min;j<end;j++) {
                double delta=a[i-1]-b[j];
                set(rows,first,i,j+1,delta*delta+Math.min(get(rows,first,i-1,j),Math.min(get(rows,first,i-1,j+1),get(rows,first,i,j))));
            }
        }
        // dtaidistance's C backend chooses the nearest end on a tie before rooting costs.
        int endRow=r,endCol=c;
        int finalRi=r-1,finalOffset=finalRi<region2?0:finalRi<region3?1+finalRi-region2:region2==region3?0:region3-region2;
        int lastCompactColumn=c-finalOffset;
        double rowCost=get(rows,first,r,c);
        for(int q=r-1;q>=Math.max(0,r-PSI);q--) {
            int ri=q-1,offset=q==0?0:ri<region2?0:ri<region3?1+ri-region2:region2==region3?0:region3-region2;
            double candidate=get(rows,first,q,lastCompactColumn+offset);
            if(candidate<rowCost){rowCost=candidate;endRow=q;}
        }
        for(int q=c-1;q>=Math.max(0,c-PSI);q--)if(get(rows,first,r,q)<get(rows,first,r,endCol))endCol=q;
        double colCost=get(rows,first,r,endCol);
        double distance=Math.sqrt(Math.min(rowCost,colCost));
        for(double[] row:rows)for(int q=0;q<row.length;q++)if(row[q]!=INF)row[q]=Math.sqrt(row[q]);
        // Reproduce compact-backend suffix flags, including its final-row offset convention.
        // Python get_warping_path_fast then reconciles an unmarked terminal cell below.
        if(rowCost<colCost) {
            for(int q=endRow+1;q<=r;q++) {
                int ri=q-1,offset=ri<region2?0:ri<region3?1+ri-region2:region2==region3?0:region3-region2;
                set(rows,first,q,compactWidth-1+offset,-1);
            }
        } else {
            int ri=r-1,offset=ri<region2?0:ri<region3?1+ri-region2:region2==region3?0:region3-region2;
            int shift=effectiveWindow!=c?1:0;
            for(int q=compactWidth-(c-endCol);q<compactWidth;q++)set(rows,first,r,q-shift+offset,-1);
        }
        // The Python wrapper returns the compact buffer directly when its width equals c+1.
        // Preserve that view, including shifted rows where the C/D regions overlap a full-width buffer.
        if(compactWidth==c+1)for(int q=1;q<=r;q++) {
            int ri=q-1,offset=ri<region2?0:ri<region3?1+ri-region2:region2==region3?0:region3-region2;
            first[q]-=offset;
        }
        if(compactWidth<c+1)for(int q=1;q<=r;q++) {
            int end=Math.min(c,q-1+Math.max(0,c-r)+window);
            for(int col=Math.max(0,end+1-first[q]);col<rows[q].length;col++)rows[q][col]=INF;
        }
        if(get(rows,first,r,c)>=0) {
            int rowBest=Math.max(0,r-PSI),colBest=Math.max(0,c-PSI);
            for(int q=rowBest+1;q<=r;q++)if(get(rows,first,q,c)<get(rows,first,rowBest,c))rowBest=q;
            for(int q=colBest+1;q<=c;q++)if(get(rows,first,r,q)<get(rows,first,r,colBest))colBest=q;
            if(get(rows,first,rowBest,c)<get(rows,first,r,colBest)) {
                for(int q=rowBest+1;q<=r;q++)set(rows,first,q,c,-1);
            } else {
                for(int q=colBest+1;q<=c;q++)set(rows,first,r,q,-1);
            }
        }
        int[] ii=new int[Math.addExact(r,c)],jj=new int[ii.length];double[] cost=new double[ii.length];boolean[] descending=new boolean[ii.length];
        int i=r,j=c,n=0;
        while(i>0 && j>0) {
            double value=get(rows,first,i,j);
            if(value!=-1) {ii[n]=i-1;jj[n]=j-1;cost[n]=value;descending[n]=a[i-1]>b[j-1];n++;}
            double diag=get(rows,first,i-1,j-1),up=get(rows,first,i-1,j),left=get(rows,first,i,j-1),minimum=Math.min(diag,Math.min(up,left));
            if(minimum==diag){i--;j--;}else if(minimum==up)i--;else j--;
        }
        int[] outI=new int[n],outJ=new int[n];double[] outCost=new double[n];boolean[] outSign=new boolean[n];
        for(int q=0;q<n;q++){int from=n-q-1;outI[q]=ii[from];outJ[q]=jj[from];outCost[q]=cost[from];outSign[q]=descending[from];}
        return new Path(outI,outJ,outCost,outSign,distance);
    }
}
