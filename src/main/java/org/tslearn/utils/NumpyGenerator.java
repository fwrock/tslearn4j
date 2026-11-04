package org.tslearn.utils;

import java.util.Arrays;
import java.util.HashSet;

/** NumPy default_rng (PCG64), integer SeedSequence seeding and sampling conventions.
 * Integer arithmetic intentionally wraps at 32/64/128 bits. */
public final class NumpyGenerator {
    private long high,low,incHigh,incLow;
    private boolean cached;
    private long nextWord;
    private static int hash(int value,int[] constant) {
        value^=constant[0];constant[0]*=0x931e8875;value*=constant[0];return value^(value>>>16);
    }
    private static int mix(int a,int b) {int value=0xca01f9dd*a-0x4973f715*b;return value^(value>>>16);}
    public NumpyGenerator(long seed) {
        if(seed<0)throw new IllegalArgumentException("Seed must be nonnegative");
        int[] entropy=seed>>>32==0?new int[]{(int)seed}:new int[]{(int)seed,(int)(seed>>>32)};
        int[] pool=new int[4],constant={0x43b0d7e5};
        for(int i=0;i<4;i++)pool[i]=hash(i<entropy.length?entropy[i]:0,constant);
        for(int i=0;i<4;i++)for(int j=0;j<4;j++)if(i!=j)pool[j]=mix(pool[j],hash(pool[i],constant));
        int[] words=new int[8];int multiplier=0x8b51f9dd;
        for(int i=0;i<8;i++) {int value=pool[i%4]^multiplier;multiplier*=0x58f38ded;value*=multiplier;words[i]=value^(value>>>16);}
        long[] state=new long[4];for(int i=0;i<4;i++)state[i]=Integer.toUnsignedLong(words[i*2])|((long)words[i*2+1]<<32);
        incHigh=(state[2]<<1)|(state[3]>>>63);incLow=(state[3]<<1)|1;
        step();long prior=low;low+=state[1];high+=state[0]+(Long.compareUnsigned(low,prior)<0?1:0);step();
    }
    private void step() {
        long product=low*0x4385df649fccf645L;
        long upper=Math.unsignedMultiplyHigh(low,0x4385df649fccf645L)+high*0x4385df649fccf645L+low*0x2360ed051fc65da4L;
        low=product+incLow;high=upper+incHigh+(Long.compareUnsigned(low,product)<0?1:0);
    }
    public long nextLong() {step();return Long.rotateRight(high^low,(int)(high>>>58));}
    public double random() {return (nextLong()>>>11)*0x1.0p-53;}
    private long next32() {
        if(cached){cached=false;return nextWord;}
        long value=nextLong();cached=true;nextWord=value>>>32;return value&0xffffffffL;
    }
    /** Exclusive upper bound; Lemire mapping matches Generator.integers. */
    public int integer(int bound) {
        if(bound<1)throw new IllegalArgumentException("Bound must be positive");
        if(bound==1)return 0;
        long product=next32()*bound,least=product&0xffffffffL;
        if(least<bound){long threshold=(0x100000000L-bound)%bound;while(least<threshold){product=next32()*bound;least=product&0xffffffffL;}}
        return (int)(product>>>32);
    }
    private int interval(int maximum) {
        if(maximum==0)return 0;
        int mask=maximum;mask|=mask>>>1;mask|=mask>>>2;mask|=mask>>>4;mask|=mask>>>8;mask|=mask>>>16;
        int value;do{value=(int)next32()&mask;}while(value>maximum);return value;
    }
    public int[] permutation(int n) {
        int[] out=new int[n];for(int i=0;i<n;i++)out[i]=i;
        for(int i=n-1;i>0;i--){int j=interval(i),v=out[i];out[i]=out[j];out[j]=v;}
        return out;
    }
    private void shuffle(int[] out,int first) {for(int i=out.length-1;i>=first;i--){int j=integer(i+1),v=out[i];out[i]=out[j];out[j]=v;}}
    public int[] sample(int n,int k) {
        if(n<0 || k<0 || k>n)throw new IllegalArgumentException("Invalid sample size");
        if(n>10000 && k>n/50){int[] full=new int[n];for(int i=0;i<n;i++)full[i]=i;shuffle(full,Math.max(n-k,1));return Arrays.copyOfRange(full,n-k,n);}
        int[] out=new int[k];var seen=new HashSet<Integer>();
        for(int j=n-k;j<n;j++){int value=integer(j+1);if(seen.contains(value))value=j;seen.add(value);out[j-n+k]=value;}
        shuffle(out,1);return out;
    }
}
