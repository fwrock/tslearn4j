package org.tslearn.utils;

/** MT19937 plus legacy interval sampling, for NumPy RandomState permutations. */
public final class NumpyRandomState {
    private final int[] state=new int[624];private int index=624;
    public NumpyRandomState(long seed) {
        if(seed<0 || seed>0xffffffffL)throw new IllegalArgumentException("Seed must be in [0,2^32-1]");
        state[0]=(int)seed;
        for(int i=1;i<624;i++)state[i]=1812433253*(state[i-1]^(state[i-1]>>>30))+i;
    }
    public NumpyRandomState(int[] key) {
        this(19650218L);
        if(key.length==0)throw new IllegalArgumentException("Empty MT19937 seed");
        int i=1,j=0;
        for(int remaining=Math.max(624,key.length);remaining>0;remaining--) {
            state[i]=(state[i]^((state[i-1]^(state[i-1]>>>30))*1664525))+key[j]+j;
            i++;j++;
            if(i>=624) {state[0]=state[623];i=1;}
            if(j>=key.length)j=0;
        }
        for(int remaining=623;remaining>0;remaining--) {
            state[i]=(state[i]^((state[i-1]^(state[i-1]>>>30))*1566083941))-i;
            i++;if(i>=624){state[0]=state[623];i=1;}
        }
        state[0]=0x80000000;
    }
    public long bits(int count) {
        if(count<1 || count>32)throw new IllegalArgumentException("Bit count must be in [1,32]");
        return Integer.toUnsignedLong(next()) >>> (32-count);
    }
    private int next() {
        if(index==624) {
            for(int i=0;i<624;i++) {
                int bits=(state[i]&0x80000000)|(state[(i+1)%624]&0x7fffffff);
                state[i]=state[(i+397)%624]^(bits>>>1)^((bits&1)!=0?0x9908b0df:0);
            }
            index=0;
        }
        int value=state[index++];value^=value>>>11;value^=(value<<7)&0x9d2c5680;value^=(value<<15)&0xefc60000;value^=value>>>18;return value;
    }
    private int interval(int maximum) {
        int mask=maximum;mask|=mask>>>1;mask|=mask>>>2;mask|=mask>>>4;mask|=mask>>>8;mask|=mask>>>16;
        int value;do {value=next()&mask;} while(value>maximum);return value;
    }
    public int[] permutation(int size) {
        if(size<0)throw new IllegalArgumentException("Invalid population size");
        int[] out=new int[size];for(int i=0;i<size;i++)out[i]=i;
        for(int i=size-1;i>0;i--) {int j=interval(i),value=out[i];out[i]=out[j];out[j]=value;}
        return out;
    }
}
