package org.tslearn.utils;

import java.util.*;
import java.math.BigInteger;

/** CPython Random integer seeding, getrandbits sampling and shuffle conventions. */
public final class PythonRandom {
    private final NumpyRandomState random;
    public PythonRandom(long seed) {
        BigInteger value=BigInteger.valueOf(seed).abs();
        int[] words=value.bitLength()>32?new int[]{value.intValue(),value.shiftRight(32).intValue()}:new int[]{value.intValue()};
        random=new NumpyRandomState(words);
    }
    public int below(int n) {
        if(n<1)throw new IllegalArgumentException("Empty population");
        int bits=32-Integer.numberOfLeadingZeros(n);long value;
        do {value=random.bits(bits);} while(value>=n);return (int)value;
    }
    public <T> void shuffle(List<T> values) {for(int i=values.size()-1;i>0;i--)Collections.swap(values,i,below(i+1));}
    public <T> List<T> sample(List<T> population,int k) {
        int n=population.size();if(k<0 || k>n)throw new IllegalArgumentException("Invalid sample size");
        List<T> result=new ArrayList<>(k);int setsize=21;
        if(k>5)setsize+=(int)Math.pow(4,Math.ceil(Math.log(k*3.0)/Math.log(4)));
        if(n<=setsize) {
            List<T> pool=new ArrayList<>(population);
            for(int i=0;i<k;i++){int j=below(n-i);result.add(pool.get(j));pool.set(j,pool.get(n-i-1));}
        } else {
            Set<Integer> seen=new HashSet<>();
            for(int i=0;i<k;i++){int j=below(n);while(seen.contains(j))j=below(n);seen.add(j);result.add(population.get(j));}
        }
        return result;
    }
}
