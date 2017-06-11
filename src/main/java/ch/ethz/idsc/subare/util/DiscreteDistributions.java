// code by jph
package ch.ethz.idsc.subare.util;

import java.util.Random;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;

// PDF, probability density function
public enum DiscreteDistributions {
  ;
  // ---
  private static final Random RANDOM = new Random();

  public static Scalar nextSample(DiscreteDistribution discreteDistribution) {
    return nextSample(discreteDistribution, RANDOM);
  }

  public static Scalar nextSample(DiscreteDistribution discreteDistribution, Random random) {
    Scalar ref = RealScalar.of(random.nextDouble());
    int sample = discreteDistribution.lowerBound().number().intValue();
    Scalar cumprob = discreteDistribution.probabilityEquals(sample);
    while (Scalars.lessThan(cumprob, ref))
      cumprob = cumprob.add(discreteDistribution.probabilityEquals(++sample));
    return RealScalar.of(sample);
  }

  /** @param discreteDistribution
   * @param n
   * @return P(X < n) */
  public static Scalar probabilityLessThan(DiscreteDistribution discreteDistribution, int n) {
    final int lowerBound = discreteDistribution.lowerBound().number().intValue();
    Scalar cumprob = RealScalar.ZERO;
    for (int k = lowerBound; k < n; ++k)
      cumprob = cumprob.add(discreteDistribution.probabilityEquals(k));
    return cumprob;
  }
}
