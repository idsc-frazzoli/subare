// code by fluric
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.pdf.BernoulliDistribution;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;
import ch.ethz.idsc.tensor.sca.Clip;

public class CoinFlip {
  /** @param successProbability
   * @return CoinFlip class with success (head) rate of successProbability */
  public static CoinFlip of(Scalar successProbability) {
    return new CoinFlip(successProbability);
  }

  private static final Scalar HEAD = RealScalar.ONE;
  private static final Scalar TAIL = RealScalar.ZERO;
  // ---
  private final Distribution coinFlip;

  private CoinFlip(Scalar successProbability) {
    Clip.unit().requireInside(successProbability);
    coinFlip = BernoulliDistribution.of(successProbability);
  }

  /** returns true if the coin toss ended up with head */
  public boolean tossHead() {
    return RandomVariate.of(coinFlip).equals(HEAD);
  }

  /** returns true if the coin toss ended up with tail */
  public boolean tossTail() {
    return RandomVariate.of(coinFlip).equals(TAIL);
  }

  /** gives the outcome of a coin toss, where HEAD=RealScalar.ONE and TAIL=RealScalar.ZERO */
  public Scalar toss() {
    return RandomVariate.of(coinFlip);
  }
}
