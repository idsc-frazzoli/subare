// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.IntegerQ;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

public interface DiscreteDistribution {
  /** @return lowest value a random variable from this distribution may attains
   * the value must be finite satisfy {@link IntegerQ} */
  Scalar lowerBound();

  /** @return highest value a random variable from this distribution may attains
   * the value may be {@link RealScalar#POSITIVE_INFINITY} */
  Scalar upperBound();

  /** @param n
   * @return P(X == n), i.e. probability of random variable X == n */
  Scalar probabilityEquals(int n);
}
