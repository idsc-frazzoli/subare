// code by fluric
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.sca.Clips;
import ch.ethz.idsc.tensor.sca.Sign;

/** using formula: epsilon = Max(minimum, maximum-(maximum-minimum)*N/decayInterval)
 * good values could be: minimum=0.01, maximum=0.5, decayInterval=1000, strongly depends on the problem */
public class LinearExplorationRate implements ExplorationRate {
  public static ExplorationRate of(Scalar decayInterval, Scalar maximum, Scalar minimum) {
    return new LinearExplorationRate(decayInterval, maximum, minimum);
  }

  public static ExplorationRate of(double decayInterval, double maximum, double minimum) {
    return of(RealScalar.of(decayInterval), RealScalar.of(maximum), RealScalar.of(minimum));
  }

  // ---
  private final Scalar decayInterval;
  private final Scalar minimum;
  private final Scalar maximum;

  private LinearExplorationRate(Scalar decayInterval, Scalar maximum, Scalar minimum) {
    this.decayInterval = Sign.requirePositive(decayInterval);
    this.minimum = Clips.unit().requireInside(minimum);
    this.maximum = Clips.unit().requireInside(maximum);
    Sign.requirePositiveOrZero(maximum.subtract(minimum));
  }

  @Override // from ExplorationRate
  public final Scalar epsilon(Tensor state, StateActionCounter sac) {
    Scalar decayedValue = maximum.subtract(maximum.subtract(minimum).multiply(sac.stateCount(state)).divide(decayInterval));
    return Max.of(minimum, decayedValue);
  }
}
