// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.LinkedHashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.util.AverageTracker;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** tracks rewards and statistics of next states for a fixed (state, action) pair
 * 
 * utility class for {@link ActionValueStatistics} */
/* package */ class TransitionTracker implements StepDigest {
  private final AverageTracker average = new AverageTracker();
  private final Map<Tensor, Integer> map = new LinkedHashMap<>();
  private long total = 0;

  @Override
  public void digest(StepInterface stepInterface) {
    // it is imperative that state0 and action do not change per transition tracker, maybe check this!?
    // Tensor state0 = stepInterface.prevState();
    // Tensor action = stepInterface.action();
    Scalar reward = stepInterface.reward();
    Tensor next = stepInterface.nextState();
    // ---
    average.track(reward);
    map.merge(next, 1, Math::addExact);
    ++total;
  }

  public Scalar expectedReward() {
    return average.getScalar();
  }

  public Tensor transitions() {
    return Tensor.of(map.keySet().stream());
  }

  public Scalar transitionProbability(Tensor next) {
    return map.containsKey(next) //
        ? RationalScalar.of(map.get(next), total)
        : RealScalar.ZERO;
  }
}
