// code by jph
package ch.ethz.idsc.subare.core.mc;

import java.util.List;

import ch.ethz.idsc.subare.core.DiscountFunction;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeVsEstimator;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** simple every-visit Monte Carlo method suitable for non-stationary environments
 * 
 * (6.1) p.127 */
public class ConstantAlphaMonteCarloVs extends ConstantAlphaMonteCarloBase implements EpisodeVsEstimator {
  public static EpisodeVsEstimator create(DiscreteModel discreteModel, LearningRate learningRate) {
    return new ConstantAlphaMonteCarloVs( //
        DiscountFunction.of(discreteModel.gamma()), //
        DiscreteVs.build(discreteModel.states()), //
        learningRate, new DiscreteStateActionCounter());
  }

  // ---
  private final VsInterface vs;

  private ConstantAlphaMonteCarloVs(DiscountFunction discountFunction, VsInterface vs, LearningRate learningRate, StateActionCounter sac) {
    super(discountFunction, learningRate, sac);
    this.vs = vs;
  }

  @Override
  protected void digest(Tensor rewards, List<StepInterface> trajectory) {
    int fromIndex = 0;
    for (StepInterface stepInterface : trajectory) {
      Tensor state = stepInterface.prevState();
      Scalar gain = discountFunction.apply(rewards.extract(fromIndex, rewards.length()));
      Scalar value0 = vs.value(state);
      Scalar alpha = learningRate.alpha(stepInterface, sac);
      Scalar delta = gain.subtract(value0).multiply(alpha);
      vs.increment(state, delta); // (6.1)
      sac.digest(stepInterface);
      ++fromIndex;
    }
  }

  @Override // from DiscreteVsSupplier
  public DiscreteVs vs() {
    return (DiscreteVs) vs;
  }
}
