// code by jph
package ch.ethz.idsc.subare.core.mc;

import java.util.ArrayList;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscountFunction;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeQsaEstimator;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** simple every-visit Monte Carlo method suitable for nonstationary environments
 * 
 * (6.1) p.127 */
public class ConstantAlphaMonteCarloQsa implements EpisodeQsaEstimator {
  private final DiscountFunction discountFunction;
  private final DiscreteQsa qsa;
  private final LearningRate learningRate;
  private final StateActionCounter sac;

  /** @param discreteModel
   * @param learningRate
   * @param sac */
  public ConstantAlphaMonteCarloQsa(DiscreteModel discreteModel, LearningRate learningRate, StateActionCounter sac) {
    discountFunction = DiscountFunction.of(discreteModel.gamma());
    qsa = DiscreteQsa.build(discreteModel); // <- "arbitrary"
    this.learningRate = learningRate;
    this.sac = sac;
  }

  @Override // from EpisodeDigest
  public void digest(EpisodeInterface episodeInterface) {
    Tensor rewards = Tensors.empty();
    List<StepInterface> trajectory = new ArrayList<>();
    while (episodeInterface.hasNext()) {
      StepInterface stepInterface = episodeInterface.step();
      rewards.append(stepInterface.reward());
      trajectory.add(stepInterface);
    }
    int fromIndex = 0;
    for (StepInterface stepInterface : trajectory) {
      Tensor state = stepInterface.prevState();
      Tensor action = stepInterface.action();
      Scalar gain = discountFunction.apply(rewards.extract(fromIndex, rewards.length()));
      Scalar value0 = qsa.value(state, action);
      Scalar alpha = learningRate.alpha(stepInterface, sac);
      Scalar delta = gain.subtract(value0).multiply(alpha);
      qsa.assign(state, action, value0.add(delta)); // (6.1)
      sac.digest(stepInterface);
      ++fromIndex;
    }
  }

  @Override // from DiscreteQsaSupplier
  public DiscreteQsa qsa() {
    return qsa;
  }
}
