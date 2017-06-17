// code by jph
package ch.ethz.idsc.subare.core.mc;

import java.util.ArrayList;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscountFunction;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeVsEstimator;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** simple every-visit Monte Carlo method suitable for nonstationary environments
 * 
 * (6.1) p.127 */
public class ConstantAlphaMonteCarloVs implements EpisodeVsEstimator {
  private final DiscountFunction discountFunction;
  private final DiscreteVs vs;
  private final LearningRate learningRate;

  /** @param discreteModel */
  public ConstantAlphaMonteCarloVs(DiscreteModel discreteModel, LearningRate learningRate) {
    discountFunction = DiscountFunction.of(discreteModel.gamma());
    vs = DiscreteVs.build(discreteModel); // <- "arbitrary"
    this.learningRate = learningRate;
  }

  @Override
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
      Scalar gain = discountFunction.apply(rewards.extract(fromIndex, rewards.length()));
      Scalar value0 = vs.value(state);
      Scalar alpha = learningRate.alpha(stepInterface);
      Scalar delta = gain.subtract(value0).multiply(alpha);
      vs.assign(state, value0.add(delta)); // (6.1)
      learningRate.digest(stepInterface);
      ++fromIndex;
    }
  }

  @Override
  public DiscreteVs vs() {
    return vs;
  }
}
