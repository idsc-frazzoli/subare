// code by jph
package ch.ethz.idsc.subare.core.mc;

import java.util.ArrayList;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeQsaEstimator;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.util.FastHorner;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** simple every-visit Monte Carlo method suitable for nonstationary environments
 * 
 * (6.1) p.127 */
public class ConstantAlphaMonteCarloQsa implements EpisodeQsaEstimator {
  private final Scalar gamma; // because discrete model is not stored
  private final DiscreteQsa qsa;
  private final LearningRate learningRate;

  /** @param discreteModel */
  public ConstantAlphaMonteCarloQsa(DiscreteModel discreteModel, LearningRate learningRate) {
    gamma = discreteModel.gamma();
    qsa = DiscreteQsa.build(discreteModel); // <- "arbitrary"
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
      Tensor action = stepInterface.action();
      Scalar gain = FastHorner.of(rewards.extract(fromIndex, rewards.length()), gamma);
      Scalar value0 = qsa.value(state, action);
      Scalar alpha = learningRate.alpha(stepInterface);
      Scalar delta = gain.subtract(value0).multiply(alpha);
      qsa.assign(state, action, value0.add(delta)); // (6.1)
      learningRate.digest(stepInterface);
      ++fromIndex;
    }
  }

  @Override
  public DiscreteQsa qsa() {
    return qsa;
  }
}
