// code by jph
package ch.ethz.idsc.subare.core.mc;

import java.util.ArrayList;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeQsaEstimator;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Multinomial;

/** simple every-visit Monte Carlo method suitable for nonstationary environments
 * 
 * (6.1) p.127 */
public class ConstantAlphaMonteCarloQsa implements EpisodeQsaEstimator {
  private final Scalar gamma;
  private final DiscreteQsa qsa;
  private Scalar alpha = null;

  /** @param discreteModel */
  public ConstantAlphaMonteCarloQsa(DiscreteModel discreteModel) {
    gamma = discreteModel.gamma();
    qsa = DiscreteQsa.build(discreteModel); // <- "arbitrary"
  }

  public void setAlpha(Scalar alpha) {
    this.alpha = alpha;
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
    // TODO more efficient update if gamma == 1
    int fromIndex = 0;
    for (StepInterface stepInterface : trajectory) {
      Tensor state = stepInterface.prevState();
      Tensor action = stepInterface.action();
      Scalar gain = Multinomial.horner(rewards.extract(fromIndex, rewards.length()), gamma);
      Scalar value0 = qsa.value(state, action);
      qsa.assign(state, action, value0.add( //
          gain.subtract(value0).multiply(alpha) // (6.1)
      ));
      ++fromIndex;
    }
  }

  @Override
  public DiscreteQsa qsa() {
    return qsa;
  }
}
