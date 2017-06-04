// code by jph
package ch.ethz.idsc.subare.core.mc;

import java.util.ArrayList;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeDigest;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Multinomial;

/** simple every-visit Monte Carlo method suitable for nonstationary environments
 * 
 * (6.1) p.127 */
// TODO test!
public class ConstantAlphaMonteCarlo implements EpisodeDigest {
  private final Scalar gamma;
  private final DiscreteVs vs;
  private Scalar alpha = null;

  /** @param discreteModel */
  public ConstantAlphaMonteCarlo(DiscreteModel discreteModel) {
    this.gamma = discreteModel.gamma();
    this.vs = DiscreteVs.build(discreteModel); // <- "arbitrary"
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
      Scalar gain = Multinomial.horner(rewards.extract(fromIndex, rewards.length()), gamma);
      Scalar value0 = vs.value(state);
      vs.assign(state, value0.add( //
          gain.subtract(value0).multiply(alpha) // (6.1)
      ));
      ++fromIndex;
    }
  }

  public DiscreteVs vs() {
    return vs;
  }
}
