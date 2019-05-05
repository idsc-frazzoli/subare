// code by jph
package ch.ethz.idsc.subare.core.mc;

import java.util.ArrayList;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscountFunction;
import ch.ethz.idsc.subare.core.EpisodeDigest;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

public abstract class ConstantAlphaMonteCarloBase implements EpisodeDigest {
  protected final DiscountFunction discountFunction;
  protected final LearningRate learningRate;
  protected final StateActionCounter sac;

  protected ConstantAlphaMonteCarloBase(DiscountFunction discountFunction, LearningRate learningRate, StateActionCounter sac) {
    this.discountFunction = discountFunction;
    this.learningRate = learningRate;
    this.sac = sac;
  }

  @Override // from EpisodeDigest
  public final void digest(EpisodeInterface episodeInterface) {
    Tensor rewards = Tensors.empty();
    List<StepInterface> trajectory = new ArrayList<>();
    while (episodeInterface.hasNext()) {
      StepInterface stepInterface = episodeInterface.step();
      rewards.append(stepInterface.reward());
      trajectory.add(stepInterface);
    }
    digest(rewards, trajectory);
  }

  protected abstract void digest(Tensor rewards, List<StepInterface> trajectory);
}
