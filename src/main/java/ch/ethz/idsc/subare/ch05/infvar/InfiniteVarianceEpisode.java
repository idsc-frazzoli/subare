// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import java.util.Random;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.ZeroScalar;

class InfiniteVarianceEpisode implements EpisodeInterface {
  final InfiniteVariance infiniteVariance;
  final PolicyInterface policyInterface; // TODO
  Tensor state = ZeroScalar.get();
  Random random = new Random();

  public InfiniteVarianceEpisode(InfiniteVariance infiniteVariance, PolicyInterface policyInterface) {
    this.infiniteVariance = infiniteVariance;
    this.policyInterface = policyInterface;
  }

  @Override
  public Tensor state() {
    return state;
  }

  @Override
  public StepInterface step() {
    Tensor actions = infiniteVariance.actions(state);
    int index = random.nextInt(actions.length());
    Tensor action = actions.get(index);
    Tensor stateS = infiniteVariance.move(state, action);
    Scalar reward = infiniteVariance.reward(state, action, stateS);
    state = stateS;
    return new StepInterface() {
      @Override
      public Tensor action() {
        return action;
      }

      @Override
      public Scalar reward() {
        return reward;
      }

      @Override
      public Tensor nextState() {
        return stateS;
      }
    };
  }

  @Override
  public boolean hasNext() {
    return state.equals(ZeroScalar.get());
  }
}
