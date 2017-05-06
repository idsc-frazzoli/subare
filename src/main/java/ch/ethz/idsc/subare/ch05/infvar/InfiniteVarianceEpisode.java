// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import java.util.Random;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Accumulate;
import ch.ethz.idsc.tensor.alg.Array;

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
  public StepInterface step() {
    final Tensor prev = state;
    Tensor actions = infiniteVariance.actions(state);
    Index actionIndex = Index.build(actions);
    Tensor prob = Array.zeros(actions.length());
    for (Tensor action : actions)
      prob.set(policyInterface.policy(state, action), actionIndex.of(action));
    prob = Accumulate.of(prob);
    // System.out.println(prob);
    double threshold = random.nextDouble();
    int index = 0;
    for (; index < prob.length(); ++index)
      if (Scalars.lessThan(DoubleScalar.of(threshold), prob.Get(index)))
        break;
    final Tensor action = actions.get(index);
    final Tensor stateS = infiniteVariance.move(state, action);
    final Scalar reward = infiniteVariance.reward(state, action, stateS);
    state = stateS;
    return new StepInterface() {
      @Override
      public Tensor prevState() {
        return prev;
      }

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
