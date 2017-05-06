// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.MoveInterface;
import ch.ethz.idsc.subare.core.RewardInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;

class InfiniteVariance implements StandardModel, RewardInterface, MoveInterface, EpisodeSupplier {
  final Tensor states = Tensors.vector(0, 1).unmodifiable();
  final Tensor actions = Tensors.vector(0, 1).unmodifiable(); // increment
  final Index statesIndex;

  public InfiniteVariance() {
    statesIndex = Index.build(states);
  }

  // List<Tensor> play() {
  // List<Tensor> list = new ArrayList<>();
  // // states.
  // // for ()
  // return null;
  // }
  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return state.equals(ZeroScalar.get()) ? actions : ZeroScalar.get();
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    return state.equals(ZeroScalar.get()) && action.equals(RealScalar.ONE) ? //
        RealScalar.ONE : ZeroScalar.get();
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    return state.add(action);
  }

  @Override
  public Scalar qsa(Tensor state, Tensor action, Tensor gvalues) {
    Tensor stateS = move(state, action);
    Scalar reward = reward(state, action, stateS);
    return reward.add(gvalues.Get(statesIndex.of(stateS)));
  }

  @Override
  public EpisodeInterface kickoff() {
    return new InfiniteVarianceEpisode(this, null);
  }
}
