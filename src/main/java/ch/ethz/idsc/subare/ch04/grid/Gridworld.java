// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.util.Random;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.core.util.DeterministicStandardModel;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;
import ch.ethz.idsc.tensor.sca.Clip;

/** produces results on p.83: */
class Gridworld extends DeterministicStandardModel implements MonteCarloInterface, EpisodeSupplier {
  final int NX = 4;
  final int NY = 4;
  private static final Tensor TERMINATE1 = Tensors.vector(0, 0); // A
  private static final Tensor TERMINATE2 = Tensors.vector(3, 3); // A'
  private static final Clip CLIP = Clip.function(0, 3);
  Random random = new Random();
  // ---
  private final Tensor states = Flatten.of(Array.of(Tensors::vector, NX, NY), 1).unmodifiable();
  final Tensor actions = Tensors.matrix(new Number[][] { //
      { -1, 0 }, //
      { +1, 0 }, //
      { 0, -1 }, //
      { 0, +1 } //
  }).unmodifiable();
  final Index statesIndex;

  public Gridworld() {
    statesIndex = Index.build(states);
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return actions;
  }

  /**************************************************/
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor stateS) {
    if (state.equals(TERMINATE1))
      return ZeroScalar.get();
    if (state.equals(TERMINATE2))
      return ZeroScalar.get();
    return RealScalar.ONE.negate();
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (state.equals(TERMINATE1))
      return TERMINATE1;
    if (state.equals(TERMINATE2))
      return TERMINATE2;
    return state.add(action).map(CLIP);
  }

  /**************************************************/
  @Override
  public EpisodeInterface kickoff(PolicyInterface policyInterface) {
    Tensor start = TERMINATE1;
    while (isTerminal(start))
      start = states.get(random.nextInt(states.length()));
    return new MonteCarloEpisode(this, policyInterface, start);
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return state.equals(TERMINATE1) || state.equals(TERMINATE2);
  }
}
