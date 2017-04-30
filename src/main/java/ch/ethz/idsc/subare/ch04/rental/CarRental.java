// code by jph
package ch.ethz.idsc.subare.ch04.rental;

import ch.ethz.idsc.subare.core.MoveInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.util.PoissonDistribution;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.sca.Clip;

/** states: number of cars at the 2 stations in the evening
 * actions: number of cars moved between the 2 stations during the night
 * the action is encoded as a 2-vector {+n,-n} */
class CarRental implements StandardModel, MoveInterface {
  private static final Clip CLIP = Clip.function(0, 20);
  // ---
  final Tensor states = Flatten.of(Array.of(Tensors::vector, 21, 21), 1).unmodifiable();
  PoissonDistribution p1_in = PoissonDistribution.of(RealScalar.of(3));
  PoissonDistribution p1out = PoissonDistribution.of(RealScalar.of(3));
  PoissonDistribution p2_in = PoissonDistribution.of(RealScalar.of(4));
  PoissonDistribution p2out = PoissonDistribution.of(RealScalar.of(2));

  public CarRental() {
    // TODO Auto-generated constructor stub
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    int min = state.Get(0).number().intValue();
    int max = state.Get(1).number().intValue();
    Tensor actions = Tensors.empty();
    for (Tensor shift : Range.of(-min, max + 1))
      actions.append(Tensors.of(shift, shift.negate()));
    return actions;
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    // TODO check if clip is necessary
    return state.add(action);
  }

  @Override
  public Scalar qsa(Tensor state, Tensor action, Tensor gvalues) {
    // TODO Auto-generated method stub
    return null;
  }
}
