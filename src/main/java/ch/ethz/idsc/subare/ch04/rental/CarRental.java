// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.rental;

import ch.ethz.idsc.subare.core.MoveInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.subare.util.PoissonDistribution;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.red.Min;
import ch.ethz.idsc.tensor.sca.Clip;

/** Example 4.2: Jack's Car Rental
 * 
 * states: number of cars at the 2 stations in the evening
 * actions: number of cars moved between the 2 stations during the night
 * the action is encoded as a 2-vector {+n, -n} */
class CarRental implements StandardModel, MoveInterface {
  private static final int MAX_MOVE_OF_CARS = 5;
  private static final Scalar RENTAL_CREDIT = RealScalar.of(10);
  private static final Scalar MOVE_CAR_COST = RealScalar.of(2);
  private static final Clip CLIP = Clip.function(0, 20);
  // ---
  private final Tensor states = Flatten.of(Array.of(Tensors::vector, 21, 21), 1).unmodifiable();
  PoissonDistribution p1out = PoissonDistribution.of(RealScalar.of(3));
  PoissonDistribution p1_in = PoissonDistribution.of(RealScalar.of(3));
  PoissonDistribution p2out = PoissonDistribution.of(RealScalar.of(4));
  PoissonDistribution p2_in = PoissonDistribution.of(RealScalar.of(2));

  CarRental() {
    // TODO Auto-generated constructor stub
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    int min = Math.min(state.Get(0).number().intValue(), MAX_MOVE_OF_CARS);
    int max = Math.min(state.Get(1).number().intValue(), MAX_MOVE_OF_CARS);
    return Range.of(-min, max + 1);
  }

  @Override
  public Scalar gamma() {
    return RealScalar.of(.9);
  }

  /**************************************************/
  @Override
  public Tensor move(Tensor state, Tensor action) {
    Tensor next = state.add(Tensors.of(action, action.negate()));
    if (Scalars.lessThan(next.Get(0), RealScalar.ZERO))
      throw new RuntimeException();
    if (Scalars.lessThan(next.Get(1), RealScalar.ZERO))
      throw new RuntimeException();
    return next;
  }

  /**************************************************/
  @Override
  public Scalar qsa(Tensor state, Tensor action, VsInterface gvalues) {
    Scalar returns = RealScalar.ZERO;
    returns = returns.subtract(MOVE_CAR_COST.multiply(((Scalar) action).abs()));
    // go through all possible rental requests
    for (int rrL1 = 0; rrL1 <= 10; ++rrL1) {
      for (int rrL2 = 0; rrL2 <= 10; ++rrL2) {
        Scalar rrS1 = RealScalar.of(rrL1);
        Scalar rrS2 = RealScalar.of(rrL2);
        // moving cars
        Tensor numCars = move(state, action);
        Scalar mrS1 = Min.of(rrS1, numCars.Get(0));
        Scalar mrS2 = Min.of(rrS2, numCars.Get(1));
        Scalar reward = mrS1.add(mrS2).multiply(RENTAL_CREDIT);
        rrS1 = rrS1.subtract(mrS1);
        rrS2 = rrS2.subtract(mrS2);
        Scalar prob = p1out.apply(rrL1).multiply(p1out.apply(rrL2));
        {
          Scalar rcL1 = p1_in.lambda();
          Scalar rcL2 = p2_in.lambda();
          mrS1 = (Scalar) mrS1.add(rcL1).map(CLIP);
          mrS2 = (Scalar) mrS2.add(rcL2).map(CLIP);
          returns = returns.add(reward.add(gvalues.value(Tensors.of(mrS1, mrS2))).multiply(prob));
        }
      }
    }
    return returns;
  }
}
