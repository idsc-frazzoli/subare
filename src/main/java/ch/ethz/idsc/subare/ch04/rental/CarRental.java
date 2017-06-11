// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.rental;

import ch.ethz.idsc.subare.core.SampleModel;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.util.DiscreteDistributions;
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
import ch.ethz.idsc.tensor.red.Total;
import ch.ethz.idsc.tensor.sca.Clip;

/** Example 4.2: Jack's Car Rental
 * Figure 4.2
 * 
 * p.87-88
 * 
 * states: number of cars at the 2 stations in the evening
 * actions: number of cars moved between the 2 stations during the night
 * the action is encoded as a 2-vector {+n, -n}
 * 
 * [no further references are provided in the book] */
class CarRental implements StandardModel, SampleModel {
  private static final int MAX_MOVE_OF_CARS = 5;
  private static final int RENTAL_REQUEST_FIRST_LOC = 3;
  private static final int RENTAL_REQUEST_SECOND_LOC = 4;
  private static final int RETURN_FIRST_LOC = 3;
  private static final int RETURN_SECOND_LOC = 2;
  private static final Scalar RENTAL_CREDIT = RealScalar.of(10);
  private static final Scalar MOVE_CAR_COST = RealScalar.of(-2);
  // ---
  private final Tensor states;
  private final PoissonDistribution p1out = PoissonDistribution.of(RealScalar.of(RENTAL_REQUEST_FIRST_LOC));
  private final PoissonDistribution p1_in = PoissonDistribution.of(RealScalar.of(RETURN_FIRST_LOC));
  private final PoissonDistribution p2out = PoissonDistribution.of(RealScalar.of(RENTAL_REQUEST_SECOND_LOC));
  private final PoissonDistribution p2_in = PoissonDistribution.of(RealScalar.of(RETURN_SECOND_LOC));
  // ---
  private final Clip CLIP;

  public CarRental(int maxCars) {
    CLIP = Clip.function(0, maxCars);
    states = Flatten.of(Array.of(Tensors::vector, maxCars + 1, maxCars + 1), 1).unmodifiable();
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

  Tensor night_move(Tensor state, Tensor action) {
    Tensor next = state.add(Tensors.of(action, action.negate()));
    if (Scalars.lessThan(next.Get(0), RealScalar.ZERO))
      throw new RuntimeException();
    if (Scalars.lessThan(next.Get(1), RealScalar.ZERO))
      throw new RuntimeException();
    return next;
  }

  /**************************************************/
  @Override
  public Tensor move(Tensor state, Tensor action) {
    Tensor morning = night_move(state, action);
    Scalar n1_in = DiscreteDistributions.nextSample(p1_in);
    Scalar n1out = DiscreteDistributions.nextSample(p1out);
    Scalar n2_in = DiscreteDistributions.nextSample(p2_in);
    Scalar n2out = DiscreteDistributions.nextSample(p2out);
    morning.set(cars -> cars.add(n1_in.subtract(n1out)), 0);
    morning.set(cars -> cars.add(n2_in.subtract(n2out)), 1);
    return morning.map(CLIP);
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    Scalar sum = action.Get().abs().multiply(MOVE_CAR_COST);
    // TODO why doesn't next appear in function and morning is not used?
    Tensor morning = night_move(state, action);
    Scalar n1out = DiscreteDistributions.nextSample(p1out);
    Scalar n2out = DiscreteDistributions.nextSample(p2out);
    n1out = Min.of(n1out, morning.Get(0)); // TODO does not account for returned vehicles?
    n2out = Min.of(n2out, morning.Get(1));
    System.out.println(n1out.add(n2out));
    Scalar rented = n1out.add(n2out);
    return sum.add(rented.multiply(RENTAL_CREDIT));
  }

  /**************************************************/
  @Override
  public Scalar expectedReward(Tensor state, Tensor action) {
    Scalar sum = RealScalar.ZERO;
    for (Tensor next : transitions(state, action))
      sum = sum.add(expectedReward(state, action, next));
    return sum;
  }

  static final int LOOK = 5;

  Scalar expectedReward(Tensor state, Tensor action, Tensor next) {
    Scalar sum = action.Get().abs().multiply(MOVE_CAR_COST);
    Tensor morning = night_move(state, action);
    Tensor delta = next.subtract(morning); // cars that have to be pop-up and disappear through the random process
    // Scalar prob = RealScalar.ZERO;
    final int d0 = (int) delta.Get(0).number();
    final int d1 = (int) delta.Get(1).number();
    final int ofs0 = Math.max(0, -d0);
    final int ofs1 = Math.max(0, -d1);
    for (int req0 = ofs0; req0 < ofs0 + LOOK; ++req0) {
      final int returns0 = req0 + d0;
      final int request0 = req0;
      if (returns0 - request0 != d0)
        throw new RuntimeException();
      for (int req1 = ofs1; req1 < ofs1 + LOOK; ++req1) {
        final int returns1 = req1 + d1;
        final int request1 = req1;
        if (returns1 - request1 != d1)
          throw new RuntimeException();
        // System.out.println(Tensors.vector(returns0, -request0, returns1, -request1));
        Scalar prob = Total.prod(Tensors.of( //
            p1_in.probabilityEquals(returns0), // returns (added)
            p1out.probabilityEquals(request0), // rental requests (subtracted)
            p2_in.probabilityEquals(returns1), // returns (added)
            p2out.probabilityEquals(request1) // rental requests (subtracted)
        )).Get();
        sum = sum.add(prob.multiply(RealScalar.of(request0 + request1).multiply(RENTAL_CREDIT)));
      }
    }
    return sum;
  }

  @Override
  public Tensor transitions(Tensor state, Tensor action) {
    return states();
  }

  @Override
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    Tensor morning = night_move(state, action);
    Tensor delta = next.subtract(morning); // cars that have to be pop-up and disappear through the random process
    Scalar prob = RealScalar.ZERO;
    final int d0 = (int) delta.Get(0).number();
    final int d1 = (int) delta.Get(1).number();
    final int ofs0 = Math.max(0, -d0);
    final int ofs1 = Math.max(0, -d1);
    for (int req0 = ofs0; req0 < ofs0 + LOOK; ++req0) {
      final int returns0 = req0 + d0;
      final int request0 = req0;
      if (returns0 - request0 != d0)
        throw new RuntimeException();
      for (int req1 = ofs1; req1 < ofs1 + LOOK; ++req1) {
        final int returns1 = req1 + d1;
        final int request1 = req1;
        if (returns1 - request1 != d1)
          throw new RuntimeException();
        // System.out.println(Tensors.vector(returns0, -request0, returns1, -request1));
        prob = prob.add(Total.prod(Tensors.of( //
            p1_in.probabilityEquals(returns0), // returns (added)
            p1out.probabilityEquals(request0), // rental requests (subtracted)
            p2_in.probabilityEquals(returns1), // returns (added)
            p2out.probabilityEquals(request1) // rental requests (subtracted)
        )));
      }
    }
    return prob;
  }
}
