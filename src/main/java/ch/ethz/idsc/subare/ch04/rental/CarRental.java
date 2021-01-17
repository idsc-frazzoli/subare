// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.rental;

import ch.ethz.idsc.subare.core.SampleModel;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.PDF;
import ch.ethz.idsc.tensor.pdf.PoissonDistribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;
import ch.ethz.idsc.tensor.red.Min;
import ch.ethz.idsc.tensor.red.Times;
import ch.ethz.idsc.tensor.sca.Abs;
import ch.ethz.idsc.tensor.sca.Clip;
import ch.ethz.idsc.tensor.sca.Clips;
import ch.ethz.idsc.tensor.sca.Sign;

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
/* package */ class CarRental implements StandardModel, SampleModel {
  private static final int MAX_MOVE_OF_CARS = 5;
  private static final int RENTAL_REQUEST_FIRST_LOC = 3;
  private static final int RENTAL_REQUEST_SECOND_LOC = 4;
  private static final int RETURN_FIRST_LOC = 3;
  private static final int RETURN_SECOND_LOC = 2;
  private static final Scalar RENTAL_CREDIT = RealScalar.of(10);
  private static final Scalar MOVE_CAR_COST = RealScalar.of(-2);
  // ---
  private final Tensor states;
  private final PDF p1out = PDF.of(PoissonDistribution.of(RealScalar.of(RENTAL_REQUEST_FIRST_LOC)));
  private final PDF p1_in = PDF.of(PoissonDistribution.of(RealScalar.of(RETURN_FIRST_LOC)));
  private final PDF p2out = PDF.of(PoissonDistribution.of(RealScalar.of(RENTAL_REQUEST_SECOND_LOC)));
  private final PDF p2_in = PDF.of(PoissonDistribution.of(RealScalar.of(RETURN_SECOND_LOC)));
  // ---
  private final Distribution d1out = PoissonDistribution.of(RealScalar.of(RENTAL_REQUEST_FIRST_LOC));
  private final Distribution d1_in = PoissonDistribution.of(RealScalar.of(RETURN_FIRST_LOC));
  private final Distribution d2out = PoissonDistribution.of(RealScalar.of(RENTAL_REQUEST_SECOND_LOC));
  private final Distribution d2_in = PoissonDistribution.of(RealScalar.of(RETURN_SECOND_LOC));
  // ---
  private final Clip CLIP;
  final int maxCars;

  public CarRental(int maxCars) {
    this.maxCars = maxCars;
    CLIP = Clips.positive(maxCars);
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
    Sign.requirePositiveOrZero(next.Get(0));
    Sign.requirePositiveOrZero(next.Get(1));
    return next;
  }

  /**************************************************/
  @Override
  public Tensor move(Tensor state, Tensor action) {
    Tensor morning = night_move(state, action);
    Scalar n1_in = RandomVariate.of(d1_in);
    Scalar n1out = RandomVariate.of(d1out);
    Scalar n2_in = RandomVariate.of(d2_in);
    Scalar n2out = RandomVariate.of(d2out);
    return effective(morning, n1_in, n1out, n2_in, n2out);
  }

  private Tensor effective(Tensor morning, Scalar n1_in, Scalar n1out, Scalar n2_in, Scalar n2out) {
    morning = morning.copy();
    morning.set(cars -> cars.add(n1_in.subtract(n1out)), 0);
    morning.set(cars -> cars.add(n2_in.subtract(n2out)), 1);
    return morning.map(CLIP);
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    Scalar sum = Abs.FUNCTION.apply((Scalar) action).multiply(MOVE_CAR_COST);
    Tensor morning = night_move(state, action).unmodifiable();
    Scalar n1_in;
    Scalar n1out = null;
    Scalar n2_in;
    Scalar n2out = null;
    boolean status = false;
    int attempts = 0;
    // TODO this is very inefficient for trajectory generation...
    while (!status) {
      if (200 < attempts) {
        System.out.println("warning: give up");
        return sum;
      }
      n1_in = RandomVariate.of(d1_in);
      n1out = RandomVariate.of(d1out);
      n2_in = RandomVariate.of(d2_in);
      n2out = RandomVariate.of(d2out);
      status = effective(morning, n1_in, n1out, n2_in, n2out).equals(next);
      ++attempts;
    }
    // System.out.println("attempts=" + attempts);
    n1out = Min.of(n1out, morning.Get(0));
    n2out = Min.of(n2out, morning.Get(1));
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
    Scalar sum = Abs.FUNCTION.apply((Scalar) action).multiply(MOVE_CAR_COST);
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
        Scalar prob = Times.of( //
            p1_in.at(RealScalar.of(returns0)), // returns (added)
            p1out.at(RealScalar.of(request0)), // rental requests (subtracted)
            p2_in.at(RealScalar.of(returns1)), // returns (added)
            p2out.at(RealScalar.of(request1)) // rental requests (subtracted)
        );
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
        prob = prob.add(Times.of( //
            p1_in.at(RealScalar.of(returns0)), // returns (added)
            p1out.at(RealScalar.of(request0)), // rental requests (subtracted)
            p2_in.at(RealScalar.of(returns1)), // returns (added)
            p2out.at(RealScalar.of(request1)) // rental requests (subtracted)
        ));
      }
    }
    return prob;
  }
}
