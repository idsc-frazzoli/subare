// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.rental;

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
 * Figure 4.2
 * 
 * p.87-88
 * 
 * states: number of cars at the 2 stations in the evening
 * actions: number of cars moved between the 2 stations during the night
 * the action is encoded as a 2-vector {+n, -n}
 * 
 * [no further references are provided in the book] */
class CarRental implements StandardModel {
  private static final int MAX_CARS = 20;
  private static final int MAX_MOVE_OF_CARS = 5;
  private static final int RENTAL_REQUEST_FIRST_LOC = 3;
  private static final int RENTAL_REQUEST_SECOND_LOC = 4;
  private static final int RETURN_FIRST_LOC = 3;
  private static final int RETURN_SECOND_LOC = 2;
  private static final Scalar RENTAL_CREDIT = RealScalar.of(10);
  private static final Scalar MOVE_CAR_COST = RealScalar.of(-2);
  private static final Clip CLIP = Clip.function(0, MAX_CARS);
  private static final int POISSON_UP_BOUND = 10;
  private final boolean constantReturnedCars;
  // ---
  private static final Tensor states = Flatten.of(Array.of(Tensors::vector, 21, 21), 1).unmodifiable();
  private final PoissonDistribution p1out = PoissonDistribution.of(RealScalar.of(RENTAL_REQUEST_FIRST_LOC));
  private final PoissonDistribution p1_in = PoissonDistribution.of(RealScalar.of(RETURN_FIRST_LOC));
  private final PoissonDistribution p2out = PoissonDistribution.of(RealScalar.of(RENTAL_REQUEST_SECOND_LOC));
  private final PoissonDistribution p2_in = PoissonDistribution.of(RealScalar.of(RETURN_SECOND_LOC));

  public CarRental(boolean constantReturnedCars) {
    this.constantReturnedCars = constantReturnedCars;
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

  public Tensor night_move(Tensor state, Tensor action) {
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
    Scalar returns = MOVE_CAR_COST.multiply(((Scalar) action).abs());
    // go through all possible rental requests
    for (int rentalRequest1Loc = 0; rentalRequest1Loc <= POISSON_UP_BOUND; ++rentalRequest1Loc) {
      for (int rentalRequest2Loc = 0; rentalRequest2Loc <= POISSON_UP_BOUND; ++rentalRequest2Loc) {
        Scalar rentalRequest1LocS = RealScalar.of(rentalRequest1Loc);
        Scalar rentalRequest2LocS = RealScalar.of(rentalRequest2Loc);
        // moving cars
        Tensor numOfCarsNext = night_move(state, action);
        // valid rental requests should be less equals actual # of cars
        Scalar realRental1Loc = Min.of(numOfCarsNext.Get(0), rentalRequest1LocS);
        Scalar realRental2Loc = Min.of(numOfCarsNext.Get(1), rentalRequest2LocS);
        // credits for renting
        Scalar reward = realRental1Loc.add(realRental2Loc).multiply(RENTAL_CREDIT);
        numOfCarsNext = numOfCarsNext.subtract(Tensors.of(realRental1Loc, realRental2Loc));
        // probability for current combination of rental requests
        Scalar prob = p1out.apply(rentalRequest1Loc).multiply(p2out.apply(rentalRequest2Loc));
        if (constantReturnedCars) {
          Scalar returnedCars1Loc = p1_in.lambda();
          Scalar returnedCars2Loc = p2_in.lambda();
          numOfCarsNext = numOfCarsNext.add(Tensors.of(returnedCars1Loc, returnedCars2Loc)).map(CLIP);
          returns = returns.add( //
              reward.add(gvalues.value(numOfCarsNext)).multiply(prob) //
          );
        } else {
          Tensor numOfCarsCopy = numOfCarsNext.copy();
          Scalar prob_ = prob;
          for (int returnedCars1Loc = 0; returnedCars1Loc <= POISSON_UP_BOUND; ++returnedCars1Loc) {
            for (int returnedCars2Loc = 0; returnedCars2Loc <= POISSON_UP_BOUND; ++returnedCars2Loc) {
              numOfCarsNext = numOfCarsCopy.copy();
              // prob = prob_;
              numOfCarsNext = numOfCarsNext.add(Tensors.vector(returnedCars1Loc, returnedCars2Loc)).map(CLIP);
              prob = p1_in.apply(returnedCars1Loc).multiply(p2_in.apply(returnedCars2Loc)).multiply(prob_);
              returns = returns.add( //
                  reward.add(gvalues.value(numOfCarsNext)).multiply(prob) //
              );
            }
          }
        }
      }
    }
    return returns;
  }
}
