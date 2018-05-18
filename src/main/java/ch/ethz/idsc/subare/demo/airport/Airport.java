// code fluric
package ch.ethz.idsc.subare.demo.airport;

import java.util.Random;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.red.Min;
import ch.ethz.idsc.tensor.red.Total;

/** A two node problem with an airport and a center. Passengers arrive at the airport and can be driven to
 * the center by taxis. The taxis don't know in advance if there are passengers to pick up when they move
 * from the airport to the center. Driving without passenger to the other node costs 10CHF. Driving from
 * the airport to the center with a customers gives 30CHF reward instead. Parking at the airport for one
 * time step at the airport costs 5CHF. */
class Airport implements StandardModel, MonteCarloInterface {
  static final int LASTT = 4;
  static final int VEHICLES = 5;
  private static final Scalar REBALANCE_COST = RealScalar.of(-10);
  private static final Scalar AIRPORT_WAIT_COST = RealScalar.of(-5);
  private static final Scalar CUSTOMER_REWARD = RealScalar.of(30);
  private final Tensor states;
  private Random random = new Random();
  private static final Tensor CUSTOMER_PROB = Tensors.vectorDouble(0.1, 0.2, 0.4, 0.3); // i.e. CUSTOMER_PROB.Get(0) is the probability that 0 customers are
                                                                                        // waiting

  public Airport() {
    states = Tensors.empty();
    states.append(Tensors.vector(0, 5, 0)); // start at time 0 with 5 taxis in the city and 0 in the airport
    for (int t = 1; t <= LASTT; t++) {
      for (int v = 0; v <= VEHICLES; v++) {
        states.append(Tensors.vector(t, v, VEHICLES - v));
      }
    }
    GlobalAssert.that(Total.of(CUSTOMER_PROB).equals(RealScalar.of(1.0)));
    GlobalAssert.that(Total.of(states.get(0)).equals(RealScalar.of(VEHICLES)));
  }

  @Override
  public Tensor states() {
    return states;
  }

  /** @return possible stakes */
  @Override
  public Tensor actions(Tensor state) {
    if (isTerminal(state))
      return Tensors.of(RealScalar.ZERO);
    Tensor actions = Tensors.empty();
    for (int i = 0; i <= state.Get(1).number().intValue(); i++) {
      for (int j = 0; j <= state.Get(2).number().intValue(); j++) {
        actions.append(Tensors.vector(i, j));
      }
    }
    return actions;
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  /**************************************************/
  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state)) {
      GlobalAssert.that(action.equals(RealScalar.ZERO));
      return state;
    }
    return Tensors.vector(state.Get(0).add(RealScalar.ONE).number(), state.Get(1).subtract(action.Get(0)).add(action.Get(1)).number(),
        state.Get(2).subtract(action.Get(1)).add(action.Get(0)).number());
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) { // deterministic
    if (isTerminal(state))
      return RealScalar.ZERO;
    // get the random variate of the number of customers
    double outcome = random.nextDouble();
    int customers = -1;
    while (outcome > 0.0) {
      customers++;
      outcome -= CUSTOMER_PROB.Get(customers).number().doubleValue();
    }
    // deal with rebalancing costs
    Scalar reward = action.Get(0).multiply(REBALANCE_COST);
    reward = reward.add(Max.of(RealScalar.ZERO, action.Get(1).subtract(RealScalar.of(customers))).multiply(REBALANCE_COST));
    // deal with parking cost of airport
    reward = reward.add(move(state, action).Get(2).multiply(AIRPORT_WAIT_COST));
    // deal with customer reward
    reward = reward.add(Min.of(RealScalar.of(customers), action.Get(1)).multiply(CUSTOMER_REWARD));
    return reward;
  }

  /**************************************************/
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return states.extract(0, 1);
  }

  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return state.get(0).equals(RealScalar.of(LASTT));
  }

  /**************************************************/
  @Override
  public Scalar expectedReward(Tensor state, Tensor action) {
    if (isTerminal(state))
      return RealScalar.ZERO;
    // deal with rebalancing costs
    Scalar reward = action.Get(0).multiply(REBALANCE_COST);
    for (int i = 0; i < CUSTOMER_PROB.length(); i++) {
      reward = reward.add(Max.of(RealScalar.ZERO, action.Get(1).subtract(RealScalar.of(i))).multiply(REBALANCE_COST).multiply(CUSTOMER_PROB.Get(i)));
    }
    // deal with parking cost of airport
    reward = reward.add(move(state, action).Get(2).multiply(AIRPORT_WAIT_COST));
    // deal with customer reward
    for (int i = 0; i < CUSTOMER_PROB.length(); i++) {
      reward = reward.add(Min.of(RealScalar.of(i), action.Get(1)).multiply(CUSTOMER_REWARD).multiply(CUSTOMER_PROB.Get(i)));
    }
    return reward;
  }

  @Override
  public Tensor transitions(Tensor state, Tensor action) {
    return Tensors.of(move(state, action));
  }

  @Override
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    if (isTerminal(state)) {
      GlobalAssert.that(move(state, action).equals(next));
      return RealScalar.ONE;
    }
    if (move(state, action).equals(next))
      return RealScalar.ONE;
    return RealScalar.ZERO;
  }
}
