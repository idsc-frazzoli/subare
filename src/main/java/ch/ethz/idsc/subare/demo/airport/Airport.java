// code by fluric
package ch.ethz.idsc.subare.demo.airport;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.nrm.NormalizeTotal;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.EmpiricalDistribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;
import ch.ethz.idsc.tensor.red.Min;
import ch.ethz.idsc.tensor.red.Total;
import ch.ethz.idsc.tensor.sca.Chop;
import ch.ethz.idsc.tensor.sca.Ramp;

/** A two node problem with an airport and a center. Passengers arrive at the airport and can be driven to
 * the center by taxis. The taxis don't know in advance if there are passengers to pick up when they move
 * from the airport to the center. Driving without passenger to the other node costs 10CHF. Driving from
 * the airport to the center with a customers gives 30CHF reward instead. Parking at the airport for one
 * time step at the airport costs 5CHF. */
public class Airport implements StandardModel, MonteCarloInterface {
  static final int LASTT = 4;
  static final int VEHICLES = 5;
  private static final Scalar REBALANCE_COST = RealScalar.of(-10);
  private static final Scalar AIRPORT_WAIT_COST = RealScalar.of(-5);
  private static final Scalar CUSTOMER_REWARD = RealScalar.of(30);
  // i.e. CUSTOMER_PROB.Get(0) is the probability that 0 customers are waiting
  private static final Tensor CUSTOMER_HIST = Tensors.vector(1, 2, 4, 3);
  private static final Tensor CUSTOMER_PROB = NormalizeTotal.FUNCTION.apply(CUSTOMER_HIST);
  // for EmpiricalDistribution#fromUnscaledPDF the numbers don't have to add up to 1
  private static final Distribution DISTRIBUTION = EmpiricalDistribution.fromUnscaledPDF(CUSTOMER_HIST);
  public static final Airport INSTANCE = new Airport();
  // ---
  private final Tensor states;

  // TODO defined parameters for complexity of scenario: # time steps, # taxis ...
  private Airport() {
    Tensor states = Tensors.empty();
    states.append(Tensors.vector(0, 5, 0)); // start at time 0 with 5 taxis in the city and 0 in the airport
    for (int t = 1; t <= LASTT; t++) {
      for (int v = 0; v <= VEHICLES; v++) {
        states.append(Tensors.vector(t, v, VEHICLES - v));
      }
    }
    Chop.NONE.requireClose(Total.of(states.get(0)), RealScalar.of(VEHICLES));
    this.states = states.unmodifiable();
  }

  @Override
  public Tensor states() {
    return states;
  }

  /** @return possible stakes */
  @Override
  public Tensor actions(Tensor state) {
    if (isTerminal(state))
      return Tensors.of(Tensors.of(RealScalar.ZERO));
    Tensor actions = Tensors.empty();
    int c1 = Scalars.intValueExact(state.Get(1));
    int c2 = Scalars.intValueExact(state.Get(2));
    for (int i = 0; i <= c1; ++i)
      for (int j = 0; j <= c2; ++j)
        actions.append(Tensors.vector(i, j));
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
      Chop.NONE.requireClose(action, Tensors.of(RealScalar.ZERO));
      return state;
    }
    Scalar delta = action.Get(0).subtract(action.Get(1));
    Tensor shift = Tensors.of(RealScalar.ONE, delta.negate(), delta);
    return state.add(shift);
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) { // deterministic
    if (isTerminal(state))
      return RealScalar.ZERO;
    Scalar customers = RandomVariate.of(DISTRIBUTION); // either 0, 1, 2, 3
    // deal with rebalancing costs
    Scalar reward = action.Get(0).multiply(REBALANCE_COST);
    reward = reward.add(Ramp.of(action.Get(1).subtract(customers)).multiply(REBALANCE_COST));
    // deal with parking cost of airport
    reward = reward.add(state.Get(2).subtract(action.Get(1)).multiply(AIRPORT_WAIT_COST));
    // deal with customer reward
    reward = reward.add(Min.of(customers, action.Get(1)).multiply(CUSTOMER_REWARD));
    return reward;
  }

  /**************************************************/
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return states.extract(0, 1);
  }

  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return state.Get(0).equals(RealScalar.of(LASTT));
  }

  /**************************************************/
  @Override
  public Scalar expectedReward(Tensor state, Tensor action) {
    if (isTerminal(state))
      return RealScalar.ZERO;
    // deal with rebalancing costs
    Scalar reward = action.Get(0).multiply(REBALANCE_COST);
    for (int i = 0; i < CUSTOMER_PROB.length(); i++) {
      reward = reward.add(Ramp.of(action.Get(1).subtract(RealScalar.of(i))).multiply(REBALANCE_COST).multiply(CUSTOMER_PROB.Get(i)));
    }
    // deal with parking cost of airport
    reward = reward.add(state.Get(2).subtract(action.Get(1)).multiply(AIRPORT_WAIT_COST));
    // deal with customer reward
    Scalar rewardCustomers = (Scalar) Range.of(0, CUSTOMER_PROB.length()) //
        .map(Min.function(action.Get(1))) //
        .dot(CUSTOMER_PROB) //
        .multiply(CUSTOMER_REWARD);
    reward = reward.add(rewardCustomers);
    return reward;
  }

  @Override
  public Tensor transitions(Tensor state, Tensor action) {
    return Tensors.of(move(state, action));
  }

  @Override
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    if (isTerminal(state)) {
      Chop.NONE.requireClose(move(state, action), next);
      return RealScalar.ONE;
    }
    if (move(state, action).equals(next))
      return RealScalar.ONE;
    return RealScalar.ZERO;
  }
}
