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
import ch.ethz.idsc.tensor.pdf.BernoulliDistribution;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.red.Total;

/** A two node problem with an airport and a center. Passengers arrive at the airport and can be driven to
 * the center by taxis. The taxis don't know in advance if there are passengers to pick up when they move
 * from the airport to the center. Driving without passenger to the other node costs 10CHF. Driving from
 * the airport to the center with a customers gives 30CHF reward instead. Parking at the airport for one
 * time step at the airport costs 5CHF. */
class Airport implements StandardModel, MonteCarloInterface {
  private static final int LASTT = 3;
  private static final int VEHICLES = 5;
  private static final Scalar REBALANCE_COST = RealScalar.of(10);
  private static final Scalar AIRPORT_WAIT_COST = RealScalar.of(5);
  private static final Scalar CUSTOMER_REWARD = RealScalar.of(30);
  private final Tensor states;
  private Random random = new Random();
  private static final Tensor CUSTOMER_PROB = Tensors.vectorDouble(0.4, 0.2, 0.1, 0.3); // i.e. CUSTOMER_PROB.Get(0) is the probability that no customer is
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
    return isTerminal(state) ? RealScalar.ZERO : RealScalar.ZERO; // TODO
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

    if (state.Get(0).equals(RealScalar.of(2)) && state.Get(2).equals(RealScalar.of(3)))
      return RealScalar.ONE;
    return RealScalar.ZERO; // TODO
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
