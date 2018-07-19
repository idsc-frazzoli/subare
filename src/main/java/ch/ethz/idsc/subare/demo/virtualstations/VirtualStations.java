// code by fluric
package ch.ethz.idsc.subare.demo.virtualstations;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.pdf.BernoulliDistribution;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.PoissonDistribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;
import ch.ethz.idsc.tensor.red.Total;
import ch.ethz.idsc.tensor.sca.Sign;

/** Toy example for AMOD system. There are 3 virtual stations where taxis are travelling between those.
 * The goal is to have at least 1 taxi in each virtual station so that the availability is given. Customers arrive
 * at the virtual stations according to a given Poisson distribution and will be driven to another virtual station.
 * The actions control the rebalancing of the taxis among the virtual stations. The arrival time is approximated
 * by a fluidic assumption of the vehicles travelling in between (stochastic and linear to the amount of taxis
 * travelling from virtual station i to j) */
public class VirtualStations implements MonteCarloInterface {
  private static final int NVNODES = 3;
  private static final int VEHICLES = 30;
  private static final int VEHICLESSENT = 3;
  private static final int TIMEINTERVALS = 24;
  private static final int TOTALTIME = 24;
  private static final int INTERVALTIME = TOTALTIME / TIMEINTERVALS;
  // ---
  private static final Scalar REBALANCE_COST = RealScalar.of(-1);
  private static final Scalar AVAILABILITY_COST = RealScalar.of(-10);
  private static final Scalar TAXI_ARRIVAL_PROB = RealScalar.of(0.5); // assuming a fluidic model
  private static final Scalar CUSTOMER_ARRIVAL_RATE = RealScalar.of(0.5);
  // ---
  private final Distribution customer_distribution = PoissonDistribution.of(CUSTOMER_ARRIVAL_RATE.multiply(RealScalar.of(INTERVALTIME)));
  private final Distribution arrival_distribution = BernoulliDistribution.of(TAXI_ARRIVAL_PROB);
  // ---
  private final Tensor states;
  private final Map<Integer, Map<Integer, Integer>> exactStateMap = new HashMap<>();
  private final Map<Integer, Map<Integer, Integer>> linkToIndex = new HashMap<>();

  public VirtualStations() {
    states = generateStates().unmodifiable();
    generateExactState();
    generateLinkMap();
  }

  private static Tensor generateStates() {
    Tensor prefixes = Tensors.empty();
    for (int t = 0; t <= TIMEINTERVALS; ++t) {
      prefixes.append(Tensors.vector(t));
    }
    Tensor states = StaticHelper.binaryVectors(NVNODES, prefixes);
    return states;
  }

  private void generateExactState() {
    GlobalAssert.that(VEHICLES % NVNODES == 0);
    for (int i = 0; i < NVNODES; ++i) {
      Map<Integer, Integer> subMap = new HashMap<>();
      for (int j = 0; j < NVNODES; ++j) {
        if (j == i) {
          subMap.put(j, VEHICLES / NVNODES);
        } else {
          subMap.put(j, 0);
        }
      }
      exactStateMap.put(i, subMap);
    }
  }

  private void generateLinkMap() {
    int index = 0;
    for (int i = 0; i < NVNODES; ++i) {
      Map<Integer, Integer> subMap = new HashMap<>();
      for (int j = 0; j < NVNODES; ++j) {
        if (i == j)
          continue;
        subMap.put(j, index);
        ++index;
      }
      linkToIndex.put(i, subMap);
    }
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    if (isTerminal(state))
      return Tensors.of(Tensors.of(RealScalar.ZERO));
    // ---
    Tensor prefix = Tensors.empty();
    for (int i = 0; i < NVNODES; ++i) {
      if (Sign.isPositive(state.Get(i + 1))) {
        prefix = StaticHelper.binaryVectors(NVNODES - 1, prefix);
      } else {
        prefix = StaticHelper.zeroVectors(NVNODES - 1, prefix);
      }
    }
    return prefix;
  }

  private int getActionElement(Tensor action, int from, int to) {
    return action.Get(linkToIndex.get(from).get(to)).number().intValue();
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state)) {
      GlobalAssert.that(action.equals(Tensors.of(RealScalar.ZERO)));
      return state;
    }
    // move arriving taxis
    for (int i = 0; i < NVNODES; ++i) {
      for (int j = 0; j < NVNODES; ++j) {
        if (j == i)
          continue;
        int arrivals = Total.of(RandomVariate.of(arrival_distribution, exactStateMap.get(i).get(j))).Get().number().intValue();
        exactStateMap.get(i).put(j, exactStateMap.get(i).get(j) - arrivals);
        exactStateMap.get(j).put(j, exactStateMap.get(j).get(j) + arrivals);
      }
    }
    // System.out.println("After moving taxis: " + exactStateMap);
    // serve customers
    for (int i = 0; i < NVNODES; ++i) {
      for (int j = 0; j < NVNODES; ++j) {
        if (j == i)
          continue;
        int customers = RandomVariate.of(customer_distribution).Get().number().intValue();
        int served = Math.min(exactStateMap.get(i).get(i), customers);
        exactStateMap.get(i).put(i, exactStateMap.get(i).get(i) - served);
        exactStateMap.get(i).put(j, exactStateMap.get(i).get(j) + served);
      }
    }
    // System.out.println("After serving customers: " + exactStateMap);
    // execute action commands
    for (int i = 0; i < NVNODES; ++i) {
      for (int j = 0; j < NVNODES; ++j) {
        if (j == i)
          continue;
        int rebalanced = Math.min(exactStateMap.get(i).get(i), getActionElement(action, i, j) * VEHICLESSENT);
        exactStateMap.get(i).put(i, exactStateMap.get(i).get(i) - rebalanced);
        exactStateMap.get(i).put(j, exactStateMap.get(i).get(j) + rebalanced);
      }
    }
    // System.out.println("After executing action: " + exactStateMap);
    // read new state
    Tensor newState = Tensors.vector(i -> exactStateMap.get(i).get(i) > 0 ? RealScalar.ONE : RealScalar.ZERO, NVNODES);
    return Join.of(Tensors.vector(state.Get(0).number().intValue() + 1), newState);
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    Scalar availabilityCost = RealScalar.of(NVNODES).subtract(Total.of(next.extract(1, next.length())).Get()).multiply(AVAILABILITY_COST);
    Scalar rebalancingCost = Total.of(action).Get().multiply(REBALANCE_COST);
    return availabilityCost.add(rebalancingCost);
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return state.Get(0).equals(RealScalar.of(TIMEINTERVALS));
  }

  @Override
  public Tensor startStates() {
    return states.extract(0, 1);
  }

  public int getTimeIntervals() {
    return TIMEINTERVALS;
  }

  public int getNVnodes() {
    return NVNODES;
  }
}
