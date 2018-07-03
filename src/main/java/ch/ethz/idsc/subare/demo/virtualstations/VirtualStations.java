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
import ch.ethz.idsc.tensor.pdf.PDF;
import ch.ethz.idsc.tensor.pdf.PoissonDistribution;
import ch.ethz.idsc.tensor.red.Total;
import ch.ethz.idsc.tensor.sca.Sign;

public class VirtualStations implements MonteCarloInterface {
  private static final int NVNODES = 3;
  private static final int VEHICLES = 30;
  private static final int VEHICLESSENT = 3;
  private static final int TIMEINTERVALS = 24;
  private static final int TOTALTIME = 24 * 60;
  // ---
  private static final Scalar REBALANCE_COST = RealScalar.of(-1);
  private static final Scalar AVAILABILITY_COST = RealScalar.of(-10);
  private static final Scalar TAXI_ARRIVAL_PROB = RealScalar.of(0.5); // assuming a fluidic model
  private static final Scalar CUSTOMER_ARRIVAL_RATE = RealScalar.of(0.5);
  private final PDF pdf_customer = PDF.of(PoissonDistribution.of(CUSTOMER_ARRIVAL_RATE));
  // ---
  private final Tensor states;
  private final Map<Integer, Map<Integer, Scalar>> exactStateMap = new HashMap<>();
  private final Map<Integer, Map<Integer, Integer>> linkToIndex = new HashMap<>();

  public VirtualStations() {
    states = generateStates().unmodifiable();
    generateExactStates();
    generateLinkMap();
  }

  private Tensor generateStates() {
    Tensor prefixes = Tensors.empty();
    for (int t = 0; t < TIMEINTERVALS; ++t) {
      prefixes.append(Tensors.vector(t));
    }
    Tensor states = binaryVectors(NVNODES, prefixes);
    states.append(Tensors.vector(0)); // terminal state
    return states;
  }

  private void generateExactStates() {
    GlobalAssert.that(VEHICLES % NVNODES == 0);
    for (int i = 0; i < NVNODES; ++i) {
      Map<Integer, Scalar> subMap = new HashMap<>();
      for (int j = 0; j < NVNODES; ++j) {
        if (j == i) {
          subMap.put(j, RealScalar.of(VEHICLES / NVNODES));
        } else {
          subMap.put(j, RealScalar.ZERO);
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
    Tensor prefix = Tensors.empty();
    for (int i = 0; i < NVNODES; ++i) {
      if (Sign.isPositive(state.Get(i + 1))) {
        prefix = binaryVectors(NVNODES, prefix);
      } else {
        //prefix = Join.of(prefix, Tensors.vector(v -> RealScalar.ZERO, NVNODES));
        prefix = zeroVectors(NVNODES, prefix);
      }
    }
    return prefix;
  }

  /** returns the tensor of all possible binary combinations in a vector of size length
   * 
   * @param length
   * @param prefixes
   * @return */
  private Tensor binaryVectors(int length, Tensor prefixes) {
    if (length == 0)
      return prefixes;
    if (prefixes.length() == 0) {
      return binaryVectors(length - 1, Tensors.of(Tensors.vector(1), Tensors.vector(0)));
    }
    Tensor extension = Tensors.empty();
    for (Tensor prefix : prefixes) {
      extension.append(Join.of(prefix, Tensors.vector(1)));
      extension.append(Join.of(prefix, Tensors.vector(0)));
    }
    return binaryVectors(length - 1, extension);
  }
  
  private Tensor zeroVectors(int length, Tensor prefixes) {
    if (length == 0)
      return prefixes;
    if (prefixes.length() == 0) {
      return zeroVectors(length - 1, Tensors.of(Tensors.vector(1), Tensors.vector(0)));
    }
    Tensor extension = Tensors.empty();
    for (Tensor prefix : prefixes) {
      extension.append(Join.of(prefix, Tensors.vector(0)));
    }
    return zeroVectors(length - 1, extension);
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    // for(int i=0;i<NVNODES;++i) {
    // for(int j=0;j<NVNODES;++j) {
    //
    // exactStateMap.get(i).get(j)
    // }
    // }
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    Scalar availabilityCost = RealScalar.of(NVNODES).subtract(Total.of(next.extract(1, next.length())).Get()).multiply(AVAILABILITY_COST);
    Scalar rebalancingCost = Total.of(action).Get().multiply(REBALANCE_COST);
    return availabilityCost.add(rebalancingCost);
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return state.length() == 1;
  }

  @Override
  public Tensor startStates() {
    return states.extract(0, 1);
  }

  public static void main(String[] args) {    
    VirtualStations vs = new VirtualStations();
    Tensor state = vs.states().get(3);
    System.out.println(state);
    System.out.println(vs.actions(state));
  }
}
