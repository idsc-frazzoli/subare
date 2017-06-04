// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Extract;

/** p.33 */
public class EGreedyPolicy implements PolicyInterface {
  // this simplicity may be the reason why q(s,a) is preferred over v(s)
  public static PolicyInterface bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, Scalar epsilon) {
    Map<Tensor, Index> map = new HashMap<>();
    Map<Tensor, Integer> sizes = new HashMap<>();
    for (Tensor state : discreteModel.states()) {
      Tensor actions = discreteModel.actions(state);
      Tensor va = Tensor.of(actions.flatten(0).map(action -> qsa.value(state, action)));
      FairArgMax fairArgMax = FairArgMax.of(va);
      Tensor feasible = Extract.of(actions, fairArgMax.options());
      map.put(state, Index.build(feasible));
      sizes.put(state, actions.length());
    }
    return new EGreedyPolicy(map, epsilon, sizes);
  }

  private final Map<Tensor, Index> map;
  /** probability of choosing a non-optimal action, if there is at least one non-optimal action */
  private final Scalar epsilon;
  private final Map<Tensor, Integer> sizes;

  EGreedyPolicy(Map<Tensor, Index> map, Scalar epsilon, Map<Tensor, Integer> sizes) {
    this.map = map;
    this.epsilon = epsilon;
    this.sizes = sizes;
    if (sizes == null && Scalars.nonZero(epsilon))
      throw new RuntimeException("sizes invalid for " + epsilon);
  }

  @Override
  public Scalar policy(Tensor state, Tensor action) {
    Index index = map.get(state);
    final int optimalCount = index.size();
    if (sizes == null) // greedy
      return index.containsKey(action) ? RationalScalar.of(1, optimalCount) : RealScalar.ZERO;
    // ---
    final int nonOptimalCount = sizes.get(state) - optimalCount;
    if (nonOptimalCount == 0) // no non-optimal action exists
      return RationalScalar.of(1, optimalCount);
    if (index.containsKey(action))
      return RealScalar.ONE.subtract(epsilon).divide(RealScalar.of(optimalCount));
    return epsilon.divide(RealScalar.of(nonOptimalCount));
  }

  /** useful for export to Mathematica
   * 
   * @param states
   * @return list of actions optimal for */
  public Tensor flatten(Tensor states) {
    Tensor result = Tensors.empty();
    for (Tensor state : states)
      for (Tensor action : map.get(state).keys())
        result.append(Tensors.of(state, action));
    return result;
  }

  /** print overview of possible actions for given states in console
   * 
   * @param states */
  public void print(Tensor states) {
    System.out.println("greedy:");
    for (Tensor state : states)
      System.out.println(state + " -> " + map.get(state).keys());
  }
}
